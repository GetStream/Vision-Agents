package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNotLeaseHolder is a runner settling a contact it no longer holds: another took it over
// after its lease ran out, or it was settled already.
var ErrNotLeaseHolder = errors.New("store: another runner holds this contact")

// Whether an answered dial's outcome is still to be read from its transcript.
const (
	classifyPending   = "pending"
	classifyDone      = "done"
	classifyAbandoned = "abandoned"
)

// Claimed is a contact a runner now holds the lease of, with its campaign.
type Claimed struct {
	Campaign Campaign
	Contact  Contact
}

// StartDueCampaigns starts the scheduled campaigns whose start time has come.
func (s *Store) StartDueCampaigns(ctx context.Context, now time.Time) error {
	_, err := s.db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", Running).
		Set("started_at = COALESCE(started_at, ?)", now).
		Where("state = ?", Scheduled).
		Where("start_at <= ?", now).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: start due campaigns: %w", err))
	}
	return nil
}

// ClaimDueContacts leases up to limit contacts that are due to be rung, to owner until now+lease.
// A campaign never has more contacts in progress than its concurrency, however many runners
// claim at once.
func (s *Store) ClaimDueContacts(ctx context.Context, owner string, now time.Time, lease time.Duration, limit int) ([]Claimed, error) {
	if owner == "" {
		return nil, stack.Wrap(errors.New("store: a claim needs an owner"))
	}
	if limit <= 0 {
		return nil, nil
	}

	var claimed []Claimed
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		// Holding the campaign's row is what keeps two runners from both counting the same
		// free slots: whoever holds it counts and claims, the other skips the campaign this
		// tick. At most limit campaigns are held, so the rest stay free for other runners.
		// The least busy go first, so one customer's big campaign does not starve another's.
		var campaigns []Campaign
		err := tx.NewSelect().Model(&campaigns).
			Where("cp.state = ?", Running).
			Where(`EXISTS (SELECT 1 FROM campaign_contacts cc WHERE cc.campaign_id = cp.id
				AND cc.stage = ? AND cc.next_attempt_at <= ?)`, Enrolled, now).
			Where(`cp.concurrency > (SELECT count(*) FROM campaign_contacts cc WHERE cc.campaign_id = cp.id
				AND cc.stage = ?)`, InProgress).
			OrderExpr(`(SELECT count(*) FROM campaign_contacts cc WHERE cc.campaign_id = cp.id
				AND cc.stage = ?)::float / cp.concurrency ASC`, InProgress).
			Order("cp.started_at ASC", "cp.id ASC").
			Limit(limit).
			For("UPDATE SKIP LOCKED").
			Scan(ctx)
		if err != nil {
			return fmt.Errorf("store: lock due campaigns: %w", err)
		}

		for index, campaign := range campaigns {
			remaining := limit - len(claimed)
			if remaining == 0 {
				break
			}
			// Each campaign left gets an even share of what is left, so what one cannot use
			// passes to the next.
			left := len(campaigns) - index
			share := (remaining + left - 1) / left
			inProgress, err := tx.NewSelect().Model((*Contact)(nil)).
				Where("campaign_id = ?", campaign.ID).
				Where("stage = ?", InProgress).
				Count(ctx)
			if err != nil {
				return fmt.Errorf("store: count calls in progress: %w", err)
			}
			free := min(campaign.Concurrency-inProgress, share)
			if free <= 0 {
				continue
			}

			var contacts []Contact
			err = tx.NewRaw(`WITH due AS (
	SELECT id FROM campaign_contacts
	WHERE campaign_id = ? AND stage = ? AND next_attempt_at <= ?
	ORDER BY next_attempt_at, seq
	LIMIT ?
	FOR UPDATE SKIP LOCKED
), claimed AS (
	UPDATE campaign_contacts cc SET stage = ?, leased_by = ?, leased_until = ?
	FROM due WHERE cc.id = due.id
	RETURNING cc.*
)
SELECT * FROM claimed ORDER BY next_attempt_at, seq`,
				campaign.ID, Enrolled, now, free, InProgress, owner, now.Add(lease)).Scan(ctx, &contacts)
			if err != nil {
				return fmt.Errorf("store: claim due contacts: %w", err)
			}
			for _, contact := range contacts {
				claimed = append(claimed, Claimed{Campaign: campaign, Contact: contact})
			}
		}
		return nil
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return claimed, nil
}

// TakeOverExpired leases to owner up to limit contacts whose call's runner stopped renewing its
// lease, most likely because it died. The new owner settles the call; it never dials again.
// A contact of a campaign a data move paused is the source's until the cut-over, and its dial
// may not have been imported yet.
func (s *Store) TakeOverExpired(ctx context.Context, owner string, now time.Time, lease time.Duration, limit int) ([]Claimed, error) {
	if owner == "" {
		return nil, stack.Wrap(errors.New("store: a take-over needs an owner"))
	}
	if limit <= 0 {
		return nil, nil
	}

	var contacts []Contact
	err := s.db.NewRaw(`WITH expired AS (
	SELECT id FROM campaign_contacts cc
	WHERE stage = ? AND leased_until < ?
		AND NOT EXISTS (SELECT 1 FROM campaigns cp WHERE cp.id = cc.campaign_id AND cp.move_paused)
	ORDER BY leased_until
	LIMIT ?
	FOR UPDATE SKIP LOCKED
)
UPDATE campaign_contacts cc SET leased_by = ?, leased_until = ?
FROM expired WHERE cc.id = expired.id
RETURNING cc.*`, InProgress, now, limit, owner, now.Add(lease)).Scan(ctx, &contacts)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: take over expired leases: %w", err))
	}
	if len(contacts) == 0 {
		return nil, nil
	}

	campaigns, err := campaignsOf(ctx, s.db, contacts)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	claimed := make([]Claimed, 0, len(contacts))
	for _, contact := range contacts {
		claimed = append(claimed, Claimed{Campaign: campaigns[contact.CampaignID], Contact: contact})
	}
	return claimed, nil
}

// Settle is where a contact goes once its lease holder is done with a call.
type Settle struct {
	// Stage is Enrolled for another attempt, AwaitingOutcome for an answered call not yet
	// classified, or where the sequence ended it.
	Stage string
	// StepID moves the contact to another step of the sequence; empty keeps the one it is on.
	StepID        string
	Attempt       int
	NextAttemptAt *time.Time
	Outcome       string
	Converted     bool
	Reason        string
	// CallID and VendorCallID are the last dial's; empty keeps the ones already stored.
	CallID, VendorCallID string
}

// SettleContact records where a call left a contact and releases owner's lease on it. A contact
// stopped while in the call, or whose campaign was stopped, is stopped rather than rung again;
// one the sequence finished, or whose answered call still awaits its outcome, settles as given. A
// contact that no longer exists, because its campaign was deleted mid-call, is nothing to settle.
func (s *Store) SettleContact(ctx context.Context, owner, contactID string, settle Settle) error {
	return s.settleContact(ctx, owner, contactID, settle, nil)
}

// errUnsettled rolls back a settle that touched no contact, with whatever was written before it.
var errUnsettled = errors.New("store: no contact settled")

// settleContact is SettleContact, with first written in the same transaction, before the
// contact, and rolled back with it when the contact does not settle.
func (s *Store) settleContact(ctx context.Context, owner, contactID string, settle Settle, first func(context.Context, bun.Tx) error) error {
	if owner == "" || contactID == "" {
		return stack.Wrap(errors.New("store: an owner and a contact id are required"))
	}
	if settle.Stage == "" || settle.Stage == InProgress {
		return stack.Wrap(fmt.Errorf("store: a contact cannot settle in stage %q", settle.Stage))
	}

	// Only a stage that would ring the contact again gives way to a stop. An answered call
	// still gets its outcome read, and settles stopped from there, so it keeps the stop mark.
	// A contact the sequence ended says why in its own reason.
	stoppable := settle.Stage == Enrolled
	keepsMark := stoppable || settle.Stage == AwaitingOutcome
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if first != nil {
			if err := first(ctx, tx); err != nil {
				return err
			}
		}
		result, err := tx.ExecContext(ctx, `UPDATE campaign_contacts cc SET
	stage = CASE WHEN ? AND (cc.reason = 'stopped' OR cp.state = ?) THEN ? ELSE ? END,
	next_attempt_at = CASE WHEN ? AND (cc.reason = 'stopped' OR cp.state = ?) THEN NULL ELSE ?::timestamptz END,
	reason = CASE WHEN (? AND cc.reason = 'stopped') OR (? AND cp.state = ?) THEN cc.reason ELSE ? END,
	step_id = COALESCE(?, cc.step_id),
	attempt = ?,
	outcome = ?,
	converted = ?,
	call_id = COALESCE(?, cc.call_id),
	vendor_call_id = COALESCE(?, cc.vendor_call_id),
	leased_by = '',
	leased_until = NULL
FROM campaigns cp
WHERE cp.id = cc.campaign_id AND cc.id = ? AND cc.leased_by = ? AND cc.stage IN (?, ?)`,
			stoppable, Stopped, ContactStopped, settle.Stage,
			stoppable, Stopped, settle.NextAttemptAt,
			keepsMark, stoppable, Stopped, nullable(settle.Reason),
			nullable(settle.StepID), settle.Attempt, settle.Outcome, settle.Converted,
			nullable(settle.CallID), nullable(settle.VendorCallID),
			contactID, owner, InProgress, AwaitingOutcome)
		if err != nil {
			return fmt.Errorf("store: settle contact: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: settle contact: %w", err)
		}
		if affected == 0 {
			return errUnsettled
		}
		if settle.Stage == AwaitingOutcome {
			return nil
		}
		// The contact no longer waits for an outcome, so no dial of it is left to wake a
		// runner later.
		if _, err := tx.NewUpdate().Model((*CampaignAttempt)(nil)).
			Set("classify_state = ?", classifyAbandoned).
			Set("classify_after = NULL").
			Where("contact_id = ?", contactID).
			Where("classify_state = ?", classifyPending).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: abandon pending classifications: %w", err)
		}
		return nil
	})
	if !errors.Is(err, errUnsettled) {
		return stack.Wrap(err)
	}
	exists, err := s.db.NewSelect().Model((*Contact)(nil)).Where("id = ?", contactID).Exists(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: settle contact: %w", err))
	}
	if exists {
		return stack.Wrap(ErrNotLeaseHolder)
	}
	return nil
}

// Classification is what reading an answered dial's transcript found: an outcome, or that
// nobody but the agent spoke, which makes the dial unanswered.
type Classification struct {
	AttemptID  string
	Outcome    string
	Converted  bool
	Unanswered bool
}

// ClassifyAndSettle records what an answered dial's transcript said and settles its contact, both
// or neither: a dial recorded while its contact stayed awaiting its outcome would never be read
// again, and the contact never move on. A dial no longer pending is not changed.
func (s *Store) ClassifyAndSettle(ctx context.Context, owner string, classification Classification, contactID string, settle Settle) error {
	if classification.AttemptID == "" {
		return stack.Wrap(errors.New("store: a dial id is required"))
	}
	return s.settleContact(ctx, owner, contactID, settle, func(ctx context.Context, tx bun.Tx) error {
		query := tx.NewUpdate().Model((*CampaignAttempt)(nil)).
			Set("classify_after = NULL").
			Where("id = ?", classification.AttemptID).
			Where("classify_state = ?", classifyPending)
		if classification.Unanswered {
			query = query.Set("result = 'unanswered'").Set("classify_state = ''")
		} else {
			query = query.Set("classify_state = ?", classifyDone).
				Set("outcome = ?", classification.Outcome).
				Set("converted = ?", classification.Converted)
		}
		if _, err := query.Exec(ctx); err != nil {
			return fmt.Errorf("store: classify a dial: %w", err)
		}
		return nil
	})
}

// StartDial records a dial about to be placed, filling in its id.
func (s *Store) StartDial(ctx context.Context, attempt *CampaignAttempt) error {
	if attempt.CampaignID == "" || attempt.ContactID == "" {
		return stack.Wrap(errors.New("store: a dial needs a campaign and a contact"))
	}
	if attempt.StartedAt.IsZero() {
		return stack.Wrap(errors.New("store: a dial needs the time it started"))
	}
	attempt.ID = newID()
	if attempt.Dial == 0 {
		attempt.Dial = 1
	}
	if _, err := s.db.NewInsert().Model(attempt).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: start dial: %w", err))
	}
	return nil
}

// EndDial records how a dial ended. A classifyAfter marks an answered call whose outcome is to
// be read from its transcript once it is written. A dial deleted with its campaign is nothing
// to end.
func (s *Store) EndDial(ctx context.Context, id string, result, callID, vendorCallID string, endedAt time.Time, classifyAfter *time.Time) error {
	classify := ""
	if classifyAfter != nil {
		classify = classifyPending
	}
	_, err := s.db.NewUpdate().Model((*CampaignAttempt)(nil)).
		Set("result = ?", result).
		Set("call_id = COALESCE(NULLIF(?, ''), call_id)", callID).
		Set("vendor_call_id = COALESCE(NULLIF(?, ''), vendor_call_id)", vendorCallID).
		Set("ended_at = ?", endedAt).
		Set("classify_state = ?", classify).
		Set("classify_after = ?", classifyAfter).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: end dial: %w", err))
	}
	return nil
}

// LatestDial returns a contact's most recent dial, and false when it was never dialed.
func (s *Store) LatestDial(ctx context.Context, contactID string) (CampaignAttempt, bool, error) {
	var attempt CampaignAttempt
	err := s.db.NewSelect().Model(&attempt).
		Where("contact_id = ?", contactID).
		Order("started_at DESC", "dial DESC").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return CampaignAttempt{}, false, nil
	}
	if err != nil {
		return CampaignAttempt{}, false, stack.Wrap(fmt.Errorf("store: latest dial: %w", err))
	}
	return attempt, true, nil
}

// Classifying is an answered dial whose outcome a runner is to read, with the contact it now
// holds the lease of and the campaign.
type Classifying struct {
	Attempt  CampaignAttempt
	Contact  Contact
	Campaign Campaign
}

// ClaimClassifications hands owner up to limit answered dials whose transcript should be written
// by now. Each is put off by lease, so another runner retries it if this one dies, and its
// contact is leased to owner so owner can settle it. Only a contact's latest answered dial is
// read, and not while another runner holds the contact, nor while a data move's source still
// reads it (move_paused).
func (s *Store) ClaimClassifications(ctx context.Context, owner string, now time.Time, lease time.Duration, limit int) ([]Classifying, error) {
	if owner == "" {
		return nil, stack.Wrap(errors.New("store: a claim needs an owner"))
	}
	if limit <= 0 {
		return nil, nil
	}

	var claimed []Classifying
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var attempts []CampaignAttempt
		err := tx.NewRaw(`WITH due AS (
	SELECT ca.id FROM campaign_attempts ca
	JOIN campaign_contacts cc ON cc.id = ca.contact_id
	WHERE ca.classify_state = ? AND ca.classify_after <= ? AND cc.stage = ?
		AND (cc.leased_by = '' OR cc.leased_until IS NULL OR cc.leased_until < ?)
		AND NOT EXISTS (SELECT 1 FROM campaigns cp WHERE cp.id = cc.campaign_id AND cp.move_paused)
		AND NOT EXISTS (SELECT 1 FROM campaign_attempts newer
			WHERE newer.contact_id = ca.contact_id AND newer.classify_state = ca.classify_state
			AND (newer.started_at, newer.dial) > (ca.started_at, ca.dial))
	ORDER BY ca.classify_after
	LIMIT ?
	FOR UPDATE OF ca, cc SKIP LOCKED
)
UPDATE campaign_attempts ca SET classify_after = ?, classify_tries = ca.classify_tries + 1
FROM due WHERE ca.id = due.id
RETURNING ca.*`, classifyPending, now, AwaitingOutcome, now, limit, now.Add(lease)).Scan(ctx, &attempts)
		if err != nil {
			return fmt.Errorf("store: claim classifications: %w", err)
		}
		if len(attempts) == 0 {
			return nil
		}

		contactIDs := make([]string, 0, len(attempts))
		for _, attempt := range attempts {
			contactIDs = append(contactIDs, attempt.ContactID)
		}
		var contacts []Contact
		err = tx.NewUpdate().Model(&contacts).
			Set("leased_by = ?", owner).
			Set("leased_until = ?", now.Add(lease)).
			Where("id IN (?)", bun.In(contactIDs)).
			Returning("*").
			Scan(ctx)
		if err != nil {
			return fmt.Errorf("store: lease contacts to classify: %w", err)
		}
		campaigns, err := campaignsOf(ctx, tx, contacts)
		if err != nil {
			return err
		}
		byID := make(map[string]Contact, len(contacts))
		for _, contact := range contacts {
			byID[contact.ID] = contact
		}
		for _, attempt := range attempts {
			contact := byID[attempt.ContactID]
			claimed = append(claimed, Classifying{Attempt: attempt, Contact: contact, Campaign: campaigns[contact.CampaignID]})
		}
		return nil
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return claimed, nil
}

// PutOffClassification gives back an answered dial whose transcript is not written yet, to be
// read again at at, and releases owner's lease on its contact so whichever runner is free then
// reads it.
func (s *Store) PutOffClassification(ctx context.Context, owner, attemptID string, at time.Time) error {
	if owner == "" || attemptID == "" {
		return stack.Wrap(errors.New("store: an owner and a dial id are required"))
	}
	var released int64
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		result, err := tx.ExecContext(ctx, `UPDATE campaign_contacts cc SET leased_by = '', leased_until = NULL
FROM campaign_attempts ca
WHERE ca.id = ? AND ca.classify_state = ? AND cc.id = ca.contact_id AND cc.stage = ? AND cc.leased_by = ?`,
			attemptID, classifyPending, AwaitingOutcome, owner)
		if err != nil {
			return fmt.Errorf("store: release a contact to classify later: %w", err)
		}
		if released, err = result.RowsAffected(); err != nil || released == 0 {
			return err
		}
		if _, err := tx.NewUpdate().Model((*CampaignAttempt)(nil)).
			Set("classify_after = ?", at).
			Where("id = ?", attemptID).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: put off a classification: %w", err)
		}
		return nil
	})
	if err != nil {
		return stack.Wrap(err)
	}
	if released == 0 {
		return stack.Wrap(ErrNotLeaseHolder)
	}
	return nil
}

// Classified records an answered dial's outcome. A dial deleted with its campaign, or no longer
// pending because another runner read it or its contact settled, is nothing to record.
func (s *Store) Classified(ctx context.Context, attemptID, outcome string, converted bool) error {
	_, err := s.db.NewUpdate().Model((*CampaignAttempt)(nil)).
		Set("classify_state = ?", classifyDone).
		Set("classify_after = NULL").
		Set("outcome = ?", outcome).
		Set("converted = ?", converted).
		Where("id = ?", attemptID).
		Where("classify_state = ?", classifyPending).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: classified: %w", err))
	}
	return nil
}

// CompleteCampaigns completes the running campaigns that have nobody left to ring and no call
// whose outcome is still unknown. An auto-enroll campaign waits for more people instead.
func (s *Store) CompleteCampaigns(ctx context.Context, now time.Time) error {
	_, err := s.db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", Completed).
		Set("finished_at = ?", now).
		Where("cp.state = ?", Running).
		Where("cp.type = ?", Standard).
		Where(`NOT EXISTS (SELECT 1 FROM campaign_contacts cc WHERE cc.campaign_id = cp.id
			AND ((cc.stage = ? AND cc.next_attempt_at IS NOT NULL) OR cc.stage IN (?, ?)))`,
			Enrolled, InProgress, AwaitingOutcome).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: complete campaigns: %w", err))
	}
	return nil
}

// campaignsOf returns the campaigns of contacts, by id.
func campaignsOf(ctx context.Context, db bun.IDB, contacts []Contact) (map[string]Campaign, error) {
	var ids []string
	for _, contact := range contacts {
		if !slices.Contains(ids, contact.CampaignID) {
			ids = append(ids, contact.CampaignID)
		}
	}
	var campaigns []Campaign
	if err := db.NewSelect().Model(&campaigns).Where("id IN (?)", bun.In(ids)).Scan(ctx); err != nil {
		return nil, fmt.Errorf("store: campaigns of contacts: %w", err)
	}
	byID := make(map[string]Campaign, len(campaigns))
	for _, campaign := range campaigns {
		byID[campaign.ID] = campaign
	}
	return byID, nil
}
