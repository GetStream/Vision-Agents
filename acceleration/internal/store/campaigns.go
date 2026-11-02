package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrCampaignLocked is a change a campaign's state does not allow.
var ErrCampaignLocked = errors.New("store: the campaign is in a state that does not allow this")

// ErrTooManyContacts is a campaign that would hold more than MaxCampaignContacts.
var ErrTooManyContacts = errors.New("store: a campaign holds at most 100000 contacts")

// ErrNoCampaignContact is a contact the campaign does not hold.
var ErrNoCampaignContact = errors.New("store: the campaign holds no such contact")

// MaxCampaignContacts is how many contacts one campaign may hold.
const MaxCampaignContacts = 100_000

// insertChunk keeps one INSERT of contacts a size the server parses quickly.
const insertChunk = 5000

// defaultConcurrency is how many calls a campaign that did not say may have at once. One
// is the safe answer: a campaign that rings a thousand people the moment it starts is a
// mistake nobody meant to make.
const defaultConcurrency = 1

const (
	defaultCampaignLimit = 50
	maxCampaignLimit     = 1001
)

// maxConcurrency bounds what a customer may ask for, since every call in flight is a
// conversation this process is running.
const maxConcurrency = 50

// CreateCampaign stores a campaign and fills in its id and timestamps.
func (s *Store) CreateCampaign(ctx context.Context, campaign *Campaign) error {
	if campaign.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if campaign.Name == "" {
		return stack.Wrap(errors.New("store: a campaign needs a name"))
	}
	if campaign.Concurrency <= 0 {
		campaign.Concurrency = defaultConcurrency
	}
	if campaign.Concurrency > maxConcurrency {
		return stack.Wrap(fmt.Errorf("store: a campaign may run at most %d calls at once", maxConcurrency))
	}

	campaign.ID = newID()
	campaign.State = Draft
	if campaign.Type == "" {
		campaign.Type = Standard
	}
	campaign.CreatedAt = time.Now().UTC()
	if campaign.Tags == nil {
		campaign.Tags = map[string]string{}
	}

	if _, err := s.db.NewInsert().Model(campaign).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: create campaign: %w", err))
	}
	return nil
}

// Campaign returns one campaign a customer holds.
func (s *Store) Campaign(ctx context.Context, customerID, id string) (Campaign, error) {
	return campaignIn(ctx, s.db, customerID, id)
}

func campaignIn(ctx context.Context, db bun.IDB, customerID, id string) (Campaign, error) {
	if customerID == "" || id == "" {
		return Campaign{}, stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}

	var campaign Campaign
	err := db.NewSelect().Model(&campaign).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Campaign{}, unknownCampaign(id)
	}
	if err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: campaign: %w", err))
	}
	return campaign, nil
}

// CampaignFilter narrows and pages QueryCampaigns. It returns at most Limit rows: a caller
// that wants to know whether another page follows asks for one more than it shows.
type CampaignFilter struct {
	State string
	Limit int
	After *CreatedPosition
}

// QueryCampaigns returns a customer's campaigns, newest first.
func (s *Store) QueryCampaigns(ctx context.Context, customerID string, filter CampaignFilter) ([]Campaign, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var campaigns []Campaign
	query := s.db.NewSelect().Model(&campaigns).Where("customer_id = ?", customerID)
	if filter.State != "" {
		query = query.Where("state = ?", filter.State)
	}
	if filter.After != nil {
		query = query.Where("(created_at, id) < (?, ?)", filter.After.At, filter.After.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(clampLimit(filter.Limit, defaultCampaignLimit, maxCampaignLimit)).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: query campaigns: %w", err))
	}
	return campaigns, nil
}

// CampaignChanges are the fields an update sets; a nil one is left as it is.
type CampaignChanges struct {
	Name, ConfigID, FromNumber, DefaultCountry *string
	Concurrency                                *int
	Sequence, CallingHours                     json.RawMessage
	Tags                                       map[string]string
}

// UpdateCampaign changes a campaign. Once it has launched only its name, number and
// concurrency may change: the rest is what the customer consented to ring people with.
func (s *Store) UpdateCampaign(ctx context.Context, customerID, id string, changes CampaignChanges) (Campaign, error) {
	if customerID == "" || id == "" {
		return Campaign{}, stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}
	if changes.Concurrency != nil && (*changes.Concurrency < 1 || *changes.Concurrency > maxConcurrency) {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: a campaign may run between 1 and %d calls at once", maxConcurrency))
	}

	query := s.db.NewUpdate().Model((*Campaign)(nil)).Where("id = ?", id).Where("customer_id = ?", customerID)
	changed := 0
	var same []string
	var sameArgs []any
	set := func(column, cast string, value any, frozen bool) {
		query = query.Set(column+" = ?"+cast, value)
		changed++
		if frozen {
			same = append(same, column+" IS NOT DISTINCT FROM ?"+cast)
			sameArgs = append(sameArgs, value)
		}
	}
	if changes.Name != nil {
		set("name", "", *changes.Name, false)
	}
	if changes.FromNumber != nil {
		set("from_number", "", *changes.FromNumber, false)
	}
	if changes.Concurrency != nil {
		set("concurrency", "", *changes.Concurrency, false)
	}
	if changes.ConfigID != nil {
		set("config_id", "", *changes.ConfigID, true)
	}
	if changes.DefaultCountry != nil {
		set("default_country", "", *changes.DefaultCountry, true)
	}
	if changes.Sequence != nil {
		set("sequence", "::jsonb", string(changes.Sequence), true)
	}
	if changes.CallingHours != nil {
		set("calling_hours", "::jsonb", string(changes.CallingHours), true)
	}
	if changes.Tags != nil {
		tags, err := json.Marshal(changes.Tags)
		if err != nil {
			return Campaign{}, stack.Wrap(fmt.Errorf("store: update campaign tags: %w", err))
		}
		set("tags", "::jsonb", string(tags), true)
	}
	if changed == 0 {
		return s.Campaign(ctx, customerID, id)
	}
	// A launched campaign accepts a locked field only when it says what is already there, so
	// a form that sends the whole object does not have to leave the rest out.
	if len(same) > 0 {
		query = query.Where("(state = ? OR ("+strings.Join(same, " AND ")+"))", append([]any{Draft}, sameArgs...)...)
	}

	result, err := query.Exec(ctx)
	if err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: update campaign: %w", err))
	}
	if affected, err := result.RowsAffected(); err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: update campaign: %w", err))
	} else if affected == 0 {
		return Campaign{}, s.whyNotChanged(ctx, s.db, customerID, id)
	}
	return s.Campaign(ctx, customerID, id)
}

// whyNotChanged names why a guarded write touched no row: there is no such campaign, or its
// state refused it.
func (s *Store) whyNotChanged(ctx context.Context, db bun.IDB, customerID, id string) error {
	if _, err := campaignIn(ctx, db, customerID, id); err != nil {
		return err
	}
	return stack.Wrap(ErrCampaignLocked)
}

// DeleteCampaign removes a campaign that is not ringing anybody, with its contacts. Its
// calls stay: they are what was billed and recorded.
func (s *Store) DeleteCampaign(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}
	result, err := s.db.NewDelete().Model((*Campaign)(nil)).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("state IN (?)", bun.In([]string{Draft, Stopped, Completed})).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete campaign: %w", err))
	}
	return s.guarded(ctx, result, customerID, id)
}

// guarded turns a write that touched no row into the error saying why.
func (s *Store) guarded(ctx context.Context, result sql.Result, customerID, id string) error {
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: campaign write: %w", err))
	}
	if affected == 0 {
		return s.whyNotChanged(ctx, s.db, customerID, id)
	}
	return nil
}

// Duplicate contacts, in DuplicateCampaign.
const (
	DuplicateNone         = "none"
	DuplicateAll          = "all"
	DuplicateNotContacted = "not_contacted"
)

// DuplicateCampaign makes a draft with a campaign's settings and, as contacts says, none of
// its contacts, all of them or those never dialed. Contacts start over, enrolled with no
// attempts. Never dialed means the campaign has finished dialing: from a stopped or completed one.
func (s *Store) DuplicateCampaign(ctx context.Context, customerID, id, contacts string) (Campaign, error) {
	if contacts != DuplicateNone && contacts != DuplicateAll && contacts != DuplicateNotContacted {
		return Campaign{}, stack.Wrap(fmt.Errorf("store: duplicate contacts must be none, all or not_contacted, not %q", contacts))
	}

	var copied Campaign
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		source, err := campaignIn(ctx, tx, customerID, id)
		if err != nil {
			return err
		}
		if contacts == DuplicateNotContacted && source.State != Stopped && source.State != Completed {
			return stack.Wrap(ErrCampaignLocked)
		}

		copied = Campaign{
			ID: newID(), CustomerID: source.CustomerID, Name: "Copy of " + source.Name, ConfigID: source.ConfigID,
			FromNumber: source.FromNumber, Concurrency: source.Concurrency, Sequence: source.Sequence,
			CallingHours: source.CallingHours, DefaultCountry: source.DefaultCountry, State: Draft, Type: source.Type,
			Tags: source.Tags, CreatedAt: time.Now().UTC(),
		}
		if _, err := tx.NewInsert().Model(&copied).Exec(ctx); err != nil {
			return fmt.Errorf("store: duplicate campaign: %w", err)
		}
		if contacts == DuplicateNone {
			return nil
		}

		// A row the migration marked duplicate repeats a number another row holds, so it is
		// left behind: copying it would break the unique number index.
		query := `INSERT INTO campaign_contacts (id, campaign_id, to_number, instructions, variables, time_zone, stage, created_at)
SELECT replace(gen_random_uuid()::text, '-', ''), ?, to_number, instructions, variables, time_zone, 'enrolled', now()
FROM campaign_contacts cc
WHERE campaign_id = ? AND reason IS DISTINCT FROM 'duplicate'`
		if contacts == DuplicateNotContacted {
			query += ` AND NOT EXISTS (SELECT 1 FROM campaign_attempts ca WHERE ca.contact_id = cc.id)`
		}
		if _, err := tx.ExecContext(ctx, query+" ORDER BY seq", copied.ID, source.ID); err != nil {
			return fmt.Errorf("store: duplicate contacts: %w", err)
		}
		return nil
	})
	if err != nil {
		return Campaign{}, stack.Wrap(err)
	}
	return copied, nil
}

// Launch is what starting a campaign records. Now is passed in so the router decides once what
// time it is for the whole launch.
type Launch struct {
	StartAt, Now time.Time
	// APIKey, UserID and UsageCategories are who launched it: the calls it places later are
	// owned by them.
	APIKey, UserID  string
	UsageCategories []string
}

// LaunchCampaign takes a draft out of draft. The customer's consent is recorded, contacts
// that opted out of voice are rejected without a call, and the rest are enrolled to be
// rung from StartAt.
func (s *Store) LaunchCampaign(ctx context.Context, customerID, id string, launch Launch) (Campaign, error) {
	state := Running
	if launch.StartAt.After(launch.Now) {
		state = Scheduled
	}
	// The column holds an empty list, never NULL, for a request that named no categories.
	categories := launch.UsageCategories
	if categories == nil {
		categories = []string{}
	}

	var launched Campaign
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		query := tx.NewUpdate().Model((*Campaign)(nil)).
			Set("state = ?", state).
			Set("start_at = ?", launch.StartAt).
			Set("launched_at = ?", launch.Now).
			Set("consent_at = ?", launch.Now).
			Set("api_key = ?", launch.APIKey).
			Set("user_id = ?", launch.UserID).
			Set("usage_categories = ?", pgdialect.Array(categories)).
			Where("id = ?", id).
			Where("customer_id = ?", customerID).
			Where("state = ?", Draft)
		if state == Running {
			query = query.Set("started_at = ?", launch.Now)
		}
		result, err := query.Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: launch campaign: %w", err)
		}
		if affected, err := result.RowsAffected(); err != nil {
			return fmt.Errorf("store: launch campaign: %w", err)
		} else if affected == 0 {
			return s.whyNotChanged(ctx, tx, customerID, id)
		}

		if _, err := tx.NewUpdate().Model((*Contact)(nil)).
			Set("stage = ?", Rejected).
			Set("reason = 'opted_out'").
			Set("next_attempt_at = NULL").
			Where("campaign_id = ?", id).
			Where("stage = ?", Enrolled).
			Where(`EXISTS (SELECT 1 FROM opt_outs oo WHERE oo.customer_id = ? AND oo.recipient = cc.to_number
				AND oo.channel IN ('voice', ?) AND oo.revoked_at IS NULL)`, customerID, OptOutAll).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: reject opted-out contacts: %w", err)
		}
		if _, err := tx.NewUpdate().Model((*Contact)(nil)).
			Set("step_id = 'step_1'").
			Set("attempt = 0").
			Set("next_attempt_at = ?", launch.StartAt).
			Where("campaign_id = ?", id).
			Where("stage = ?", Enrolled).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: enroll contacts: %w", err)
		}
		launched, err = campaignIn(ctx, tx, customerID, id)
		return err
	})
	if err != nil {
		return Campaign{}, stack.Wrap(err)
	}
	return launched, nil
}

// PauseCampaign pauses a running campaign. A reason names what refused its next call when the
// router paused it itself; empty means somebody paused it.
func (s *Store) PauseCampaign(ctx context.Context, customerID, id, reason string) error {
	return s.transition(ctx, customerID, id, []string{Running}, Paused, func(query *bun.UpdateQuery) *bun.UpdateQuery {
		return query.Set("paused_reason = NULLIF(?, '')", reason)
	})
}

// ResumeCampaign starts a paused campaign again.
func (s *Store) ResumeCampaign(ctx context.Context, customerID, id string) error {
	return s.transition(ctx, customerID, id, []string{Paused}, Running, func(query *bun.UpdateQuery) *bun.UpdateQuery {
		return query.Set("paused_reason = NULL")
	})
}

// StopCampaign ends a campaign for good. Contacts still waiting stay enrolled, and are never
// rung; calls in progress are left to finish.
func (s *Store) StopCampaign(ctx context.Context, customerID, id string, now time.Time) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		err := s.transitionIn(ctx, tx, customerID, id, []string{Scheduled, Running, Paused}, Stopped, func(query *bun.UpdateQuery) *bun.UpdateQuery {
			return query.Set("paused_reason = NULL").Set("stopped_at = ?", now)
		})
		if err != nil {
			return err
		}
		if _, err := tx.NewUpdate().Model((*Contact)(nil)).
			Set("next_attempt_at = NULL").
			Where("campaign_id = ?", id).
			Where("stage = ?", Enrolled).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: stop campaign contacts: %w", err)
		}
		return nil
	})
	return stack.Wrap(err)
}

func (s *Store) transition(ctx context.Context, customerID, id string, from []string, to string, set func(*bun.UpdateQuery) *bun.UpdateQuery) error {
	return stack.Wrap(s.transitionIn(ctx, s.db, customerID, id, from, to, set))
}

// transitionIn moves a campaign from one of the states to another, and says why when it was not
// in one of them. A move clears the data move's mark: the customer has now decided what the
// campaign does here.
func (s *Store) transitionIn(ctx context.Context, db bun.IDB, customerID, id string, from []string, to string, set func(*bun.UpdateQuery) *bun.UpdateQuery) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a campaign id are required")
	}
	query := db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", to).
		Set("move_paused = false").
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("state IN (?)", bun.In(from))
	result, err := set(query).Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: move campaign to %s: %w", to, err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: move campaign to %s: %w", to, err)
	}
	if affected == 0 {
		return s.whyNotChanged(ctx, db, customerID, id)
	}
	return nil
}

// SetCampaignState records what a campaign is now doing. Starting one stamps when it
// first started, and finishing one stamps when it stopped having anybody left to ring. Either
// clears why the router last paused it, and that a data move did (ResumeMovedRows): the
// customer has decided what it does here.
func (s *Store) SetCampaignState(ctx context.Context, customerID, id, state string) error {
	return s.setCampaignState(ctx, customerID, id, state, "")
}

func (s *Store) setCampaignState(ctx context.Context, customerID, id, state, reason string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a campaign id are required"))
	}

	now := time.Now().UTC()
	query := s.db.NewUpdate().Model((*Campaign)(nil)).
		Set("state = ?", state).
		Set("paused_reason = NULLIF(?, '')", reason).
		Set("move_paused = false").
		Where("id = ?", id).
		Where("customer_id = ?", customerID)
	switch state {
	case Running:
		query = query.Set("started_at = COALESCE(started_at, ?)", now).
			Set("finished_at = NULL")
	case Completed:
		query = query.Set("finished_at = ?", now)
	}

	result, err := query.Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set campaign state: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set campaign state: %w", err))
	}
	if affected == 0 {
		return unknownCampaign(id)
	}
	return nil
}

// AddContacts stores people to ring in a draft campaign, in the order given. A number the
// campaign already holds, or an earlier contact of the same call repeats, is not stored: its
// index in contacts is returned instead, so the caller can say which were skipped.
func (s *Store) AddContacts(ctx context.Context, campaignID string, contacts []Contact) (added []Contact, duplicates []int, err error) {
	if campaignID == "" {
		return nil, nil, stack.Wrap(errors.New("store: a campaign id is required"))
	}
	if len(contacts) == 0 {
		return nil, nil, stack.Wrap(errors.New("store: there is nobody to add"))
	}

	now := time.Now().UTC()
	for index := range contacts {
		if contacts[index].ToNumber == "" {
			return nil, nil, stack.Wrap(errors.New("store: a contact needs a number to ring"))
		}
		contacts[index].ID = newID()
		contacts[index].CampaignID = campaignID
		contacts[index].Stage = Enrolled
		contacts[index].CreatedAt = now
		if contacts[index].Variables == nil {
			contacts[index].Variables = map[string]string{}
		}
	}

	stored := make(map[string]int64, len(contacts))
	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockDraft(ctx, tx, campaignID); err != nil {
			return err
		}
		for start := 0; start < len(contacts); start += insertChunk {
			chunk := contacts[start:min(start+insertChunk, len(contacts))]
			var inserted []struct {
				ID  string
				Seq int64
			}
			// A repeat is skipped rather than failing the batch; rows the migration marked
			// duplicate are outside the index, so they do not count as holding the number.
			err := tx.NewInsert().Model(&chunk).
				On("CONFLICT (campaign_id, to_number) WHERE reason IS DISTINCT FROM 'duplicate' DO NOTHING").
				Returning("id, seq").
				Scan(ctx, &inserted)
			if err != nil {
				return fmt.Errorf("store: add contacts: %w", err)
			}
			for _, row := range inserted {
				stored[row.ID] = row.Seq
			}
		}
		// Counted after the insert, so a batch of numbers already held does not count against
		// the cap; the lock on the campaign row keeps the count from moving, and returning
		// the error rolls the insert back.
		held, err := tx.NewSelect().Model((*Contact)(nil)).Where("campaign_id = ?", campaignID).Count(ctx)
		if err != nil {
			return fmt.Errorf("store: count contacts: %w", err)
		}
		if held > MaxCampaignContacts {
			return ErrTooManyContacts
		}
		return nil
	})
	if err != nil {
		return nil, nil, stack.Wrap(err)
	}
	for index, contact := range contacts {
		seq, ok := stored[contact.ID]
		if !ok {
			duplicates = append(duplicates, index)
			continue
		}
		contact.Seq = seq
		added = append(added, contact)
	}
	return added, duplicates, nil
}

// lockDraft holds a campaign's row, which keeps a launch from slipping in between a check
// that it is still a draft and the write that relies on it.
func lockDraft(ctx context.Context, tx bun.Tx, campaignID string) error {
	var state string
	err := tx.NewSelect().Model((*Campaign)(nil)).Column("state").Where("id = ?", campaignID).For("UPDATE").Scan(ctx, &state)
	if errors.Is(err, sql.ErrNoRows) {
		return unknownCampaign(campaignID)
	}
	if err != nil {
		return fmt.Errorf("store: campaign state: %w", err)
	}
	if state != Draft {
		return ErrCampaignLocked
	}
	return nil
}

// DeleteContacts removes every contact of a draft campaign.
func (s *Store) DeleteContacts(ctx context.Context, campaignID string) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockDraft(ctx, tx, campaignID); err != nil {
			return err
		}
		if _, err := tx.NewDelete().Model((*Contact)(nil)).Where("campaign_id = ?", campaignID).Exec(ctx); err != nil {
			return fmt.Errorf("store: delete contacts: %w", err)
		}
		return nil
	})
	return stack.Wrap(err)
}

// ContactFilter narrows and pages QueryContacts. After is the seq of the last contact seen.
// Stage in_progress also finds calls awaiting an outcome, which the API shows the same way.
type ContactFilter struct {
	Stage  string
	Number string
	Limit  int
	After  *int64
}

// likeEscaper makes a typed number match as itself rather than as a pattern.
var likeEscaper = strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)

// QueryContacts returns a campaign's contacts in the order they were added. Like
// QueryCampaigns it returns at most Limit rows.
func (s *Store) QueryContacts(ctx context.Context, campaignID string, filter ContactFilter) ([]Contact, error) {
	if campaignID == "" {
		return nil, stack.Wrap(errors.New("store: a campaign id is required"))
	}

	var contacts []Contact
	query := s.db.NewSelect().Model(&contacts).Where("campaign_id = ?", campaignID)
	switch filter.Stage {
	case "":
	case InProgress:
		query = query.Where("stage IN (?)", bun.In([]string{InProgress, AwaitingOutcome}))
	default:
		query = query.Where("stage = ?", filter.Stage)
	}
	if filter.Number != "" {
		query = query.Where(`to_number LIKE ? ESCAPE '\'`, "%"+likeEscaper.Replace(filter.Number)+"%")
	}
	if filter.After != nil {
		query = query.Where("seq > ?", *filter.After)
	}
	if err := query.Order("seq ASC").Limit(clampLimit(filter.Limit, defaultCampaignLimit, maxCampaignLimit)).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: query contacts: %w", err))
	}
	return contacts, nil
}

// CampaignContact returns one contact of a campaign with its attempts, oldest first.
func (s *Store) CampaignContact(ctx context.Context, campaignID, contactID string) (Contact, []CampaignAttempt, error) {
	var contact Contact
	err := s.db.NewSelect().Model(&contact).Where("id = ?", contactID).Where("campaign_id = ?", campaignID).Limit(1).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Contact{}, nil, unknownContact(contactID)
	}
	if err != nil {
		return Contact{}, nil, stack.Wrap(fmt.Errorf("store: campaign contact: %w", err))
	}
	var attempts []CampaignAttempt
	err = s.db.NewSelect().Model(&attempts).Where("contact_id = ?", contactID).Order("started_at ASC", "id ASC").Scan(ctx)
	if err != nil {
		return Contact{}, nil, stack.Wrap(fmt.Errorf("store: contact attempts: %w", err))
	}
	return contact, attempts, nil
}

// StopContact leaves a contact unrung. One waiting for its next attempt is stopped now; one
// in a call is marked, and is stopped when the call settles, not cut off. A contact the
// sequence is done with is left as it is.
func (s *Store) StopContact(ctx context.Context, campaignID, contactID string) error {
	result, err := s.db.NewUpdate().Model((*Contact)(nil)).
		Set("stage = CASE WHEN stage = ? THEN ? ELSE stage END", Enrolled, ContactStopped).
		Set("next_attempt_at = CASE WHEN stage = ? THEN NULL ELSE next_attempt_at END", Enrolled).
		Set("reason = CASE WHEN stage = ? THEN reason ELSE 'stopped' END", Enrolled).
		Where("id = ?", contactID).
		Where("campaign_id = ?", campaignID).
		Where("stage IN (?)", bun.In([]string{Enrolled, InProgress, AwaitingOutcome})).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: stop contact: %w", err))
	}
	if affected, err := result.RowsAffected(); err != nil {
		return stack.Wrap(fmt.Errorf("store: stop contact: %w", err))
	} else if affected == 0 {
		if _, _, err := s.CampaignContact(ctx, campaignID, contactID); err != nil {
			return err
		}
	}
	return nil
}

// nullable keeps an empty string out of a column that means "nothing happened" by being
// null rather than blank.
func nullable(text string) any {
	if text == "" {
		return nil
	}
	return text
}

func unknownContact(id string) error {
	return stack.Wrap(fmt.Errorf("%w: %s", ErrNoCampaignContact, id))
}

func unknownCampaign(id string) error {
	return stack.Wrap(fmt.Errorf("store: there is no campaign %s", id))
}
