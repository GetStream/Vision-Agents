package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// OpenCallLeg records a leg of a call in this shard: an outbound one ringing, an inbound one
// started. It reports false, and records nothing, for an inbound leg its Stream session already
// has, ended or not: every delivery of the event announcing a call that arrives opens it, and
// only the first does. A leg that opens takes the join of its phone delivered before it
// (takeParticipant), and leg is what was recorded: started, or ended too, by that join.
func (s *Store) OpenCallLeg(ctx context.Context, leg *CallLeg) (bool, error) {
	switch {
	case leg.CustomerID == "":
		return false, stack.Wrap(errors.New("store: a call leg needs a customer"))
	case leg.CallType == "" || leg.CallID == "":
		return false, stack.Wrap(errors.New("store: a call leg needs a call"))
	case leg.Direction != Inbound && leg.Direction != Outbound:
		return false, stack.Wrap(fmt.Errorf("store: a call leg is inbound or outbound, not %q", leg.Direction))
	}
	if leg.DialedAt.IsZero() {
		leg.DialedAt = time.Now()
	}
	leg.DialedAt = leg.DialedAt.UTC().Truncate(time.Microsecond)
	if leg.StartedAt != nil {
		started := leg.StartedAt.UTC().Truncate(time.Microsecond)
		leg.StartedAt = &started
	}
	result, err := s.shard.NewInsert().Model(leg).On("CONFLICT DO NOTHING").Returning("id").Exec(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: open call leg: %w", err))
	}
	opened, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: open call leg: %w", err))
	}
	if opened == 0 {
		return false, nil
	}
	if err := s.takeParticipant(ctx, leg); err != nil {
		return true, stack.Wrap(fmt.Errorf("store: give a call leg the phone that joined before it: %w", err))
	}
	return true, nil
}

// takeParticipant gives a leg that just opened the join of its phone kept from before it
// (RecordParticipant): by the leg's remote number, then, for an outbound leg, the number it
// was placed from, then its participant session, then, for a leg that knows none of them, the
// call's first phone to join. The leg starts when that phone joined, or when the leg opened if
// that is later, and ends when the phone left if its leave was kept too.
func (s *Store) takeParticipant(ctx context.Context, leg *CallLeg) error {
	var matches []func(*bun.SelectQuery) *bun.SelectQuery
	byNumber := func(number string) func(*bun.SelectQuery) *bun.SelectQuery {
		return func(q *bun.SelectQuery) *bun.SelectQuery { return q.Where("ltrim(cp.number, '+') = ?", number) }
	}
	remote, local := strings.TrimPrefix(leg.RemoteNumber, "+"), strings.TrimPrefix(leg.LocalNumber, "+")
	if remote != "" {
		matches = append(matches, byNumber(remote))
	}
	if leg.Direction == Outbound && local != "" {
		matches = append(matches, byNumber(local))
	}
	if leg.ParticipantSession != "" {
		matches = append(matches, func(q *bun.SelectQuery) *bun.SelectQuery {
			return q.Where("cp.participant_session = ?", leg.ParticipantSession)
		})
	}
	if len(matches) == 0 {
		matches = append(matches, func(q *bun.SelectQuery) *bun.SelectQuery { return q })
	}
	for _, match := range matches {
		for _, db := range s.shards() {
			taken, err := s.takeFrom(ctx, db, leg, match)
			if err != nil || taken {
				return err
			}
		}
	}
	return nil
}

// errLegGone rolls back a take whose leg was given another phone, or ended, meanwhile; and
// errPhoneHasLeg one whose phone was given another leg.
var (
	errLegGone     = errors.New("store: the leg is no longer waiting for its phone")
	errPhoneHasLeg = errors.New("store: the phone has another leg")
)

// takeFrom takes, from one shard, the first kept join match finds for the leg, and gives the
// leg that phone, in one transaction with the delete when the shard is the leg's own. A join
// in another shard is deleted only once the leg has it. It reports true when there is nothing
// more to look for: the leg has a phone, or no longer waits for one.
func (s *Store) takeFrom(ctx context.Context, db bun.IDB, leg *CallLeg, match func(*bun.SelectQuery) *bun.SelectQuery) (bool, error) {
	var joined CallLeg
	take := func(ctx context.Context, from, legs bun.IDB) error {
		kept := match(from.NewSelect().Model((*CallParticipant)(nil)).Column("cp.id").
			Where("cp.call_type = ?", leg.CallType).Where("cp.call_id = ?", leg.CallID).
			Where("cp.joined_at IS NOT NULL").Where("cp.participant_session <> ''").
			Where("NOT EXISTS (SELECT 1 FROM call_legs AS l WHERE l.call_type = cp.call_type AND l.call_id = cp.call_id " +
				"AND l.participant_session = cp.participant_session)"))
		if leg.StreamAppPK != 0 {
			kept = kept.Where("cp.stream_app_pk = ?", leg.StreamAppPK)
		} else {
			kept = kept.Where("cp.unpinned")
		}
		var taken []CallParticipant
		err := from.NewDelete().Model((*CallParticipant)(nil)).
			Where("id IN (?)", kept.OrderExpr("cp.joined_at").Limit(1).For("UPDATE SKIP LOCKED")).
			Returning("*").
			Scan(ctx, &taken)
		if err != nil || len(taken) == 0 {
			return err
		}
		joined, err = joinTaken(ctx, legs, *leg, taken[0])
		return err
	}
	var err error
	if db == bun.IDB(s.shard) {
		err = s.shard.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error { return take(ctx, tx, tx) })
	} else {
		err = db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error { return take(ctx, tx, s.shard) })
	}
	switch {
	case errors.Is(err, errLegGone):
		return true, nil
	case errors.Is(err, errPhoneHasLeg):
		return false, nil
	case err != nil:
		return false, err
	case joined.ID != 0:
		*leg = joined
		return true, nil
	}
	return false, nil
}

// joinTaken starts a leg, which legs holds, from the kept join of its phone, no earlier than the
// leg opened, and ends it if the phone's leave was kept too. A leg given another phone
// meanwhile keeps it, and its start (errLegGone), and so does a phone given another leg
// (errPhoneHasLeg).
func joinTaken(ctx context.Context, legs bun.IDB, leg CallLeg, taken CallParticipant) (CallLeg, error) {
	update := legs.NewUpdate().Model((*CallLeg)(nil)).
		Set("participant_session = ?", taken.ParticipantSession).
		Set("started_at = GREATEST(?::timestamptz, dialed_at)", *taken.JoinedAt)
	if taken.LeftAt != nil {
		update = update.Set("ended_at = GREATEST(?::timestamptz, ?::timestamptz, dialed_at)", *taken.LeftAt, *taken.JoinedAt).
			Set("end_reason = ?", LegLeft)
	}
	var joined []CallLeg
	err := update.Where("id = ?", leg.ID).Where("ended_at IS NULL").
		Where("participant_session IN ('', ?)", taken.ParticipantSession).
		Returning("*").Scan(ctx, &joined)
	if isViolation(err, uniqueViolation) {
		return CallLeg{}, errPhoneHasLeg
	}
	if err != nil {
		return CallLeg{}, err
	}
	if len(joined) == 0 {
		return CallLeg{}, errLegGone
	}
	return joined[0], nil
}

// Connection is what a phone joining a call did to the call's legs.
type Connection int

const (
	// NothingWaiting is no leg of the call waiting for a phone: the join is kept instead
	// (RecordParticipant).
	NothingWaiting Connection = iota
	// LegConnected is the phone given its leg, which starts.
	LegConnected
	// LegKnown is the phone already having its leg, from an earlier or concurrent delivery.
	LegKnown
)

// ConnectCallLeg gives a phone that joined a call, at at, the leg it is: the waiting leg whose
// number is the phone's (phoneNumbersOf), or when none is or the number is unknown, the call's
// inbound leg that does not know its phone yet, or else the oldest outbound leg still ringing,
// which starts then. Stream does not order its call events, so a call whose inbound leg opened
// without its caller has a transfer's phone joining it, and only the number tells them apart.
// A phone whose leave came first, at left, has its leg ended then too.
func (s *Store) ConnectCallLeg(ctx context.Context, scope AppScope, callType, callID, sessionID, participant, number string,
	at time.Time, left *time.Time) (CallLeg, Connection, error) {
	at = at.UTC().Truncate(time.Microsecond)
	known, err := s.ParticipantHasLeg(ctx, scope, callType, callID, participant)
	if err != nil {
		return CallLeg{}, NothingWaiting, stack.Wrap(fmt.Errorf("store: connect call leg: %w", err))
	}
	if known {
		return CallLeg{}, LegKnown, nil
	}
	if number = strings.TrimPrefix(number, "+"); number != "" {
		leg, connection, err := s.connectWaitingLeg(ctx, scope, callType, callID, sessionID, participant, number, at, left)
		if err != nil || connection != NothingWaiting {
			return leg, connection, err
		}
	}
	return s.connectWaitingLeg(ctx, scope, callType, callID, sessionID, participant, "", at, left)
}

// connectWaitingLeg gives the phone the first leg of the call, across the shards, that is
// waiting for one and, when number is not empty, that a phone of number may be: one whose
// remote number it is first.
func (s *Store) connectWaitingLeg(ctx context.Context, scope AppScope, callType, callID, sessionID, participant, number string,
	at time.Time, left *time.Time) (CallLeg, Connection, error) {
	for _, db := range s.shards() {
		waiting := db.NewSelect().Model((*CallLeg)(nil)).Column("id").
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("ended_at IS NULL").Where("participant_session = ''").
			WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
				return q.Where("direction = ?", Inbound).WhereOr("started_at IS NULL")
			}).
			ApplyQueryBuilder(scope.where)
		if number != "" {
			waiting = waiting.WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
				return q.Where("ltrim(remote_number, '+') = ?", number).
					WhereOr("direction = ? AND ltrim(local_number, '+') = ?", Outbound, number)
			}).OrderExpr("ltrim(remote_number, '+') = ? DESC", number)
		}
		waiting = waiting.OrderExpr("direction = ? DESC, dialed_at", Inbound).Limit(1).For("UPDATE SKIP LOCKED")
		update := db.NewUpdate().Model((*CallLeg)(nil)).
			Set("participant_session = ?", participant).
			Set("stream_session_id = CASE WHEN stream_session_id = '' THEN ? ELSE stream_session_id END", sessionID).
			Set("started_at = COALESCE(started_at, ?)", at)
		if left != nil {
			update = update.Set("ended_at = GREATEST(?::timestamptz, COALESCE(started_at, ?::timestamptz))",
				left.UTC().Truncate(time.Microsecond), at).
				Set("end_reason = ?", LegLeft)
		}
		var connected []CallLeg
		err := update.Where("id IN (?)", waiting).Returning("*").Scan(ctx, &connected)
		if isViolation(err, uniqueViolation) {
			return CallLeg{}, LegKnown, nil
		}
		if err != nil {
			return CallLeg{}, NothingWaiting, stack.Wrap(fmt.Errorf("store: connect call leg: %w", err))
		}
		if len(connected) > 0 {
			return connected[0], LegConnected, nil
		}
	}
	return CallLeg{}, NothingWaiting, nil
}

// RecordParticipant keeps, in this shard, a phone's join (JoinedAt) or leave (LeftAt) of a
// call that no leg was waiting for, in the scope of the hook that delivered it, at now. A
// second event of the same phone fills in what the first left empty: its first join and its
// first leave are what is kept.
func (s *Store) RecordParticipant(ctx context.Context, scope AppScope, kept CallParticipant, now time.Time) error {
	kept.StreamAppPK, kept.Unpinned = scope.App, scope.Unpinned
	kept.Number = strings.TrimPrefix(kept.Number, "+")
	kept.RecordedAt = now.UTC().Truncate(time.Microsecond)
	for _, at := range []**time.Time{&kept.JoinedAt, &kept.LeftAt} {
		if *at != nil {
			truncated := (*at).UTC().Truncate(time.Microsecond)
			*at = &truncated
		}
	}
	_, err := s.shard.NewInsert().Model(&kept).
		On("CONFLICT (call_type, call_id, COALESCE(stream_app_pk, 0), participant_session) DO UPDATE").
		Set("joined_at = COALESCE(cp.joined_at, EXCLUDED.joined_at)").
		Set("left_at = COALESCE(cp.left_at, EXCLUDED.left_at)").
		Set("number = CASE WHEN cp.number = '' THEN EXCLUDED.number ELSE cp.number END").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: record call participant: %w", err))
	}
	return nil
}

// HeardCall records, in this shard, that the call hook delivered a participant event of the
// call, anybody's: a row with no participant session. A hook subscribed to them delivers the
// agent's own join on every placed call, so a placed call ending with none heard is a hook that
// does not deliver them.
func (s *Store) HeardCall(ctx context.Context, scope AppScope, callType, callID string, now time.Time) error {
	_, err := s.shard.NewInsert().Model(&CallParticipant{
		StreamAppPK: scope.App, Unpinned: scope.Unpinned, CallType: callType, CallID: callID,
		RecordedAt: now.UTC().Truncate(time.Microsecond),
	}).On("CONFLICT DO NOTHING").Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: heard call: %w", err))
	}
	return nil
}

// CallHeard reports whether the call hook delivered any participant event of the call
// (HeardCall), in any shard.
func (s *Store) CallHeard(ctx context.Context, scope AppScope, callType, callID string) (bool, error) {
	heard, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (bool, error) {
		return db.NewSelect().Model((*CallParticipant)(nil)).
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			ApplyQueryBuilder(scope.where).Exists(ctx)
	})
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: call heard: %w", err))
	}
	return slices.Contains(heard, true), nil
}

// ParticipantLeft is when a phone left a call, from the leave kept when no leg had it, or nil.
func (s *Store) ParticipantLeft(ctx context.Context, scope AppScope, callType, callID, participant string) (*time.Time, error) {
	lefts, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]time.Time, error) {
		var left []time.Time
		err := db.NewSelect().Model((*CallParticipant)(nil)).Column("left_at").
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("participant_session = ?", participant).Where("left_at IS NOT NULL").
			ApplyQueryBuilder(scope.where).Scan(ctx, &left)
		return left, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: participant left: %w", err))
	}
	all := slices.Concat(lefts...)
	if len(all) == 0 {
		return nil, nil
	}
	first := slices.MinFunc(all, time.Time.Compare)
	return &first, nil
}

// ParticipantHasLeg reports whether a leg of the call, open or ended, is the phone's.
func (s *Store) ParticipantHasLeg(ctx context.Context, scope AppScope, callType, callID, participant string) (bool, error) {
	known, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (bool, error) {
		return db.NewSelect().Model((*CallLeg)(nil)).
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("participant_session = ?", participant).ApplyQueryBuilder(scope.where).Exists(ctx)
	})
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: participant has leg: %w", err))
	}
	return slices.Contains(known, true), nil
}

// ForgetParticipant deletes what was kept of a phone that has its leg now: its join, and its
// leave only when its leg was ended at it (applied) or later. A leave kept after that is left
// for the Left that kept it, which ends the leg itself.
func (s *Store) ForgetParticipant(ctx context.Context, scope AppScope, callType, callID, participant string,
	applied *time.Time) error {
	_, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (struct{}, error) {
		_, err := db.NewDelete().Model((*CallParticipant)(nil)).
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("participant_session = ?", participant).ApplyQueryBuilder(scope.where).
			WhereGroup(" AND ", func(q *bun.DeleteQuery) *bun.DeleteQuery {
				q = q.Where("left_at IS NULL")
				if applied != nil {
					q = q.WhereOr("left_at <= ?", applied.UTC())
				}
				return q
			}).
			Exec(ctx)
		return struct{}{}, err
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: forget call participant: %w", err))
	}
	return nil
}

// DeleteCallParticipants deletes, from this shard, at most limit of the participants kept
// before before, and reports how many.
func (s *Store) DeleteCallParticipants(ctx context.Context, before time.Time, limit int) (int, error) {
	result, err := s.shard.NewDelete().Model((*CallParticipant)(nil)).
		Where("id IN (?)", s.shard.NewSelect().Model((*CallParticipant)(nil)).Column("id").
			Where("recorded_at < ?", before.UTC()).OrderExpr("recorded_at").Limit(limit)).
		Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: delete call participants: %w", err))
	}
	deleted, err := result.RowsAffected()
	return int(deleted), err
}

// EndParticipantLeg ends the open leg of a phone that left a call, at at, and returns it.
func (s *Store) EndParticipantLeg(ctx context.Context, scope AppScope, callType, callID, participant string,
	at time.Time) ([]CallLeg, error) {
	return s.endLegs(ctx, func(q *bun.UpdateQuery) *bun.UpdateQuery {
		return q.Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("participant_session = ?", participant).ApplyQueryBuilder(scope.where)
	}, LegLeft, at)
}

// EndCallLegs ends the legs of one call still open in one Stream app, at at, and returns them:
// one whose phone joined is ended, and one still ringing is unanswered. A call's id is only
// unique within its app, and its legs may have been opened from any shard. A leg already
// ended is not ended again, so a repeated event ends nothing.
func (s *Store) EndCallLegs(ctx context.Context, scope AppScope, callType, callID string, at time.Time) ([]CallLeg, error) {
	return s.endLegs(ctx, func(q *bun.UpdateQuery) *bun.UpdateQuery {
		return q.Where("call_type = ?", callType).Where("call_id = ?", callID).ApplyQueryBuilder(scope.where)
	}, LegEnded, at)
}

// endLegs ends the open legs where narrows to, in every shard, at at or when each began if
// that is later, for reason, or as unanswered when the leg's phone never joined.
func (s *Store) endLegs(ctx context.Context, where func(*bun.UpdateQuery) *bun.UpdateQuery, reason string,
	at time.Time) ([]CallLeg, error) {
	at = at.UTC().Truncate(time.Microsecond)
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]CallLeg, error) {
		var ended []CallLeg
		err := where(db.NewUpdate().Model((*CallLeg)(nil)).
			Set("ended_at = GREATEST(?, COALESCE(started_at, dialed_at))", at).
			Set("end_reason = CASE WHEN started_at IS NULL THEN ? ELSE ? END", LegUnanswered, reason).
			Where("ended_at IS NULL")).
			Returning("*").
			Scan(ctx, &ended)
		return ended, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: end call legs: %w", err))
	}
	return slices.Concat(lists...), nil
}

// SweepCallLegs ends the legs of this shard open longer than longest at now: one whose phone
// joined before now less longest is ended at its start plus longest, as swept, and one that has
// rung since before then as unanswered. At most limit of each are ended at once.
func (s *Store) SweepCallLegs(ctx context.Context, now time.Time, longest time.Duration, limit int) ([]CallLeg, error) {
	since := now.UTC().Add(-longest)
	var swept, unanswered []CallLeg
	err := s.shard.NewUpdate().Model((*CallLeg)(nil)).
		Set("ended_at = started_at + make_interval(secs => ?)", longest.Seconds()).
		Set("end_reason = ?", LegSwept).
		Where("id IN (?)", s.shard.NewSelect().Model((*CallLeg)(nil)).Column("id").
			Where("ended_at IS NULL").Where("started_at < ?", since).OrderExpr("started_at").Limit(limit)).
		Where("ended_at IS NULL").
		Returning("*").
		Scan(ctx, &swept)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: sweep call legs: %w", err))
	}
	err = s.shard.NewUpdate().Model((*CallLeg)(nil)).
		Set("ended_at = ?", now.UTC().Truncate(time.Microsecond)).
		Set("end_reason = ?", LegUnanswered).
		Where("id IN (?)", s.shard.NewSelect().Model((*CallLeg)(nil)).Column("id").
			Where("ended_at IS NULL").Where("started_at IS NULL").Where("dialed_at < ?", since).
			OrderExpr("dialed_at").Limit(limit)).
		Where("ended_at IS NULL").
		Returning("*").
		Scan(ctx, &unanswered)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: sweep call legs: %w", err))
	}
	return append(swept, unanswered...), nil
}

// EndedCallLegs is the legs of this shard whose phone joined and that ended in [from, to),
// oldest first, at most limit: what the sweep checks were billed.
func (s *Store) EndedCallLegs(ctx context.Context, from, to time.Time, limit int) ([]CallLeg, error) {
	var ended []CallLeg
	err := s.shard.NewSelect().Model(&ended).
		Where("started_at IS NOT NULL").Where("ended_at >= ?", from.UTC()).Where("ended_at < ?", to.UTC()).
		OrderExpr("ended_at").Limit(limit).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: ended call legs: %w", err))
	}
	return ended, nil
}

// NumbersHeldSince is every number held at some time after since: still held, or released
// after it. It is what a month of rental is billed from.
func (s *Store) NumbersHeldSince(ctx context.Context, since time.Time) ([]PhoneNumber, error) {
	var held []PhoneNumber
	err := s.db.NewSelect().Model(&held).
		WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
			return q.Where("released_at IS NULL").WhereOr("released_at > ?", since.UTC())
		}).
		Order("id").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: numbers held since: %w", err))
	}
	return held, nil
}

// RecordedUsage is which of the usage ids any shard holds a row for: work billed before, which
// is not billed again.
func (s *Store) RecordedUsage(ctx context.Context, usageIDs []uuid.UUID) (map[uuid.UUID]bool, error) {
	recorded := make(map[uuid.UUID]bool, len(usageIDs))
	for batch := range slices.Chunk(usageIDs, 1000) {
		lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]uuid.UUID, error) {
			var found []uuid.UUID
			err := db.NewSelect().Model((*Request)(nil)).Column("usage_id").
				Where("usage_id IN (?)", bun.In(batch)).Scan(ctx, &found)
			return found, err
		})
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("store: recorded usage: %w", err))
		}
		for _, id := range slices.Concat(lists...) {
			recorded[id] = true
		}
	}
	return recorded, nil
}
