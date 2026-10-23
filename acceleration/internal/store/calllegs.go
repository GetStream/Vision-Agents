package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// OpenCallLeg records a leg of a call in this shard: an outbound one ringing, an inbound one
// started. It reports false, and records nothing, for an inbound leg its Stream session already
// has, ended or not: every delivery of the event announcing a call that arrives opens it, and
// only the first does.
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
	return opened > 0, nil
}

// ConnectCallLeg gives a phone that joined a call, at at, the leg it is: the call's inbound
// leg that does not know its phone yet, or else the oldest outbound leg still ringing, which
// starts then. It reports false when the phone already has its leg, from an earlier delivery,
// or no leg of the call is waiting for one.
func (s *Store) ConnectCallLeg(ctx context.Context, scope AppScope, callType, callID, sessionID, participant string,
	at time.Time) (CallLeg, bool, error) {
	at = at.UTC().Truncate(time.Microsecond)
	known, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (bool, error) {
		return db.NewSelect().Model((*CallLeg)(nil)).
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("participant_session = ?", participant).ApplyQueryBuilder(scope.where).Exists(ctx)
	})
	if err != nil {
		return CallLeg{}, false, stack.Wrap(fmt.Errorf("store: connect call leg: %w", err))
	}
	if slices.Contains(known, true) {
		return CallLeg{}, false, nil
	}
	for _, db := range s.shards() {
		waiting := db.NewSelect().Model((*CallLeg)(nil)).Column("id").
			Where("call_type = ?", callType).Where("call_id = ?", callID).
			Where("ended_at IS NULL").Where("participant_session = ''").
			WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
				return q.Where("direction = ?", Inbound).WhereOr("started_at IS NULL")
			}).
			ApplyQueryBuilder(scope.where).
			OrderExpr("direction = ? DESC, dialed_at", Inbound).Limit(1).For("UPDATE SKIP LOCKED")
		var connected []CallLeg
		err := db.NewUpdate().Model((*CallLeg)(nil)).
			Set("participant_session = ?", participant).
			Set("stream_session_id = CASE WHEN stream_session_id = '' THEN ? ELSE stream_session_id END", sessionID).
			Set("started_at = COALESCE(started_at, ?)", at).
			Where("id IN (?)", waiting).
			Returning("*").
			Scan(ctx, &connected)
		if isViolation(err, uniqueViolation) {
			return CallLeg{}, false, nil
		}
		if err != nil {
			return CallLeg{}, false, stack.Wrap(fmt.Errorf("store: connect call leg: %w", err))
		}
		if len(connected) > 0 {
			return connected[0], true, nil
		}
	}
	return CallLeg{}, false, nil
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
