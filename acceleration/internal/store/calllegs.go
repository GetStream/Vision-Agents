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

// OpenCallLeg records a leg of a call that has connected, in this shard. It reports false,
// and records nothing, for an inbound leg already open on the same call: every delivery of
// the event announcing a call that arrives opens it, and only the first does.
func (s *Store) OpenCallLeg(ctx context.Context, leg *CallLeg) (bool, error) {
	switch {
	case leg.CustomerID == "":
		return false, stack.Wrap(errors.New("store: a call leg needs a customer"))
	case leg.CallType == "" || leg.CallID == "":
		return false, stack.Wrap(errors.New("store: a call leg needs a call"))
	case leg.Direction != Inbound && leg.Direction != Outbound:
		return false, stack.Wrap(fmt.Errorf("store: a call leg is inbound or outbound, not %q", leg.Direction))
	}
	if leg.StartedAt.IsZero() {
		leg.StartedAt = time.Now()
	}
	leg.StartedAt = leg.StartedAt.UTC().Truncate(time.Microsecond)
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

// EndCallLegs ends the legs of one call still open in one Stream app, at at, and returns
// them. A call's id is only unique within its app, and its legs may have been opened from
// any shard. A leg already ended is not ended again, so a repeated event ends nothing.
func (s *Store) EndCallLegs(ctx context.Context, scope AppScope, callType, callID string, at time.Time) ([]CallLeg, error) {
	at = at.UTC().Truncate(time.Microsecond)
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]CallLeg, error) {
		var ended []CallLeg
		err := db.NewUpdate().Model((*CallLeg)(nil)).
			Set("ended_at = GREATEST(?, started_at)", at).
			Where("call_type = ?", callType).
			Where("call_id = ?", callID).
			Where("ended_at IS NULL").
			ApplyQueryBuilder(scope.where).
			Returning("*").
			Scan(ctx, &ended)
		return ended, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: end call legs: %w", err))
	}
	return slices.Concat(lists...), nil
}

// NumbersHeldSince is every number that costs something a month and was held at some time
// after since: still held, or released after it. It is what a month of rental is billed from.
func (s *Store) NumbersHeldSince(ctx context.Context, since time.Time) ([]PhoneNumber, error) {
	var held []PhoneNumber
	err := s.db.NewSelect().Model(&held).
		Where("monthly_cost_micros > 0").
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

// RecordedUsage is which of the usage ids any shard holds a row for: work a job billed
// before, which it does not bill again.
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

// EndCallLegsOpenSince ends the legs of this shard that opened before since and are still
// open, each at its start plus longest, and returns them: a call whose end was never
// announced is not billed forever. At most limit are ended at once.
func (s *Store) EndCallLegsOpenSince(ctx context.Context, since time.Time, longest time.Duration, limit int) ([]CallLeg, error) {
	var ended []CallLeg
	err := s.shard.NewUpdate().Model((*CallLeg)(nil)).
		Set("ended_at = started_at + make_interval(secs => ?)", longest.Seconds()).
		Where("id IN (?)", s.shard.NewSelect().Model((*CallLeg)(nil)).Column("id").
			Where("ended_at IS NULL").Where("started_at < ?", since.UTC()).
			OrderExpr("started_at").Limit(limit)).
		Where("ended_at IS NULL").
		Returning("*").
		Scan(ctx, &ended)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: end stale call legs: %w", err))
	}
	return ended, nil
}
