package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoPhoneCall is what a phone call no shard holds reads as.
var ErrNoPhoneCall = errors.New("store: no such phone call")

// PhoneCallWhere finds one phone call: by Token when it is set, else by the rest. The newest
// match is the one.
type PhoneCallWhere struct {
	Token        string
	CustomerID   string
	StreamAppPK  int64
	CallType     string
	CallID       string
	LocalNumber  string
	RemoteNumber string
}

func (w PhoneCallWhere) on(q *bun.SelectQuery) *bun.SelectQuery {
	if w.Token != "" {
		return q.Where("pc.token = ?", w.Token)
	}
	return q.Where("pc.customer_id = ?", w.CustomerID).
		Where("COALESCE(pc.stream_app_pk, 0) = ?", w.StreamAppPK).
		Where("pc.call_type = ?", w.CallType).
		Where("pc.call_id = ?", w.CallID).
		Where("pc.local_number = ?", w.LocalNumber).
		Where("pc.remote_number = ?", w.RemoteNumber)
}

// InsertPhoneCall saves a phone call this shard is placing.
func (s *Store) InsertPhoneCall(ctx context.Context, call *PhoneCall) error {
	if _, err := s.shard.NewInsert().Model(call).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: insert phone call: %w", err))
	}
	return nil
}

// DeletePhoneCall removes the phone call with token, from whichever shard holds it.
func (s *Store) DeletePhoneCall(ctx context.Context, token string) error {
	_, err := onEveryShard(ctx, s, func(ctx context.Context, db bun.IDB) (sql.Result, error) {
		return db.NewDelete().Model((*PhoneCall)(nil)).Where("token = ?", token).Exec(ctx)
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete phone call: %w", err))
	}
	return nil
}

// ChangePhoneCall locks the row where finds, in whichever shard holds it, hands it to change,
// and writes back what change left. change returning false writes nothing. It returns the row
// as it now stands, or ErrNoPhoneCall.
//
// The lock is what makes concurrent changes safe: a vendor's post and the call's own events
// can arrive at once, and each must see what the other wrote.
func (s *Store) ChangePhoneCall(ctx context.Context, where PhoneCallWhere, change func(*PhoneCall) bool) (PhoneCall, error) {
	// A vendor's post lands on whichever shard is nearest it, not the one that placed the call.
	for _, db := range s.shards() {
		var call PhoneCall
		err := db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			err := where.on(tx.NewSelect().Model(&call)).OrderExpr("pc.id DESC").Limit(1).For("UPDATE").Scan(ctx)
			if err != nil {
				return err
			}
			changed := call
			if !change(&changed) {
				return nil
			}
			if _, err := tx.NewUpdate().Model(&changed).WherePK().Exec(ctx); err != nil {
				return err
			}
			call = changed
			return nil
		})
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return PhoneCall{}, stack.Wrap(fmt.Errorf("store: change phone call: %w", err))
		}
		return call, nil
	}
	return PhoneCall{}, ErrNoPhoneCall
}

// PhoneCalls returns the customer's newest phone call for each of callIDs, from every shard.
// A call id with none is left out.
func (s *Store) PhoneCalls(ctx context.Context, customerID string, callIDs []string) ([]PhoneCall, error) {
	if len(callIDs) == 0 {
		return nil, nil
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]PhoneCall, error) {
		var calls []PhoneCall
		err := db.NewSelect().Model(&calls).
			DistinctOn("pc.call_id").
			Where("pc.customer_id = ?", customerID).
			Where("pc.call_id IN (?)", bun.In(callIDs)).
			OrderExpr("pc.call_id, pc.id DESC").
			Scan(ctx)
		return calls, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: phone calls: %w", err))
	}
	newest := map[string]PhoneCall{}
	for _, call := range slices.Concat(lists...) {
		if kept, ok := newest[call.CallID]; !ok || call.PlacedAt.After(kept.PlacedAt) {
			newest[call.CallID] = call
		}
	}
	return slices.Collect(maps.Values(newest)), nil
}

// ClearPhoneCallTokens clears the token of every call this shard placed before placedBefore,
// so a vendor's post for it is refused, and keeps the rows. Each shard clears its own.
func (s *Store) ClearPhoneCallTokens(ctx context.Context, placedBefore time.Time) (int64, error) {
	result, err := s.shard.NewUpdate().Model((*PhoneCall)(nil)).
		Set("token = NULL").
		Where("token IS NOT NULL").
		Where("placed_at < ?", placedBefore).
		Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: clear phone call tokens: %w", err))
	}
	cleared, err := result.RowsAffected()
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: clear phone call tokens: %w", err))
	}
	return cleared, nil
}
