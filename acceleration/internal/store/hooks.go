package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// AppScope is the Stream app a hook came from, as the rows it may act on are found by.
type AppScope struct {
	// App is the app's id. Zero matches only unpinned rows.
	App int64
	// Unpinned also matches rows with no pin, which were written into the deployment's
	// own app. Only a hook from the deployment's own app sets it.
	Unpinned bool
}

// where narrows a query to the rows in the scope's app.
func (a AppScope) where(query *bun.SelectQuery) *bun.SelectQuery {
	return query.WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
		if a.App != 0 {
			q = q.WhereOr("stream_app_pk = ?", a.App)
		}
		if a.Unpinned {
			q = q.WhereOr("stream_app_pk IS NULL")
		}
		if a.App == 0 && !a.Unpinned {
			q = q.Where("false")
		}
		return q
	})
}

// ErrAmbiguousHook is a hook that matches rows of more than one customer in its app, which
// is acted on for nobody rather than for whichever came first.
var ErrAmbiguousHook = errors.New("store: more than one customer holds what that hook names")

// StreamAppByPK is the app registered for a Stream app id, with its keys.
func (s *Store) StreamAppByPK(ctx context.Context, app int64) (StreamApp, error) {
	var found StreamApp
	err := s.db.NewSelect().Model(&found).Column("customer_id").Where("stream_app_pk = ?", app).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return StreamApp{}, ErrNoStreamApp
	}
	if err != nil {
		return StreamApp{}, fmt.Errorf("store: find stream app: %w", err)
	}
	return s.StreamApp(ctx, found.CustomerID)
}

// NumberByCallInApp is NumberByCall, among the numbers attached in one app.
func (s *Store) NumberByCallInApp(ctx context.Context, scope AppScope, callType, callID string) (PhoneNumber, error) {
	number, err := s.NumberByCall(ctx, callType, callID)
	if err != nil {
		return PhoneNumber{}, err
	}
	// The call names a number; the number has to have been attached in the hook's app.
	var held []PhoneNumber
	if err := scope.where(s.db.NewSelect().Model(&held).
		Where("e164 = ?", number.E164).Where("released_at IS NULL")).Scan(ctx); err != nil {
		return PhoneNumber{}, fmt.Errorf("store: number by call: %w", err)
	}
	switch customers(held, func(n PhoneNumber) string { return n.CustomerID }) {
	case 0:
		return PhoneNumber{}, fmt.Errorf("store: no number in that app reaches call %s:%s", callType, callID)
	case 1:
		return held[0], nil
	}
	return PhoneNumber{}, ErrAmbiguousHook
}

// CallByAgentInApp is CallByAgent, among the calls made in one app.
func (s *Store) CallByAgentInApp(ctx context.Context, scope AppScope, agentID string) (Call, error) {
	if agentID == "" {
		return Call{}, errors.New("store: an agent id is required")
	}
	var calls []Call
	err := scope.where(s.db.NewSelect().Model(&calls).Where("agent_id = ?", agentID)).
		Order("started_at DESC").Limit(20).Scan(ctx)
	if err != nil {
		return Call{}, fmt.Errorf("store: call by agent: %w", err)
	}
	switch customers(calls, func(c Call) string { return c.CustomerID }) {
	case 0:
		return Call{}, unknownCall(agentID)
	case 1:
		return calls[0], nil
	}
	return Call{}, ErrAmbiguousHook
}

// AgentConfigFor is a config a customer holds, by id.
func (s *Store) AgentConfigFor(ctx context.Context, customerID, id string) (AgentConfig, error) {
	config, err := s.AgentConfigOwner(ctx, id)
	if err != nil {
		return AgentConfig{}, err
	}
	if config.CustomerID != customerID {
		return AgentConfig{}, unknownAgentConfig(id)
	}
	return config, nil
}

func customers[T any](rows []T, of func(T) string) int {
	seen := map[string]bool{}
	for _, row := range rows {
		seen[of(row)] = true
	}
	return len(seen)
}

// deliveryKeptFor is how long a delivery is remembered, which is longer than Stream retries.
const deliveryKeptFor = 24 * time.Hour

// FirstDelivery records a hook delivery and reports whether it is the first with that key.
func (s *Store) FirstDelivery(ctx context.Context, key string) (bool, error) {
	if _, err := s.db.ExecContext(ctx, "DELETE FROM hook_deliveries WHERE seen_at < ?",
		time.Now().UTC().Add(-deliveryKeptFor)); err != nil {
		return false, fmt.Errorf("store: forget hook deliveries: %w", err)
	}
	result, err := s.db.ExecContext(ctx,
		"INSERT INTO hook_deliveries (key, seen_at) VALUES (?, now()) ON CONFLICT (key) DO NOTHING", key)
	if err != nil {
		return false, fmt.Errorf("store: record hook delivery: %w", err)
	}
	inserted, _ := result.RowsAffected()
	return inserted == 1, nil
}
