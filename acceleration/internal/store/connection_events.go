package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoConnectionEventSubscription is a token or an id no subscription has.
var ErrNoConnectionEventSubscription = errors.New("store: no such connection event subscription")

// How far a connection event subscription has got.
const (
	// ConnectionEventPending is a subscription the server has not accepted yet.
	ConnectionEventPending = "pending"
	// ConnectionEventActive is one the server accepted, and delivers to.
	ConnectionEventActive = "active"
	// ConnectionEventFailed is one the server refused, Error saying why.
	ConnectionEventFailed = "failed"
)

// ConnectionEventSubscription is one MCP Events webhook subscription on a connection's server,
// for one event one binding of one agent config declares (internal/mcpevents).
type ConnectionEventSubscription struct {
	bun.BaseModel `bun:"table:connection_event_subscriptions,alias:ces"`

	ID           string `bun:"id,pk"`
	CustomerID   string `bun:"customer_id,notnull"`
	ConnectionID string `bun:"connection_id,notnull"`
	ConfigID     string `bun:"config_id,notnull"`
	// Binding is the alias of the config's binding that declares the event.
	Binding   string         `bun:"binding,notnull"`
	Event     string         `bun:"event,notnull"`
	Arguments map[string]any `bun:"arguments,type:jsonb,notnull"`
	// Key is the event and its arguments, hashed (mcpevents.Key).
	Key string `bun:"key,notnull"`
	// Token is the callback's path segment.
	Token string `bun:"token,notnull"`
	// SecretSealed is the subscription's own signing secret, sealed under KEKVersion.
	SecretSealed []byte `bun:"secret_sealed,notnull"`
	KEKVersion   int    `bun:"kek_version,notnull"`
	// RemoteID is the id the server gave the subscription.
	RemoteID string `bun:"remote_id,notnull"`
	// RefreshBefore is when the server stops delivering unless asked again. Nil is a grant that
	// does not expire, or none yet.
	RefreshBefore *time.Time `bun:"refresh_before"`
	Status        string     `bun:"status,notnull"`
	Error         string     `bun:"error,notnull"`
	// NextAttemptAt is when a worker next asks the server for it. Nil is never.
	NextAttemptAt *time.Time `bun:"next_attempt_at"`
	// Failures is how many times in a row the server refused it.
	Failures  int       `bun:"failures,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	UpdatedAt time.Time `bun:"updated_at,notnull"`
}

// AddConnectionEventSubscription stores a new subscription unless one for the same connection,
// config, binding and key is already there, and reports whether it stored it. Two routers
// adding the same one at once store it once (connection_event_subscriptions_one).
func (s *Store) AddConnectionEventSubscription(ctx context.Context, sub *ConnectionEventSubscription) (bool, error) {
	if sub.CustomerID == "" || sub.ConnectionID == "" || sub.ConfigID == "" || sub.Binding == "" || sub.Event == "" {
		return false, stack.Wrap(errors.New("store: a customer, a connection, a config, a binding and an event are required"))
	}
	now := time.Now().UTC()
	sub.ID, sub.CreatedAt, sub.UpdatedAt = newID(), now, now
	if sub.Arguments == nil {
		sub.Arguments = map[string]any{}
	}
	result, err := s.db.NewInsert().Model(sub).
		On("CONFLICT (connection_id, config_id, binding, key) DO NOTHING").
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: add connection event subscription: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: add connection event subscription: %w", err))
	}
	return affected == 1, nil
}

// ConnectionEventSubscriptionByToken is the subscription a callback's token names.
func (s *Store) ConnectionEventSubscriptionByToken(ctx context.Context, token string) (ConnectionEventSubscription, error) {
	if token == "" {
		return ConnectionEventSubscription{}, ErrNoConnectionEventSubscription
	}
	var sub ConnectionEventSubscription
	err := s.db.NewSelect().Model(&sub).Where("token = ?", token).Limit(1).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectionEventSubscription{}, ErrNoConnectionEventSubscription
	}
	if err != nil {
		return ConnectionEventSubscription{}, stack.Wrap(fmt.Errorf("store: connection event subscription: %w", err))
	}
	return sub, nil
}

// ConnectionEventSubscriptionOf is the subscription one binding of one config has for an event
// (its Key) on a connection.
func (s *Store) ConnectionEventSubscriptionOf(ctx context.Context, connectionID, configID, binding, key string) (ConnectionEventSubscription, error) {
	var sub ConnectionEventSubscription
	err := s.db.NewSelect().Model(&sub).
		Where("connection_id = ?", connectionID).
		Where("config_id = ?", configID).
		Where("binding = ?", binding).
		Where("key = ?", key).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectionEventSubscription{}, ErrNoConnectionEventSubscription
	}
	if err != nil {
		return ConnectionEventSubscription{}, stack.Wrap(fmt.Errorf("store: connection event subscription: %w", err))
	}
	return sub, nil
}

// DueOtherConnectionEventSubscriptions makes due at now every subscription of a connection for
// an event (its Key) but the one with id.
func (s *Store) DueOtherConnectionEventSubscriptions(ctx context.Context, connectionID, key, id string, now time.Time) error {
	_, err := s.db.NewUpdate().Model((*ConnectionEventSubscription)(nil)).
		Set("next_attempt_at = ?", now.UTC()).
		Where("connection_id = ?", connectionID).
		Where("key = ?", key).
		Where("id != ?", id).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: due connection event subscriptions: %w", err))
	}
	return nil
}

// DueConnectionEventSubscription makes one subscription due at now, so the next worker to look
// asks the server for it again, or drops it.
func (s *Store) DueConnectionEventSubscription(ctx context.Context, id string, now time.Time) error {
	_, err := s.db.NewUpdate().Model((*ConnectionEventSubscription)(nil)).
		Set("next_attempt_at = ?", now.UTC()).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: due connection event subscription: %w", err))
	}
	return nil
}

// DueConnectionEventSubscriptions makes every subscription of a connection due at now, so the
// next worker to look asks the server for each again, or drops it.
func (s *Store) DueConnectionEventSubscriptions(ctx context.Context, customerID, connectionID string, now time.Time) error {
	_, err := s.db.NewUpdate().Model((*ConnectionEventSubscription)(nil)).
		Set("next_attempt_at = ?", now.UTC()).
		Where("customer_id = ?", customerID).
		Where("connection_id = ?", connectionID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: due connection event subscriptions: %w", err))
	}
	return nil
}

// claimConnectionEventSubscriptionsQuery takes at most limit due rows and pushes each one's
// next_attempt_at to the lease's end, so no other router takes it until the lease runs out.
// due checks next_attempt_at again on the row it locks, as claimEventDeliveriesQuery does: a
// row another router leased after the statement started is skipped.
const claimConnectionEventSubscriptionsQuery = `
WITH due AS (
    SELECT id FROM connection_event_subscriptions
    WHERE next_attempt_at <= ?
    ORDER BY next_attempt_at
    LIMIT ?
    FOR UPDATE SKIP LOCKED
)
UPDATE connection_event_subscriptions AS ces
SET next_attempt_at = ?
FROM due
WHERE ces.id = due.id AND ces.next_attempt_at <= ?
RETURNING ces.*`

// ClaimConnectionEventSubscriptions takes at most limit subscriptions due at now, each leased
// until leaseUntil.
func (s *Store) ClaimConnectionEventSubscriptions(ctx context.Context, now time.Time, limit int, leaseUntil time.Time) ([]ConnectionEventSubscription, error) {
	return s.claimConnectionEventSubscriptions(ctx, claimConnectionEventSubscriptionsQuery, now, limit, leaseUntil)
}

// claimConnectionEventSubscriptions runs query, ClaimConnectionEventSubscriptions' statement or
// a test's variant of it with a pause in it.
func (s *Store) claimConnectionEventSubscriptions(ctx context.Context, query string, now time.Time, limit int, leaseUntil time.Time) ([]ConnectionEventSubscription, error) {
	claimed := []ConnectionEventSubscription{}
	if limit < 1 {
		return claimed, nil
	}
	err := s.db.NewRaw(query, now.UTC(), limit, leaseUntil.UTC(), now.UTC()).Scan(ctx, &claimed)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim connection event subscriptions: %w", err))
	}
	return claimed, nil
}

// NextConnectionEventSubscriptionAt is when the subscription due first is due, any router's.
// found is false when none is ever due: there are no subscriptions, or none expires.
func (s *Store) NextConnectionEventSubscriptionAt(ctx context.Context) (next time.Time, found bool, err error) {
	var at sql.NullTime
	if err := s.db.NewRaw("SELECT min(next_attempt_at) FROM connection_event_subscriptions").Scan(ctx, &at); err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next connection event subscription: %w", err))
	}
	return at.Time, at.Valid, nil
}

// SaveConnectionEventSubscription writes what asking the server for a subscription came to.
func (s *Store) SaveConnectionEventSubscription(ctx context.Context, sub *ConnectionEventSubscription) error {
	sub.UpdatedAt = time.Now().UTC()
	_, err := s.db.NewUpdate().Model(sub).
		Column("remote_id", "refresh_before", "status", "error", "next_attempt_at", "failures", "updated_at").
		Where("id = ?", sub.ID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save connection event subscription: %w", err))
	}
	return nil
}

// DeleteConnectionEventSubscription drops a subscription, so deliveries to its token are
// refused. One already gone is not an error.
func (s *Store) DeleteConnectionEventSubscription(ctx context.Context, id string) error {
	_, err := s.db.NewDelete().Model((*ConnectionEventSubscription)(nil)).Where("id = ?", id).Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete connection event subscription: %w", err))
	}
	return nil
}

// DeleteConnectionEventSubscriptions drops every subscription of a connection.
func (s *Store) DeleteConnectionEventSubscriptions(ctx context.Context, customerID, connectionID string) error {
	_, err := s.db.NewDelete().Model((*ConnectionEventSubscription)(nil)).
		Where("customer_id = ?", customerID).
		Where("connection_id = ?", connectionID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete connection event subscriptions: %w", err))
	}
	return nil
}

// ClaimConnectionEvent records that a subscription's event arrived, and reports whether this
// is the first time: a retried delivery of the same event id is not.
func (s *Store) ClaimConnectionEvent(ctx context.Context, subscriptionID, eventID string) (bool, error) {
	result, err := s.db.NewRaw(
		"INSERT INTO connection_event_deliveries (subscription_id, event_id, received_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
		subscriptionID, eventID, time.Now().UTC()).Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim connection event: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: claim connection event: %w", err))
	}
	return affected == 1, nil
}
