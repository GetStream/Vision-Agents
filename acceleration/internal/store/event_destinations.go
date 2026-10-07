package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// MaxEventDestinations is how many event destinations a customer may have for one connector:
// the three Vercel Connect allows a connector («A connector can have up to three trigger
// destinations», https://vercel.com/docs/connect/concepts/triggers, opened October 6, 2026).
// One provider event is one send to each, so the cap bounds the fan-out of one event.
const MaxEventDestinations = 3

// eventDestinationLockSeed is the second argument to hashtextextended for the lock creates of
// one customer's destinations of one connector take turns under, so two creates at once cannot
// both find room for a third. 875 is this table's issue, AI-875, as definitionLockSeed's 833 is
// its own.
const eventDestinationLockSeed = 875

const (
	defaultEventDestinationLimit = 25
	maxEventDestinationLimit     = 200
)

// What an event destination is sent (EventDestination.Forward).
const (
	// ForwardUnhandled is a delivery the router acted on in no way: no signal, and no message
	// an agent of the customer answers.
	ForwardUnhandled = "unhandled"
	// ForwardAll is every verified delivery but a URL handshake.
	ForwardAll = "all"
)

// ErrNoEventDestination says the customer has no such destination for the connector.
var ErrNoEventDestination = errors.New("store: no such event destination")

// ErrEventDestinationsFull says the customer already has MaxEventDestinations for the connector.
var ErrEventDestinationsFull = errors.New("store: the connector has as many event destinations as it may")

// EventDestination is one customer URL a connector's raw provider events are forwarded to,
// with its signing secret sealed by the caller.
type EventDestination struct {
	bun.BaseModel `bun:"table:connector_event_destinations,alias:ced"`

	ID          string `bun:"id,pk"`
	CustomerID  string `bun:"customer_id,notnull"`
	ConnectorID string `bun:"connector_id,notnull"`
	URL         string `bun:"url,notnull"`
	Forward     string `bun:"forward,notnull"`
	// SecretSealed is the signing secret sealed under KEKVersion.
	SecretSealed []byte `bun:"secret_sealed,notnull"`
	KEKVersion   int    `bun:"kek_version,notnull"`
	// PreviousSecretSealed is the secret the last rotation replaced, which still signs until
	// PreviousUntil. Nil when none does.
	PreviousSecretSealed []byte     `bun:"previous_secret_sealed"`
	PreviousKEKVersion   int        `bun:"previous_kek_version,notnull"`
	PreviousUntil        *time.Time `bun:"previous_until"`
	CreatedAt            time.Time  `bun:"created_at,notnull"`
	UpdatedAt            time.Time  `bun:"updated_at,notnull"`
}

// EventDestinationPosition is where a page of event destinations ended, newest first.
type EventDestinationPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

// EventDestinationLimit is the page size a destination list uses for the limit asked for.
// EventDestinations returns one row more than this, so a caller can tell the page is not the
// last without counting.
func EventDestinationLimit(asked int) int {
	return clampLimit(asked, defaultEventDestinationLimit, maxEventDestinationLimit)
}

// CreateEventDestination stores a new destination, unless the customer already has
// MaxEventDestinations for the connector, which is ErrEventDestinationsFull. The count and the
// insert run under a transaction-scoped advisory lock on the customer and connector, so two
// creates at once take turns.
func (s *Store) CreateEventDestination(ctx context.Context, destination *EventDestination) error {
	if destination.ID == "" || destination.CustomerID == "" || destination.ConnectorID == "" || destination.URL == "" ||
		len(destination.SecretSealed) == 0 || destination.KEKVersion < 1 {
		return stack.Wrap(errors.New("store: an event destination needs an id, a customer, a connector, a url and a sealed secret under a key version of 1 or more"))
	}
	if destination.Forward != ForwardUnhandled && destination.Forward != ForwardAll {
		return stack.Wrap(fmt.Errorf("store: an event destination forwards %s or %s, not %q", ForwardUnhandled, ForwardAll, destination.Forward))
	}
	// Truncated to what Postgres keeps, so the cursor a page hands out finds this row again.
	now := time.Now().UTC().Truncate(time.Microsecond)
	destination.CreatedAt, destination.UpdatedAt = now, now
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			destination.CustomerID+"/"+destination.ConnectorID, eventDestinationLockSeed); err != nil {
			return err
		}
		held, err := tx.NewSelect().Model((*EventDestination)(nil)).
			Where("customer_id = ?", destination.CustomerID).
			Where("connector_id = ?", destination.ConnectorID).
			Count(ctx)
		if err != nil {
			return err
		}
		if held >= MaxEventDestinations {
			return fmt.Errorf("%w: %d for %s", ErrEventDestinationsFull, MaxEventDestinations, destination.ConnectorID)
		}
		_, err = tx.NewInsert().Model(destination).Exec(ctx)
		return err
	}))
}

// EventDestinations returns a page of the customer's destinations of the connector, newest
// first, after the position after when it is set, with one row more than the page holds.
func (s *Store) EventDestinations(ctx context.Context, customerID, connectorID string, limit int, after *EventDestinationPosition) ([]EventDestination, error) {
	if customerID == "" || connectorID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	destinations := []EventDestination{}
	query := s.db.NewSelect().Model(&destinations).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID)
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(EventDestinationLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list event destinations: %w", err))
	}
	return destinations, nil
}

// DeleteEventDestination removes one of the customer's destinations of the connector, and with
// it every forward to it not yet sent.
func (s *Store) DeleteEventDestination(ctx context.Context, customerID, connectorID, id string) error {
	result, err := s.db.NewDelete().Model((*EventDestination)(nil)).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete event destination: %w", err))
	}
	if affected, _ := result.RowsAffected(); affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoEventDestination, id))
	}
	return nil
}

// RotateEventDestinationSecret makes sealed the destination's signing secret and keeps the one
// it replaces signing beside it until previousUntil. A rotation during another one drops the
// oldest secret: two sign at most. It returns the destination as it is now.
func (s *Store) RotateEventDestinationSecret(ctx context.Context, customerID, connectorID, id string, sealed []byte, version int, previousUntil time.Time) (EventDestination, error) {
	if len(sealed) == 0 || version < 1 {
		return EventDestination{}, stack.Wrap(errors.New("store: a rotation needs a sealed secret under a key version of 1 or more"))
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	var destination EventDestination
	err := s.db.NewRaw(`
UPDATE connector_event_destinations
SET previous_secret_sealed = secret_sealed,
    previous_kek_version = kek_version,
    previous_until = ?,
    secret_sealed = ?,
    kek_version = ?,
    updated_at = ?
WHERE id = ? AND customer_id = ? AND connector_id = ?
RETURNING *`, previousUntil.UTC(), sealed, version, now, id, customerID, connectorID).Scan(ctx, &destination)
	if errors.Is(err, sql.ErrNoRows) {
		return EventDestination{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoEventDestination, id))
	}
	if err != nil {
		return EventDestination{}, stack.Wrap(fmt.Errorf("store: rotate event destination secret: %w", err))
	}
	return destination, nil
}

// EventDelivery is one forward of a provider delivery to one destination.
type EventDelivery struct {
	bun.BaseModel `bun:"table:connector_event_deliveries,alias:cedl"`

	DestinationID string `bun:"destination_id,pk"`
	// ID is the Standard Webhooks webhook-id, the same on every attempt.
	ID string `bun:"id,pk"`
	// Headers are the provider's own headers the receiver verifies the body with, as sent.
	Headers map[string]string `bun:"headers,type:jsonb,notnull"`
	// Body is the provider's raw body, unchanged.
	Body          []byte    `bun:"body,notnull"`
	Attempts      int       `bun:"attempts,notnull"`
	NextAttemptAt time.Time `bun:"next_attempt_at,notnull"`
	CreatedAt     time.Time `bun:"created_at,notnull"`
	// ProviderHeadersUntil is when the provider's signature headers in Headers stop verifying
	// at a receiver that checks them. Nil when they do not age.
	ProviderHeadersUntil *time.Time `bun:"provider_headers_until"`
}

// ClaimedEventDelivery is a delivery a worker took, with what sending it needs of its
// destination.
type ClaimedEventDelivery struct {
	EventDelivery
	CustomerID           string     `bun:"customer_id"`
	ConnectorID          string     `bun:"connector_id"`
	URL                  string     `bun:"url"`
	SecretSealed         []byte     `bun:"secret_sealed"`
	KEKVersion           int        `bun:"kek_version"`
	PreviousSecretSealed []byte     `bun:"previous_secret_sealed"`
	PreviousKEKVersion   int        `bun:"previous_kek_version"`
	PreviousUntil        *time.Time `bun:"previous_until"`
}

// QueueEventDeliveries queues one provider delivery for each of the customer's destinations of
// the connector whose forward is one of forwards (ForwardAll, ForwardUnhandled). A delivery
// already queued for a destination under the same id is not queued again. It returns how many
// it queued.
func (s *Store) QueueEventDeliveries(ctx context.Context, customerID, connectorID string, forwards []string, delivery EventDelivery) (int, error) {
	if customerID == "" || connectorID == "" || delivery.ID == "" || len(forwards) == 0 {
		return 0, stack.Wrap(errors.New("store: a customer, a connector id, a delivery id and the forwards that take it are required"))
	}
	headers, err := json.Marshal(delivery.Headers)
	if err != nil {
		return 0, stack.Wrap(err)
	}
	var headersUntil *time.Time
	if delivery.ProviderHeadersUntil != nil {
		until := delivery.ProviderHeadersUntil.UTC()
		headersUntil = &until
	}
	result, err := s.db.NewRaw(`
INSERT INTO connector_event_deliveries (destination_id, id, headers, body, attempts, next_attempt_at, provider_headers_until)
SELECT ced.id, ?, ?::jsonb, ?, 0, ?, ?
FROM connector_event_destinations AS ced
WHERE ced.customer_id = ? AND ced.connector_id = ? AND ced.forward IN (?)
ON CONFLICT (destination_id, id) DO NOTHING`,
		delivery.ID, string(headers), delivery.Body, delivery.NextAttemptAt.UTC(), headersUntil, customerID, connectorID, bun.In(forwards)).Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: queue event deliveries: %w", err))
	}
	queued, _ := result.RowsAffected()
	return int(queued), nil
}

// ClaimEventDeliveries takes at most limit deliveries due at now, oldest due first, and moves
// each one's next attempt to leaseUntil, so no other worker takes it while this one sends it.
// Rows another worker is claiming at the same moment are skipped, not waited for.
//
// It takes at most perDestination deliveries of one destination, less the ones of it the
// caller is still sending (sending, by destination id). So one destination that never answers
// holds perDestination of the caller's sends, and other destinations' deliveries are taken
// past its own.
func (s *Store) ClaimEventDeliveries(ctx context.Context, now time.Time, limit, perDestination int, sending map[string]int, leaseUntil time.Time) ([]ClaimedEventDelivery, error) {
	return s.claimEventDeliveries(ctx, claimEventDeliveriesQuery, now, limit, perDestination, sending, leaseUntil)
}

// claimEventDeliveriesQuery is ClaimEventDeliveries' statement. Postgres refuses FOR UPDATE
// beside a window function in one SELECT, so ranked numbers each destination's due rows first
// and due locks the ones within the cap.
//
// due checks next_attempt_at on the row it locks, not only in ranked, which reads the rows as
// they were when the statement started. When another router's claim commits in between,
// Postgres locks the row's newer version and checks only due's own WHERE on it again («the
// second updater ... re-evaluates its WHERE condition»: Read Committed Isolation Level,
// https://www.postgresql.org/docs/16/transaction-iso.html, opened October 7, 2026). Without
// the check in due, a row another router just leased is leased again (AI-924 review of #778).
const claimEventDeliveriesQuery = `
WITH sending AS (
    SELECT * FROM unnest(?::text[], ?::int[]) AS s (destination_id, count)
), ranked AS (
    SELECT cedl.destination_id, cedl.id,
        row_number() OVER (PARTITION BY cedl.destination_id ORDER BY cedl.next_attempt_at, cedl.id)
            + coalesce(sending.count, 0) AS slot
    FROM connector_event_deliveries AS cedl
    LEFT JOIN sending ON sending.destination_id = cedl.destination_id
    WHERE cedl.next_attempt_at <= ?
), due AS (
    SELECT cedl.destination_id, cedl.id FROM connector_event_deliveries AS cedl
    JOIN ranked ON ranked.destination_id = cedl.destination_id AND ranked.id = cedl.id
    WHERE ranked.slot <= ? AND cedl.next_attempt_at <= ?
    ORDER BY cedl.next_attempt_at
    LIMIT ?
    FOR UPDATE OF cedl SKIP LOCKED
)
UPDATE connector_event_deliveries AS cedl
SET next_attempt_at = ?
FROM due, connector_event_destinations AS ced
WHERE cedl.destination_id = due.destination_id AND cedl.id = due.id AND ced.id = cedl.destination_id
RETURNING cedl.destination_id, cedl.id, cedl.headers, cedl.body, cedl.attempts, cedl.next_attempt_at, cedl.created_at,
    cedl.provider_headers_until, ced.customer_id, ced.connector_id, ced.url, ced.secret_sealed, ced.kek_version,
    ced.previous_secret_sealed, ced.previous_kek_version, ced.previous_until`

// claimEventDeliveries runs query, ClaimEventDeliveries' statement or a test's variant of it
// with a pause in it, with ClaimEventDeliveries' arguments.
func (s *Store) claimEventDeliveries(ctx context.Context, query string, now time.Time, limit, perDestination int, sending map[string]int, leaseUntil time.Time) ([]ClaimedEventDelivery, error) {
	claimed := []ClaimedEventDelivery{}
	if limit < 1 || perDestination < 1 {
		return claimed, nil
	}
	busy, counts := make([]string, 0, len(sending)), make([]int, 0, len(sending))
	for destination, count := range sending {
		busy, counts = append(busy, destination), append(counts, count)
	}
	err := s.db.NewRaw(query, pgdialect.Array(busy), pgdialect.Array(counts), now.UTC(), perDestination, now.UTC(), limit,
		leaseUntil.UTC()).Scan(ctx, &claimed)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim event deliveries: %w", err))
	}
	return claimed, nil
}

// NextEventDeliveryAt is when the delivery due first is due, any router's, leased ones at the
// end of their lease. found is false when no delivery is queued at all.
func (s *Store) NextEventDeliveryAt(ctx context.Context) (next time.Time, found bool, err error) {
	var at sql.NullTime
	if err := s.db.NewRaw("SELECT min(next_attempt_at) FROM connector_event_deliveries").Scan(ctx, &at); err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next event delivery: %w", err))
	}
	return at.Time, at.Valid, nil
}

// FinishEventDelivery removes a delivery the destination took, refused, or that ran out of
// retries. One the destination was deleted under is already gone, which is not an error.
func (s *Store) FinishEventDelivery(ctx context.Context, destinationID, id string) error {
	_, err := s.db.NewDelete().Model((*EventDelivery)(nil)).
		Where("destination_id = ?", destinationID).
		Where("id = ?", id).
		Exec(ctx)
	return stack.Wrap(err)
}

// RetryEventDelivery records a failed attempt of a delivery, the attempts it has now, and when
// the next one is due.
func (s *Store) RetryEventDelivery(ctx context.Context, destinationID, id string, attempts int, at time.Time) error {
	_, err := s.db.NewUpdate().Model((*EventDelivery)(nil)).
		Set("attempts = ?", attempts).
		Set("next_attempt_at = ?", at.UTC()).
		Where("destination_id = ?", destinationID).
		Where("id = ?", id).
		Exec(ctx)
	return stack.Wrap(err)
}
