package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/driver/pgdriver"
)

var (
	// ErrNoStreamApp is a customer that never registered a Stream app of its own.
	ErrNoStreamApp = errors.New("store: no Stream app is registered for this customer")
	// ErrStreamAppChanged is a write made against a revision that is no longer current.
	ErrStreamAppChanged = errors.New("store: the Stream app was changed since it was read")
	// ErrStreamAppTaken is a Stream app another customer already registered.
	ErrStreamAppTaken = errors.New("store: that Stream app is registered to another customer")
	// ErrStreamAppKeyTaken is an api key another customer's app already holds.
	ErrStreamAppKeyTaken = errors.New("store: that api key belongs to another customer's Stream app")
)

// MaxStreamAppKeys is how many keys one app may hold at once: enough to rotate through,
// few enough that checking every one with Stream stays quick.
const MaxStreamAppKeys = 8

// StreamAppState is whether the router acts in a registered app.
type StreamAppState string

const (
	// StreamAppConnected is an app the router acts in with its keys.
	StreamAppConnected StreamAppState = "connected"
	// StreamAppDisconnected is an app its customer took back: no keys are held, and the
	// customer's work is written nowhere rather than into the deployment's app.
	StreamAppDisconnected StreamAppState = "disconnected"
	// StreamAppBlocked is an app Stream suspended, or one that stopped checking the
	// tokens the router mints, so nothing is written there until it is registered again.
	StreamAppBlocked StreamAppState = "blocked"
)

// StreamAppKeyStatus is whether Stream still accepts a key.
type StreamAppKeyStatus string

const (
	StreamAppKeyActive   StreamAppKeyStatus = "active"
	StreamAppKeyRejected StreamAppKeyStatus = "rejected"
)

// StreamApp is the Stream app a customer registered as its own.
type StreamApp struct {
	bun.BaseModel `bun:"table:stream_apps,alias:sa"`

	CustomerID     string         `bun:"customer_id,pk"`
	StreamAppPK    int64          `bun:"stream_app_pk,notnull"`
	OrganizationID string         `bun:"organization_id,notnull"`
	State          StreamAppState `bun:"state,notnull"`
	StateReason    string         `bun:"state_reason,notnull"`
	// PrimaryKey is the key tokens are minted with, empty once disconnected.
	PrimaryKey  string          `bun:"primary_key,nullzero"`
	Revision    int64           `bun:"revision,notnull"`
	AllowGuests bool            `bun:"allow_guests,notnull"`
	Checks      json.RawMessage `bun:"checks,type:jsonb,nullzero"`
	CheckedAt   *time.Time      `bun:"checked_at"`
	CreatedAt   time.Time       `bun:"created_at,notnull"`
	UpdatedAt   time.Time       `bun:"updated_at,notnull"`
	UpdatedBy   string          `bun:"updated_by,notnull"`

	// Keys are the app's keys, oldest first.
	Keys []StreamAppKey `bun:"-"`
}

// StreamAppKey is one key of a registered app, its secret sealed under the deployment's
// keyring and bound to this row.
type StreamAppKey struct {
	bun.BaseModel `bun:"table:stream_app_keys,alias:sak"`

	APIKey     string `bun:"api_key,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Sealed     []byte `bun:"secret_sealed,notnull"`
	// KEKVersion names which key encryption key sealed this row.
	KEKVersion int    `bun:"kek_version,notnull"`
	Last4      string `bun:"secret_last4,notnull"`
	// KeyCreatedAt is when Stream says the key was made, which orders keys by age.
	KeyCreatedAt   *time.Time         `bun:"key_created_at"`
	Status         StreamAppKeyStatus `bun:"status,notnull"`
	RejectedAt     *time.Time         `bun:"rejected_at"`
	RejectedReason string             `bun:"rejected_reason,notnull"`
	VerifiedAt     *time.Time         `bun:"verified_at"`
	LastWebhookAt  *time.Time         `bun:"last_webhook_at"`
	CreatedAt      time.Time          `bun:"created_at,notnull"`
}

// StreamAppRegistration is a customer's app and the whole set of keys it now holds.
type StreamAppRegistration struct {
	CustomerID     string
	OrganizationID string
	StreamAppPK    int64
	// Keys are sealed already: the store never sees a secret.
	Keys        []StreamAppKey
	PrimaryKey  string
	AllowGuests bool
	// ExpectedRevision is the revision the writer read, zero for an app never registered.
	// Nil writes whatever is there, which only the operator's command line does.
	ExpectedRevision *int64
	UpdatedBy        string
	VerifiedAt       time.Time
}

// PutStreamApp registers a customer's app, or replaces its keys, in one transaction. Keys
// left out are deleted, the app is connected again, and the revision moves on.
func (s *Store) PutStreamApp(ctx context.Context, registration StreamAppRegistration) (StreamApp, error) {
	if err := registration.valid(); err != nil {
		return StreamApp{}, err
	}
	var stored StreamApp
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := lockStreamApp(ctx, tx, registration.CustomerID, registration.ExpectedRevision); err != nil && !errors.Is(err, ErrNoStreamApp) {
			return err
		}
		now := time.Now().UTC()
		app := StreamApp{
			CustomerID: registration.CustomerID, StreamAppPK: registration.StreamAppPK,
			OrganizationID: registration.OrganizationID, State: StreamAppConnected,
			PrimaryKey: registration.PrimaryKey, Revision: 1, AllowGuests: registration.AllowGuests,
			CreatedAt: now, UpdatedAt: now, UpdatedBy: registration.UpdatedBy,
		}
		_, err := tx.NewInsert().Model(&app).
			On("CONFLICT (customer_id) DO UPDATE").
			Set("stream_app_pk = EXCLUDED.stream_app_pk").
			Set("organization_id = EXCLUDED.organization_id").
			Set("state = EXCLUDED.state").
			Set("state_reason = ''").
			Set("primary_key = EXCLUDED.primary_key").
			Set("revision = sa.revision + 1").
			Set("allow_guests = EXCLUDED.allow_guests").
			Set("updated_at = EXCLUDED.updated_at").
			Set("updated_by = EXCLUDED.updated_by").
			Exec(ctx)
		if constraint(err) == "stream_apps_stream_app_pk_key" {
			return ErrStreamAppTaken
		}
		if err != nil {
			return fmt.Errorf("store: register stream app: %w", err)
		}

		kept := make([]string, 0, len(registration.Keys))
		for _, key := range registration.Keys {
			kept = append(kept, key.APIKey)
		}
		if _, err := tx.NewDelete().Model((*StreamAppKey)(nil)).
			Where("customer_id = ?", registration.CustomerID).
			Where("api_key NOT IN (?)", bun.In(kept)).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: drop stream app keys: %w", err)
		}
		verified := registration.VerifiedAt.UTC()
		for _, key := range registration.Keys {
			key.CustomerID, key.Status, key.CreatedAt, key.VerifiedAt = registration.CustomerID, StreamAppKeyActive, now, &verified
			key.RejectedAt, key.RejectedReason = nil, ""
			result, err := tx.NewInsert().Model(&key).
				On("CONFLICT (api_key) DO UPDATE").
				Set("secret_sealed = EXCLUDED.secret_sealed").
				Set("kek_version = EXCLUDED.kek_version").
				Set("secret_last4 = EXCLUDED.secret_last4").
				Set("key_created_at = EXCLUDED.key_created_at").
				Set("status = EXCLUDED.status").
				Set("rejected_at = NULL").
				Set("rejected_reason = ''").
				Set("verified_at = EXCLUDED.verified_at").
				// A key another customer's app holds is not this one's to overwrite.
				Where("sak.customer_id = EXCLUDED.customer_id").
				Exec(ctx)
			if err != nil {
				return fmt.Errorf("store: write stream app key: %w", err)
			}
			if written, _ := result.RowsAffected(); written == 0 {
				return ErrStreamAppKeyTaken
			}
		}
		stored, err = readStreamApp(ctx, tx, registration.CustomerID)
		return err
	})
	return stored, err
}

func (r StreamAppRegistration) valid() error {
	switch {
	case r.CustomerID == "":
		return errors.New("store: a Stream app needs a customer")
	case r.StreamAppPK <= 0:
		return errors.New("store: a Stream app needs its id")
	case len(r.Keys) == 0:
		return errors.New("store: a Stream app needs a key; disconnect it to hold none")
	case len(r.Keys) > MaxStreamAppKeys:
		return fmt.Errorf("store: a Stream app holds at most %d keys", MaxStreamAppKeys)
	}
	seen := map[string]bool{}
	for _, key := range r.Keys {
		switch {
		case key.APIKey == "":
			return errors.New("store: a Stream app key needs its api key")
		case seen[key.APIKey]:
			return fmt.Errorf("store: api key %s is named twice", key.APIKey)
		case len(key.Sealed) == 0 || key.KEKVersion < 1:
			return fmt.Errorf("store: api key %s needs a sealed secret", key.APIKey)
		}
		seen[key.APIKey] = true
	}
	if !seen[r.PrimaryKey] {
		return fmt.Errorf("store: the primary key %q is not one of the app's keys", r.PrimaryKey)
	}
	return nil
}

// lockStreamApp reads a customer's app for update, and refuses a write made against a
// revision that is no longer current. Writes for one customer are taken one at a time,
// first registrations included, which have no row yet to lock.
func lockStreamApp(ctx context.Context, tx bun.Tx, customer string, expected *int64) (StreamApp, error) {
	if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtext('stream_apps:' || ?))", customer); err != nil {
		return StreamApp{}, fmt.Errorf("store: lock stream app: %w", err)
	}
	var app StreamApp
	err := tx.NewSelect().Model(&app).Where("customer_id = ?", customer).For("UPDATE").Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		if expected != nil && *expected != 0 {
			return StreamApp{}, ErrStreamAppChanged
		}
		return StreamApp{}, ErrNoStreamApp
	}
	if err != nil {
		return StreamApp{}, fmt.Errorf("store: read stream app: %w", err)
	}
	if expected != nil && *expected != app.Revision {
		return StreamApp{}, ErrStreamAppChanged
	}
	return app, nil
}

// StreamApp is the app a customer registered, with its keys.
func (s *Store) StreamApp(ctx context.Context, customer string) (StreamApp, error) {
	return readStreamApp(ctx, s.db, customer)
}

func readStreamApp(ctx context.Context, db bun.IDB, customer string) (StreamApp, error) {
	var app StreamApp
	err := db.NewSelect().Model(&app).Where("customer_id = ?", customer).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return StreamApp{}, ErrNoStreamApp
	}
	if err != nil {
		return StreamApp{}, fmt.Errorf("store: read stream app: %w", err)
	}
	if err := db.NewSelect().Model(&app.Keys).Where("customer_id = ?", customer).
		OrderExpr("key_created_at ASC NULLS LAST, created_at ASC, api_key ASC").
		Scan(ctx); err != nil {
		return StreamApp{}, fmt.Errorf("store: read stream app keys: %w", err)
	}
	return app, nil
}

// StreamAppByAPIKey is the app holding an api key, with all its keys.
func (s *Store) StreamAppByAPIKey(ctx context.Context, apiKey string) (StreamApp, error) {
	var key StreamAppKey
	err := s.db.NewSelect().Model(&key).Column("customer_id").Where("api_key = ?", apiKey).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return StreamApp{}, ErrNoStreamApp
	}
	if err != nil {
		return StreamApp{}, fmt.Errorf("store: find stream app key: %w", err)
	}
	return s.StreamApp(ctx, key.CustomerID)
}

// Key is one of the app's keys by its api key.
func (a StreamApp) Key(apiKey string) (StreamAppKey, bool) {
	index := slices.IndexFunc(a.Keys, func(key StreamAppKey) bool { return key.APIKey == apiKey })
	if index < 0 {
		return StreamAppKey{}, false
	}
	return a.Keys[index], true
}

// RejectStreamAppKey records that Stream refuses a key, which is then passed over for
// the app's other active keys.
func (s *Store) RejectStreamAppKey(ctx context.Context, customer, apiKey, reason string, at time.Time) error {
	at = at.UTC()
	_, err := s.db.NewUpdate().Model((*StreamAppKey)(nil)).
		Set("status = ?", StreamAppKeyRejected).
		Set("rejected_at = ?", at).
		Set("rejected_reason = ?", reason).
		Where("customer_id = ?", customer).
		Where("api_key = ?", apiKey).
		Where("status = ?", StreamAppKeyActive).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: reject stream app key: %w", err)
	}
	return nil
}

// RewrapStreamAppKey replaces a key's sealed secret with the same secret sealed under a
// newer key encryption key. It changes nothing when the secret was replaced meanwhile, and
// reports whether it applied.
func (s *Store) RewrapStreamAppKey(ctx context.Context, customer, apiKey string, was, sealed []byte, version int) (bool, error) {
	result, err := s.db.NewUpdate().Model((*StreamAppKey)(nil)).
		Set("secret_sealed = ?", sealed).
		Set("kek_version = ?", version).
		Where("customer_id = ?", customer).
		Where("api_key = ?", apiKey).
		Where("secret_sealed = ?", was).
		Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("store: rewrap stream app key: %w", err)
	}
	written, _ := result.RowsAffected()
	return written > 0, nil
}

// RecordStreamAppChecks keeps what the last check of an app found.
func (s *Store) RecordStreamAppChecks(ctx context.Context, customer string, checks json.RawMessage, at time.Time) error {
	_, err := s.db.NewUpdate().Model((*StreamApp)(nil)).
		Set("checks = ?", string(checks)).
		Set("checked_at = ?", at.UTC()).
		Where("customer_id = ?", customer).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: record stream app checks: %w", err)
	}
	return nil
}

// BlockStreamApp stops the router acting in a connected app, and reports whether it was
// connected until now.
func (s *Store) BlockStreamApp(ctx context.Context, customer, reason string) (bool, error) {
	result, err := s.db.NewUpdate().Model((*StreamApp)(nil)).
		Set("state = ?", StreamAppBlocked).
		Set("state_reason = ?", reason).
		Set("updated_at = ?", time.Now().UTC()).
		Where("customer_id = ?", customer).
		Where("state = ?", StreamAppConnected).
		Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("store: block stream app: %w", err)
	}
	written, _ := result.RowsAffected()
	return written > 0, nil
}

// webhookTouchEvery is the least time between two records of a hook arriving signed by one
// key, so a busy app does not write on every event.
const webhookTouchEvery = time.Minute

// TouchStreamAppWebhook records that a hook arrived signed by a key, at most once a minute.
func (s *Store) TouchStreamAppWebhook(ctx context.Context, apiKey string, at time.Time) error {
	at = at.UTC()
	_, err := s.db.NewUpdate().Model((*StreamAppKey)(nil)).
		Set("last_webhook_at = ?", at).
		Where("api_key = ?", apiKey).
		Where("last_webhook_at IS NULL OR last_webhook_at < ?", at.Add(-webhookTouchEvery)).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: touch stream app webhook: %w", err)
	}
	return nil
}

// DisconnectStreamApp deletes every key of a customer's app and leaves the app behind with
// no secret, so the customer's work is written nowhere rather than into the deployment's
// app.
func (s *Store) DisconnectStreamApp(ctx context.Context, customer string, expected *int64, updatedBy string) (StreamApp, error) {
	var stored StreamApp
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := lockStreamApp(ctx, tx, customer, expected); err != nil {
			return err
		}
		if _, err := tx.NewDelete().Model((*StreamAppKey)(nil)).Where("customer_id = ?", customer).Exec(ctx); err != nil {
			return fmt.Errorf("store: drop stream app keys: %w", err)
		}
		if _, err := tx.NewUpdate().Model((*StreamApp)(nil)).
			Set("state = ?", StreamAppDisconnected).
			Set("state_reason = ''").
			Set("primary_key = NULL").
			Set("revision = revision + 1").
			Set("updated_at = ?", time.Now().UTC()).
			Set("updated_by = ?", updatedBy).
			Where("customer_id = ?", customer).
			Exec(ctx); err != nil {
			return fmt.Errorf("store: disconnect stream app: %w", err)
		}
		var err error
		stored, err = readStreamApp(ctx, tx, customer)
		return err
	})
	return stored, err
}

// ForgetStreamApp deletes a customer's app outright, tombstone and all, so the customer
// acts wherever the fallback says again. Only the operator's command line does this.
func (s *Store) ForgetStreamApp(ctx context.Context, customer string) error {
	if _, err := s.db.NewDelete().Model((*StreamApp)(nil)).Where("customer_id = ?", customer).Exec(ctx); err != nil {
		return fmt.Errorf("store: forget stream app: %w", err)
	}
	return nil
}

// constraint is the name of the unique constraint an error violated, empty for anything
// else.
func constraint(err error) string {
	var failure pgdriver.Error
	if errors.As(err, &failure) && failure.Field('C') == "23505" {
		return failure.Field('n')
	}
	return ""
}

// StreamFallbackUse is how often app mode wrote a customer with no app of its own into
// the deployment's app, and over what time.
type StreamFallbackUse struct {
	bun.BaseModel `bun:"table:stream_fallback_uses,alias:sfu"`

	CustomerID string    `bun:"customer_id,pk"`
	FirstAt    time.Time `bun:"first_at,notnull"`
	LastAt     time.Time `bun:"last_at,notnull"`
	Uses       int64     `bun:"uses,notnull"`
}

// RecordStreamFallbackUses adds uses of the fallback by a customer, the last at the time
// given.
func (s *Store) RecordStreamFallbackUses(ctx context.Context, customer string, uses int64, at time.Time) error {
	at = at.UTC()
	use := StreamFallbackUse{CustomerID: customer, FirstAt: at, LastAt: at, Uses: uses}
	_, err := s.db.NewInsert().Model(&use).
		On("CONFLICT (customer_id) DO UPDATE").
		Set("last_at = GREATEST(sfu.last_at, EXCLUDED.last_at)").
		Set("uses = sfu.uses + EXCLUDED.uses").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: record stream fallback use: %w", err)
	}
	return nil
}

// StreamFallbackUses are the customers that used the fallback since the time given, the
// most recent first.
func (s *Store) StreamFallbackUses(ctx context.Context, since time.Time) ([]StreamFallbackUse, error) {
	var uses []StreamFallbackUse
	err := s.db.NewSelect().Model(&uses).Where("last_at >= ?", since.UTC()).Order("last_at DESC").Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: read stream fallback uses: %w", err)
	}
	return uses, nil
}

// StreamApps are every registered app, with its keys, or with connected only those the
// router acts in with their own keys.
func (s *Store) StreamApps(ctx context.Context, connected bool) ([]StreamApp, error) {
	var apps []StreamApp
	query := s.db.NewSelect().Model(&apps).Order("customer_id")
	if connected {
		query = query.Where("state = ?", StreamAppConnected)
	}
	if err := query.Scan(ctx); err != nil {
		return nil, fmt.Errorf("store: list stream apps: %w", err)
	}
	if len(apps) == 0 {
		return apps, nil
	}
	customers := make([]string, 0, len(apps))
	for _, app := range apps {
		customers = append(customers, app.CustomerID)
	}
	var keys []StreamAppKey
	if err := s.db.NewSelect().Model(&keys).Where("customer_id IN (?)", bun.In(customers)).
		OrderExpr("key_created_at ASC NULLS LAST, created_at ASC, api_key ASC").Scan(ctx); err != nil {
		return nil, fmt.Errorf("store: list stream app keys: %w", err)
	}
	held := make(map[string][]StreamAppKey, len(apps))
	for _, key := range keys {
		held[key.CustomerID] = append(held[key.CustomerID], key)
	}
	for i := range apps {
		apps[i].Keys = held[apps[i].CustomerID]
	}
	return apps, nil
}
