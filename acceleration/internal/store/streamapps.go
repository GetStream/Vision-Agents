package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/driver/pgdriver"
)

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

// StreamApp is the Stream app a customer registered as its own, from before customers' apps
// were acted in with the gateway's tokens. Nothing reads or writes it any more.
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

// StreamAppToken is the newest server token the gateway minted for a Stream app, sealed
// under the deployment's keyring and bound to this row.
type StreamAppToken struct {
	bun.BaseModel `bun:"table:stream_app_tokens,alias:sat"`

	StreamAppPK int64  `bun:"stream_app_pk,pk"`
	CustomerID  string `bun:"customer_id,notnull"`
	APIKey      string `bun:"api_key,notnull"`
	Sealed      []byte `bun:"token_sealed,notnull"`
	// KEKVersion names which key encryption key sealed this row.
	KEKVersion int       `bun:"kek_version,notnull"`
	ExpiresAt  time.Time `bun:"expires_at,notnull"`
	UpdatedAt  time.Time `bun:"updated_at,notnull"`
}

// ErrNoStreamAppToken is an app the gateway has minted no token for here.
var ErrNoStreamAppToken = errors.New("store: no token is held for this stream app")

// KeepStreamAppToken holds a token for its app, unless the one already held outlasts it,
// and reports whether it was kept.
func (s *Store) KeepStreamAppToken(ctx context.Context, token StreamAppToken) (bool, error) {
	token.ExpiresAt = token.ExpiresAt.UTC()
	token.UpdatedAt = time.Now().UTC()
	result, err := s.db.NewInsert().Model(&token).
		On("CONFLICT (stream_app_pk) DO UPDATE").
		Set("customer_id = EXCLUDED.customer_id").
		Set("api_key = EXCLUDED.api_key").
		Set("token_sealed = EXCLUDED.token_sealed").
		Set("kek_version = EXCLUDED.kek_version").
		Set("expires_at = EXCLUDED.expires_at").
		Set("updated_at = EXCLUDED.updated_at").
		Where("sat.expires_at < EXCLUDED.expires_at").
		Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("store: keep stream app token: %w", err)
	}
	written, _ := result.RowsAffected()
	return written > 0, nil
}

// StreamAppToken is the token held for an app.
func (s *Store) StreamAppToken(ctx context.Context, app int64) (StreamAppToken, error) {
	var token StreamAppToken
	err := s.db.NewSelect().Model(&token).Where("stream_app_pk = ?", app).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return StreamAppToken{}, ErrNoStreamAppToken
	}
	if err != nil {
		return StreamAppToken{}, fmt.Errorf("store: read stream app token: %w", err)
	}
	return token, nil
}
