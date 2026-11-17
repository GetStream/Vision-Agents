package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// providerAppLockNamespace is the first key of the two-integer advisory lock one customer's
// provider app of one connector is managed under; the second is credentialLockKey over the
// customer and the connector id. 872 is this lock's issue, AI-872, as 839 is
// credentialLockNamespace's, so no key here is a credential lock's.
const providerAppLockNamespace int32 = 872

// ErrNoConnectorConfigToken says the customer gave no configuration token for the connector.
var ErrNoConnectorConfigToken = errors.New("store: no configuration token for this connector")

// ConnectorConfigToken is the app configuration token a customer's workspace admin gave the
// router for one connector, with its refresh token, sealed by the caller as one blob. It is
// what creates, updates and deletes the customer's managed provider app.
type ConnectorConfigToken struct {
	CustomerID   string    `bun:"customer_id,pk"`
	ConnectorID  string    `bun:"connector_id,pk"`
	TokensSealed []byte    `bun:"tokens_sealed,notnull"`
	KEKVersion   int       `bun:"kek_version,notnull"`
	ExpiresAt    time.Time `bun:"expires_at,notnull"`
	CreatedAt    time.Time `bun:"created_at,notnull"`
	UpdatedAt    time.Time `bun:"updated_at,notnull"`
}

// WithConnectorProviderAppLock runs fn while this router holds the session-level advisory
// lock on the customer's provider app of the connector. Whatever reads, rotates or spends
// the configuration token, or creates, updates or deletes the app, runs inside it, so two
// routers never rotate one refresh token or create two apps for one customer. fn's own writes
// go through the pool and commit as they are made: a rotated token is saved even when what
// follows it fails. fn must not take the same lock again.
func (s *Store) WithConnectorProviderAppLock(ctx context.Context, customerID, connectorID string, fn func() error) error {
	if customerID == "" || connectorID == "" {
		return stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	key := credentialLockKey(customerID, connectorID)
	conn, err := s.acquireCredentialLock(ctx, providerAppLockNamespace, key)
	if err != nil {
		return stack.Wrap(err)
	}
	defer releaseCredentialLock(conn, providerAppLockNamespace, key)
	return fn()
}

// PutConnectorConfigToken stores the customer's configuration token for the connector,
// replacing the one it had. It runs detached from ctx, within credentialDetachedTimeout, as a
// credentials write does: the token is one the provider just rotated, and the refresh token it
// replaced may no longer work.
func (s *Store) PutConnectorConfigToken(ctx context.Context, token *ConnectorConfigToken) error {
	if token.CustomerID == "" || token.ConnectorID == "" || len(token.TokensSealed) == 0 || token.KEKVersion < 1 || token.ExpiresAt.IsZero() {
		return stack.Wrap(errors.New("store: a configuration token needs a customer, a connector id, a sealed blob, a key version of 1 or more and an expiry"))
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), credentialDetachedTimeout)
	defer cancel()
	now := time.Now().UTC().Truncate(time.Microsecond)
	err := s.db.NewRaw(`
INSERT INTO connector_config_tokens AS cct
    (customer_id, connector_id, tokens_sealed, kek_version, expires_at, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (customer_id, connector_id) DO UPDATE
SET tokens_sealed = EXCLUDED.tokens_sealed,
    kek_version = EXCLUDED.kek_version,
    expires_at = EXCLUDED.expires_at,
    updated_at = EXCLUDED.updated_at
RETURNING cct.created_at`,
		token.CustomerID, token.ConnectorID, token.TokensSealed, token.KEKVersion, token.ExpiresAt.UTC(), now, now).
		Scan(ctx, &token.CreatedAt)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: put connector config token: %w", err))
	}
	token.UpdatedAt = now
	return nil
}

// ConnectorConfigToken returns the customer's configuration token for the connector.
func (s *Store) ConnectorConfigToken(ctx context.Context, customerID, connectorID string) (ConnectorConfigToken, error) {
	if customerID == "" || connectorID == "" {
		return ConnectorConfigToken{}, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	var token ConnectorConfigToken
	err := s.db.NewRaw(`
SELECT customer_id, connector_id, tokens_sealed, kek_version, expires_at, created_at, updated_at
FROM connector_config_tokens WHERE customer_id = ? AND connector_id = ?`, customerID, connectorID).
		Scan(ctx, &token.CustomerID, &token.ConnectorID, &token.TokensSealed, &token.KEKVersion, &token.ExpiresAt, &token.CreatedAt, &token.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConfigToken{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConfigToken, connectorID))
	}
	if err != nil {
		return ConnectorConfigToken{}, stack.Wrap(fmt.Errorf("store: connector config token: %w", err))
	}
	return token, nil
}

// DeleteConnectorConfigToken removes the customer's configuration token for the connector. A
// token already gone is not an error.
func (s *Store) DeleteConnectorConfigToken(ctx context.Context, customerID, connectorID string) error {
	if customerID == "" || connectorID == "" {
		return stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	_, err := s.db.ExecContext(ctx, "DELETE FROM connector_config_tokens WHERE customer_id = ? AND connector_id = ?", customerID, connectorID)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete connector config token: %w", err))
	}
	return nil
}
