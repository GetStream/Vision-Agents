package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// oauthClientRegistrations are the registration methods a client record can have: the two
// kinds of client registered in advance (core.ClientRegistrationMethod). dcr and cimd clients
// are registered during a consent and live in its connection, never here. T40 adds managed.
var oauthClientRegistrations = []core.ClientRegistrationMethod{core.ClientCustomer, core.ClientOperator}

// ErrNoConnectorOAuthClient says the customer has no client record of that registration for
// the connector.
var ErrNoConnectorOAuthClient = errors.New("store: no OAuth client for this connector")

// ErrOAuthClientRegistration says a write would replace a client record of another
// registration: an app's own client cannot overwrite the one the operator registered for it,
// nor the other way round.
var ErrOAuthClientRegistration = errors.New("store: the connector's OAuth client is of another registration")

// ConnectorOAuthClient is the OAuth client one customer uses at one connector, registered in
// advance. Its secret is sealed by the caller before it reaches the store.
type ConnectorOAuthClient struct {
	bun.BaseModel `bun:"table:connector_oauth_clients,alias:coc"`

	CustomerID   string                        `bun:"customer_id,pk"`
	ConnectorID  string                        `bun:"connector_id,pk"`
	Registration core.ClientRegistrationMethod `bun:"registration,notnull"`
	ClientID     string                        `bun:"client_id,notnull"`
	// AuthMethod is empty when the scheme chooses it.
	AuthMethod core.ClientAuthMethod `bun:"auth_method,notnull"`
	// SecretSealed is the client secret sealed under KEKVersion. Empty, with version 0, for a
	// client without one.
	SecretSealed []byte    `bun:"secret_sealed,notnull"`
	KEKVersion   int       `bun:"kek_version,notnull"`
	CreatedAt    time.Time `bun:"created_at,notnull"`
	UpdatedAt    time.Time `bun:"updated_at,notnull"`
}

// PutConnectorOAuthClient stores the customer's client for the connector, replacing the one
// it had of the same registration in the same statement, so a rotated secret is one row
// changed. created is false when a client was replaced. A client of another registration is
// left alone and ErrOAuthClientRegistration returned.
func (s *Store) PutConnectorOAuthClient(ctx context.Context, client *ConnectorOAuthClient) (created bool, err error) {
	if err := checkOAuthClient(client); err != nil {
		return false, err
	}
	// Truncated to what Postgres keeps, so the created_at an insert returns equals it.
	now := time.Now().UTC().Truncate(time.Microsecond)
	var createdAt time.Time
	err = s.db.NewRaw(`
INSERT INTO connector_oauth_clients AS coc
    (customer_id, connector_id, registration, client_id, auth_method, secret_sealed, kek_version, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (customer_id, connector_id) DO UPDATE
SET client_id = EXCLUDED.client_id,
    auth_method = EXCLUDED.auth_method,
    secret_sealed = EXCLUDED.secret_sealed,
    kek_version = EXCLUDED.kek_version,
    updated_at = EXCLUDED.updated_at
WHERE coc.registration = EXCLUDED.registration
RETURNING coc.created_at`,
		client.CustomerID, client.ConnectorID, client.Registration, client.ClientID, client.AuthMethod,
		client.SecretSealed, client.KEKVersion, now, now).Scan(ctx, &createdAt)
	// The conflict's WHERE kept the row, so nothing was written or returned.
	if errors.Is(err, sql.ErrNoRows) {
		return false, stack.Wrap(fmt.Errorf("%w: %s", ErrOAuthClientRegistration, client.ConnectorID))
	}
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: put connector oauth client: %w", err))
	}
	client.CreatedAt, client.UpdatedAt = createdAt, now
	return createdAt.Equal(now), nil
}

// ConnectorOAuthClient returns the customer's client for the connector, of whichever
// registration it is.
func (s *Store) ConnectorOAuthClient(ctx context.Context, customerID, connectorID string) (ConnectorOAuthClient, error) {
	if customerID == "" || connectorID == "" {
		return ConnectorOAuthClient{}, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	var client ConnectorOAuthClient
	err := s.db.NewSelect().Model(&client).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorOAuthClient{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorOAuthClient, connectorID))
	}
	if err != nil {
		return ConnectorOAuthClient{}, stack.Wrap(fmt.Errorf("store: connector oauth client: %w", err))
	}
	return client, nil
}

// DeleteConnectorOAuthClient removes the customer's client of that registration for the
// connector. A client of another registration is not the caller's to remove, so it is
// ErrNoConnectorOAuthClient as a missing one is.
func (s *Store) DeleteConnectorOAuthClient(ctx context.Context, customerID, connectorID string, registration core.ClientRegistrationMethod) error {
	if customerID == "" || connectorID == "" {
		return stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	result, err := s.db.NewDelete().Model((*ConnectorOAuthClient)(nil)).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID).
		Where("registration = ?", registration).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete connector oauth client: %w", err))
	}
	deleted, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete connector oauth client: %w", err))
	}
	if deleted == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorOAuthClient, connectorID))
	}
	return nil
}

// checkOAuthClient is what the migration leaves to the store: a known registration, and a
// sealed secret with the key version it was sealed under, or neither.
func checkOAuthClient(client *ConnectorOAuthClient) error {
	if client.CustomerID == "" || client.ConnectorID == "" || client.ClientID == "" {
		return stack.Wrap(errors.New("store: an OAuth client needs a customer, a connector id and a client id"))
	}
	if !slices.Contains(oauthClientRegistrations, client.Registration) {
		return stack.Wrap(fmt.Errorf("store: OAuth client registration %q is not one of %v", client.Registration, oauthClientRegistrations))
	}
	if client.SecretSealed == nil {
		client.SecretSealed = []byte{}
	}
	if (len(client.SecretSealed) == 0) != (client.KEKVersion == 0) || client.KEKVersion < 0 {
		return stack.Wrap(errors.New("store: a sealed secret has a key version of 1 or more, and no secret has version 0"))
	}
	return nil
}
