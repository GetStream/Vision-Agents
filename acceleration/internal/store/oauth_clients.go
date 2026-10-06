package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// oauthClientRegistrations are the registration methods a client record can have: the three
// kinds of client registered in advance (core.ClientRegistrationMethod). dcr and cimd clients
// are registered during a consent and live in its connection, never here.
var oauthClientRegistrations = []core.ClientRegistrationMethod{core.ClientCustomer, core.ClientManaged, core.ClientOperator}

// providerAppIDPattern is what a provider app id may be: RFC 3986 section 2.3's unreserved
// characters, so it is one path segment of an events URL (T38) as it is, unescaped. Slack's
// look like A012ABCD0A0 (https://docs.slack.dev/reference/methods/apps.manifest.create).
var providerAppIDPattern = regexp.MustCompile(`^[A-Za-z0-9._~-]+$`)

// ErrNoConnectorOAuthClient says the customer has no client record of that registration for
// the connector.
var ErrNoConnectorOAuthClient = errors.New("store: no OAuth client for this connector")

// ErrOAuthClientRegistration says a write would replace a client record of another
// registration: an app's own client cannot overwrite the one the operator registered for it,
// nor the other way round.
var ErrOAuthClientRegistration = errors.New("store: the connector's OAuth client is of another registration")

// ErrOAuthClientRegistrationNotListed says the connector's latest manifest does not list the
// record's registration in client.registration, so no consent would ever use it.
var ErrOAuthClientRegistrationNotListed = errors.New("store: the connector's client.registration does not list this registration")

// ErrProviderAppTaken says another customer's record for the connector already names the
// provider app: an app belongs to one customer, and its events URL names one tenant.
var ErrProviderAppTaken = errors.New("store: another customer's OAuth client is this provider app")

// ConnectorOAuthClient is the OAuth client one customer uses at one connector, registered in
// advance, and the provider app it belongs to. Its secrets are sealed by the caller before they
// reach the store.
//
// Registration says whose app it is and who registered it: operator is Stream's app,
// registered by this deployment's operator; customer is the customer's app, registered by the
// customer and put through the API; managed is the customer's app, registered by the router
// (T54). So no column repeats either.
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
	SecretSealed []byte `bun:"secret_sealed,notnull"`
	KEKVersion   int    `bun:"kek_version,notnull"`
	// ProviderAppID is the provider's id for the app, such as a Slack app id: unique among the
	// connector's records. Empty when the router needs none; a managed record always has one.
	ProviderAppID string `bun:"provider_app_id,notnull"`
	// StreamAppPK is the Stream app the record was created in, and the one the provider app's
	// work is finished in. Zero, stored as NULL, is the deployment's own app. Set when the
	// record is created; a put that replaces it keeps the pin it has.
	StreamAppPK int64 `bun:"stream_app_pk,nullzero"`
	// SigningSecretSealed is the secret the provider signs the app's inbound requests with,
	// sealed under SigningKEKVersion. Empty, with version 0, for an app that posts none.
	SigningSecretSealed []byte    `bun:"signing_secret_sealed,notnull"`
	SigningKEKVersion   int       `bun:"signing_kek_version,notnull"`
	CreatedAt           time.Time `bun:"created_at,notnull"`
	UpdatedAt           time.Time `bun:"updated_at,notnull"`
}

// PutConnectorOAuthClient stores the customer's client for the connector, replacing the one
// it had of the same registration in the same statement, so a rotated secret is one row
// changed. created is false when a client was replaced. A client of another registration is
// left alone and ErrOAuthClientRegistration returned. A registration the connector's latest
// manifest does not list is ErrOAuthClientRegistrationNotListed, and a provider app another
// customer's record names ErrProviderAppTaken. client.StreamAppPK is the pin of a new record;
// on return it is the pin the record has.
func (s *Store) PutConnectorOAuthClient(ctx context.Context, client *ConnectorOAuthClient) (created bool, err error) {
	if err := checkOAuthClient(client); err != nil {
		return false, err
	}
	definition, err := s.LatestConnectorDefinition(ctx, client.CustomerID, client.ConnectorID)
	if err != nil {
		return false, err
	}
	if !slices.Contains(definition.Manifest.Client.Registration, client.Registration) {
		return false, stack.Wrap(fmt.Errorf("%w: %s lists %v, not %s", ErrOAuthClientRegistrationNotListed,
			client.ConnectorID, definition.Manifest.Client.Registration, client.Registration))
	}
	// Truncated to what Postgres keeps, so the created_at an insert returns equals it.
	now := time.Now().UTC().Truncate(time.Microsecond)
	var createdAt time.Time
	var pin sql.NullInt64
	err = s.db.NewRaw(`
INSERT INTO connector_oauth_clients AS coc
    (customer_id, connector_id, registration, client_id, auth_method, secret_sealed, kek_version,
     provider_app_id, stream_app_pk, signing_secret_sealed, signing_kek_version, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (customer_id, connector_id) DO UPDATE
SET client_id = EXCLUDED.client_id,
    auth_method = EXCLUDED.auth_method,
    secret_sealed = EXCLUDED.secret_sealed,
    kek_version = EXCLUDED.kek_version,
    provider_app_id = EXCLUDED.provider_app_id,
    signing_secret_sealed = EXCLUDED.signing_secret_sealed,
    signing_kek_version = EXCLUDED.signing_kek_version,
    updated_at = EXCLUDED.updated_at
WHERE coc.registration = EXCLUDED.registration
RETURNING coc.created_at, coc.stream_app_pk`,
		client.CustomerID, client.ConnectorID, client.Registration, client.ClientID, client.AuthMethod,
		client.SecretSealed, client.KEKVersion, client.ProviderAppID, nullablePin(client.StreamAppPK),
		client.SigningSecretSealed, client.SigningKEKVersion, now, now).Scan(ctx, &createdAt, &pin)
	// The conflict's WHERE kept the row, so nothing was written or returned.
	if errors.Is(err, sql.ErrNoRows) {
		return false, stack.Wrap(fmt.Errorf("%w: %s", ErrOAuthClientRegistration, client.ConnectorID))
	}
	// ON CONFLICT arbitrates the primary key alone, so another customer's row with the app
	// fails the unique index (20261006195500_connector_oauth_clients_provider_app.sql).
	if constraint(err) == "connector_oauth_clients_provider_app" {
		return false, stack.Wrap(fmt.Errorf("%w: %s %s", ErrProviderAppTaken, client.ConnectorID, client.ProviderAppID))
	}
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: put connector oauth client: %w", err))
	}
	client.CreatedAt, client.UpdatedAt, client.StreamAppPK = createdAt, now, pin.Int64
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

// ConnectorOAuthClientByProviderApp returns the record of the connector's provider app, the
// one customer whose app it is. An inbound request names the app and carries no app
// credential, so the app id, not a customer, is what finds the row, as the OAuth state finds an
// attempt (ConnectorAuthorizationAttemptByState).
func (s *Store) ConnectorOAuthClientByProviderApp(ctx context.Context, connectorID, providerAppID string) (ConnectorOAuthClient, error) {
	if connectorID == "" || providerAppID == "" {
		return ConnectorOAuthClient{}, stack.Wrap(errors.New("store: a connector id and a provider app id are required"))
	}
	var client ConnectorOAuthClient
	err := s.db.NewSelect().Model(&client).
		Where("connector_id = ?", connectorID).
		Where("provider_app_id = ?", providerAppID).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorOAuthClient{}, stack.Wrap(fmt.Errorf("%w: %s provider app %s", ErrNoConnectorOAuthClient, connectorID, providerAppID))
	}
	if err != nil {
		return ConnectorOAuthClient{}, stack.Wrap(fmt.Errorf("store: connector oauth client by provider app: %w", err))
	}
	return client, nil
}

// RewrapConnectorOAuthClientSecret replaces the record's sealed client secret with the same
// secret sealed under a newer key encryption key, as RewrapStreamAppKey does for a Stream key.
// It changes nothing when the secret was replaced meanwhile, and reports whether it applied.
func (s *Store) RewrapConnectorOAuthClientSecret(ctx context.Context, customerID, connectorID string, was, sealed []byte, version int) (bool, error) {
	return s.rewrapConnectorOAuthClient(ctx, "secret_sealed", "kek_version", customerID, connectorID, was, sealed, version)
}

// RewrapConnectorOAuthClientSigningSecret is RewrapConnectorOAuthClientSecret for the provider
// app's signing secret.
func (s *Store) RewrapConnectorOAuthClientSigningSecret(ctx context.Context, customerID, connectorID string, was, sealed []byte, version int) (bool, error) {
	return s.rewrapConnectorOAuthClient(ctx, "signing_secret_sealed", "signing_kek_version", customerID, connectorID, was, sealed, version)
}

func (s *Store) rewrapConnectorOAuthClient(ctx context.Context, column, versionColumn, customerID, connectorID string, was, sealed []byte, version int) (bool, error) {
	if len(was) == 0 || len(sealed) == 0 || version < 1 {
		return false, stack.Wrap(errors.New("store: a rewrap replaces one sealed secret with another, under a key version of 1 or more"))
	}
	result, err := s.db.NewUpdate().Model((*ConnectorOAuthClient)(nil)).
		Set("? = ?", bun.Ident(column), sealed).
		Set("? = ?", bun.Ident(versionColumn), version).
		Where("customer_id = ?", customerID).
		Where("connector_id = ?", connectorID).
		Where("? = ?", bun.Ident(column), was).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: rewrap connector oauth client: %w", err))
	}
	written, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: rewrap connector oauth client: %w", err))
	}
	return written > 0, nil
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

// checkOAuthClient is what the migration leaves to the store: a known registration, a
// provider app for a managed client, and each sealed secret with the key version it was
// sealed under, or neither.
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
	// The router created a managed app, so it has the id the provider gave it.
	if client.Registration == core.ClientManaged && client.ProviderAppID == "" {
		return stack.Wrap(errors.New("store: a managed OAuth client names the provider app the router created"))
	}
	// RFC 3986 section 3.3: a dot segment would be removed from the events URL it is in.
	if client.ProviderAppID != "" && (!providerAppIDPattern.MatchString(client.ProviderAppID) || client.ProviderAppID == "." || client.ProviderAppID == "..") {
		return stack.Wrap(fmt.Errorf("store: provider app id %q is not unreserved characters (RFC 3986 section 2.3) or is a dot segment", client.ProviderAppID))
	}
	if client.SigningSecretSealed == nil {
		client.SigningSecretSealed = []byte{}
	}
	if (len(client.SigningSecretSealed) == 0) != (client.SigningKEKVersion == 0) || client.SigningKEKVersion < 0 {
		return stack.Wrap(errors.New("store: a sealed signing secret has a key version of 1 or more, and no signing secret has version 0"))
	}
	if len(client.SigningSecretSealed) > 0 && client.ProviderAppID == "" {
		return stack.Wrap(errors.New("store: a signing secret is found by its provider app, so it needs a provider app id"))
	}
	return nil
}
