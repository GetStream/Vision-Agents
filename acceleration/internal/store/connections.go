package store

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// The two owners a connection can have (architecture doc, one-way door 8). An app-owned
// connection has no owner id; a user-owned one names the user. The migration checks it too.
const (
	OwnerApp  = "app"
	OwnerUser = "user"
)

// Connection statuses, the prototype's (internal/store/connectors.go on codex/connector-support
// at cf62af0d) less failed, which nothing in the design moves a connection to:
// core.CredentialState names connected, needs_reauthorization and disconnected, and pending
// is a connection no credentials were saved onto yet.
const (
	ConnectionPending              = "pending"
	ConnectionConnected            = "connected"
	ConnectionNeedsReauthorization = "needs_reauthorization"
	ConnectionDisconnected         = "disconnected"
)

// Attempt kinds, from the architecture doc («Add» item 5): consent and reconnect for T17,
// step_up and admin_consent for T27.
const (
	AttemptConsent      = "consent"
	AttemptReconnect    = "reconnect"
	AttemptStepUp       = "step_up"
	AttemptAdminConsent = "admin_consent"
)

// How many connections are handed back at once: the session list's sizes (sessions.go),
// since both are lists a dashboard shows a page of. Neither is measured.
const (
	defaultConnectionLimit = 25
	maxConnectionLimit     = 200
)

// ErrNoConnectorConnection says there is no live connection with that id for this customer:
// none was made, it is another customer's, or it was deleted. A sentinel, because each of
// those is an ordinary 404.
var ErrNoConnectorConnection = errors.New("store: no such connector connection")

// ErrUnregisteredScheme says a connection names a scheme no adapter is registered for.
var ErrUnregisteredScheme = errors.New("store: no such scheme is registered")

// ErrSchemeNotAllowed says a connection names a registered scheme its connector's manifest
// does not list. The manifest's schemes are the ones a connection may use
// (core.Manifest.Schemes), so a registered scheme alone is not enough.
var ErrSchemeNotAllowed = errors.New("store: the connector does not allow this scheme")

// ErrNoAuthorizationAttempt says the attempt is absent, expired, already consumed, or for a
// connection that was deleted. One error for all four, so a callback learns nothing about
// which.
var ErrNoAuthorizationAttempt = errors.New("store: authorization attempt is absent, expired or already used")

// ConnectorConnection is one account at one connector. Stored credentials are sealed by the
// caller before it reaches the store, and saved by a revisioned write (T8), never by Create.
type ConnectorConnection struct {
	bun.BaseModel `bun:"table:connector_connections,alias:cc"`

	ID          string `bun:"id,pk"`
	CustomerID  string `bun:"customer_id,notnull"`
	ConnectorID string `bun:"connector_id,notnull"`
	// DefinitionRevision is the connector_definitions revision the connection was made from.
	DefinitionRevision int    `bun:"definition_revision,notnull"`
	OwnerType          string `bun:"owner_type,notnull"`
	OwnerID            string `bun:"owner_id,notnull"`
	// AuthScheme and TLSScheme are names in the scheme registry. TLSScheme is empty when the
	// transport needs no client certificate.
	AuthScheme string `bun:"auth_scheme,notnull"`
	TLSScheme  string `bun:"tls_scheme,nullzero"`
	// Inputs are what the connection was created with; Metadata what consent captured.
	Inputs        map[string]string `bun:"inputs,type:jsonb,notnull"`
	Metadata      map[string]string `bun:"metadata,type:jsonb,notnull"`
	Label         string            `bun:"label,notnull"`
	AccountID     string            `bun:"account_id,notnull"`
	Status        string            `bun:"status,notnull"`
	GrantedScopes []string          `bun:"granted_scopes,type:jsonb,notnull"`
	// Revision advances with every new stored credentials, which are sealed against it.
	Revision int `bun:"revision,notnull"`
	// CredentialsSealed is core.StoredCredentials sealed under CredentialsKEKVersion, with
	// customer, id and revision as AAD. Empty, with version 0, until credentials are saved.
	CredentialsSealed     []byte          `bun:"credentials_sealed,notnull"`
	CredentialsKEKVersion int             `bun:"credentials_kek_version,notnull"`
	ExpiresAt             *time.Time      `bun:"expires_at"`
	CachedTools           []ConnectorTool `bun:"cached_tools,type:jsonb,notnull"`
	ToolsDigest           string          `bun:"tools_digest,notnull"`
	ToolsCheckedAt        *time.Time      `bun:"tools_checked_at"`
	LastError             string          `bun:"last_error,notnull"`
	CreatedAt             time.Time       `bun:"created_at,notnull"`
	UpdatedAt             time.Time       `bun:"updated_at,notnull"`
	DeletedAt             *time.Time      `bun:"deleted_at"`
}

// ConnectorTool is one tool a connection offered when it was last checked, with the digest
// of the schema a ToolGrant pins.
type ConnectorTool struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"input_schema"`
	SchemaDigest string         `json:"schema_digest"`
}

// ConnectionFilter picks one owner's live connections, optionally of one connector, a page
// at a time.
type ConnectionFilter struct {
	OwnerType   string
	OwnerID     string
	ConnectorID string
	Limit       int
	// After is the last connection of the previous page.
	After *ConnectionPosition
}

// ConnectionPosition is where a page of connections ended, newest first.
type ConnectionPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

// ConnectorAuthorizationAttempt is one interactive acquisition in flight. Its state is sealed
// by the caller; only the hash of the OAuth state is searchable.
type ConnectorAuthorizationAttempt struct {
	bun.BaseModel `bun:"table:connector_authorization_attempts,alias:caa"`

	// ID is chosen by the caller, so it can be sealed into the attempt before the row exists.
	ID            string     `bun:"id,pk"`
	CustomerID    string     `bun:"customer_id,notnull"`
	ConnectionID  string     `bun:"connection_id,notnull"`
	Kind          string     `bun:"kind,notnull"`
	StateHash     string     `bun:"state_hash,notnull"`
	AttemptSealed []byte     `bun:"attempt_sealed,notnull"`
	KEKVersion    int        `bun:"kek_version,notnull"`
	ExpiresAt     time.Time  `bun:"expires_at,notnull"`
	ConsumedAt    *time.Time `bun:"consumed_at"`
	CreatedAt     time.Time  `bun:"created_at,notnull"`
}

// ConnectionLimit is the page size a connection list uses for the limit asked for.
// ConnectorConnectionsByOwner returns one row more than this, so a caller can tell the page
// is not the last without counting.
func ConnectionLimit(asked int) int {
	return clampLimit(asked, defaultConnectionLimit, maxConnectionLimit)
}

// CreateConnectorConnection records a new connection, pending until credentials are saved
// onto it. Its schemes must be in registry and listed by the definition revision it pins, which
// must be one the customer can see. The registry is passed in rather than held by the store, so which
// schemes exist is decided by whoever built it, and a test can register its own.
func (s *Store) CreateConnectorConnection(ctx context.Context, registry core.Registry, connection *ConnectorConnection) error {
	if connection.CustomerID == "" || connection.ConnectorID == "" {
		return errors.New("store: a customer and a connector id are required")
	}
	if !validOwner(connection.OwnerType, connection.OwnerID) {
		return fmt.Errorf("store: owner %q with id %q: an app owner has no id and a user owner has one", connection.OwnerType, connection.OwnerID)
	}
	if _, found := registry.Schemes[connection.AuthScheme]; !found {
		return fmt.Errorf("%w: auth scheme %q", ErrUnregisteredScheme, connection.AuthScheme)
	}
	if _, found := registry.Schemes[connection.TLSScheme]; connection.TLSScheme != "" && !found {
		return fmt.Errorf("%w: tls scheme %q", ErrUnregisteredScheme, connection.TLSScheme)
	}
	// The stored credentials' AAD binds the connection id, which does not exist until this returns.
	if len(connection.CredentialsSealed) > 0 {
		return errors.New("store: credentials are saved onto a connection after it exists, not with it")
	}
	// Definitions are never updated or deleted, so a revision found here stays.
	definition, err := s.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	if !slices.Contains(definition.Manifest.Schemes, connection.AuthScheme) {
		return fmt.Errorf("%w: %s revision %d does not list auth scheme %q", ErrSchemeNotAllowed,
			connection.ConnectorID, connection.DefinitionRevision, connection.AuthScheme)
	}
	if connection.TLSScheme != "" && !slices.Contains(definition.Manifest.Schemes, connection.TLSScheme) {
		return fmt.Errorf("%w: %s revision %d does not list tls scheme %q", ErrSchemeNotAllowed,
			connection.ConnectorID, connection.DefinitionRevision, connection.TLSScheme)
	}

	// Truncated to what Postgres keeps, so the row handed back is the row a read returns.
	now := time.Now().UTC().Truncate(time.Microsecond)
	connection.ID = newID()
	connection.Status = ConnectionPending
	connection.Revision = 1
	connection.CredentialsSealed = []byte{}
	connection.CredentialsKEKVersion = 0
	connection.CreatedAt = now
	connection.UpdatedAt = now
	connection.DeletedAt = nil
	if connection.Inputs == nil {
		connection.Inputs = map[string]string{}
	}
	if connection.Metadata == nil {
		connection.Metadata = map[string]string{}
	}
	if connection.GrantedScopes == nil {
		connection.GrantedScopes = []string{}
	}
	if connection.CachedTools == nil {
		connection.CachedTools = []ConnectorTool{}
	}
	if _, err := s.db.NewInsert().Model(connection).Exec(ctx); err != nil {
		return fmt.Errorf("store: create connector connection: %w", err)
	}
	return nil
}

// ConnectorConnection returns one live connection of the customer's.
func (s *Store) ConnectorConnection(ctx context.Context, customerID, id string) (ConnectorConnection, error) {
	if customerID == "" || id == "" {
		return ConnectorConnection{}, errors.New("store: a customer and a connection id are required")
	}
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, fmt.Errorf("%w: %s", ErrNoConnectorConnection, id)
	}
	if err != nil {
		return ConnectorConnection{}, fmt.Errorf("store: connector connection: %w", err)
	}
	return connection, nil
}

// ConnectorConnectionsByOwner lists one owner's live connections, newest first, one more
// than ConnectionLimit(filter.Limit).
func (s *Store) ConnectorConnectionsByOwner(ctx context.Context, customerID string, filter ConnectionFilter) ([]ConnectorConnection, error) {
	if customerID == "" {
		return nil, errors.New("store: a customer id is required")
	}
	if !validOwner(filter.OwnerType, filter.OwnerID) {
		return nil, fmt.Errorf("store: owner %q with id %q: an app owner has no id and a user owner has one", filter.OwnerType, filter.OwnerID)
	}
	connections := []ConnectorConnection{}
	query := s.db.NewSelect().Model(&connections).
		Where("customer_id = ?", customerID).
		Where("owner_type = ?", filter.OwnerType).
		Where("owner_id = ?", filter.OwnerID).
		Where("deleted_at IS NULL")
	if filter.ConnectorID != "" {
		query = query.Where("connector_id = ?", filter.ConnectorID)
	}
	if after := filter.After; after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.
		Order("created_at DESC", "id DESC").
		Limit(ConnectionLimit(filter.Limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: list connector connections: %w", err)
	}
	return connections, nil
}

// DeleteConnectorConnection soft deletes a live connection and drops its credentials at once,
// so nothing can use it from here on. Whether a config still binds it is the caller's to ask
// first (ConnectorConnectionReferenced).
func (s *Store) DeleteConnectorConnection(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a connection id are required")
	}
	now := time.Now().UTC()
	result, err := s.db.NewUpdate().Model((*ConnectorConnection)(nil)).
		Set("status = ?", ConnectionDisconnected).
		Set("credentials_sealed = ?", []byte{}).
		Set("credentials_kek_version = 0").
		Set("expires_at = NULL").
		Set("deleted_at = ?", now).
		Set("updated_at = ?", now).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete connector connection: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete connector connection: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("%w: %s", ErrNoConnectorConnection, id)
	}
	return nil
}

// ConnectorConnectionReferenced reports whether a live agent config of the customer's binds
// the live connection as its fixed connection, so deleting it would break that agent.
//
// A binding is matched by containment on the shape the prototype stored
// ({"connection": {"type": "fixed", "connection_id": ...}}, ConnectorBinding in
// internal/store/models.go on codex/connector-support at cf62af0d), which T20 keeps. A
// customer has few configs and agent_configs_customer_idx finds them, so the column has no
// index of its own.
func (s *Store) ConnectorConnectionReferenced(ctx context.Context, customerID, id string) (bool, error) {
	if customerID == "" || id == "" {
		return false, errors.New("store: a customer and a connection id are required")
	}
	var referenced bool
	err := s.db.QueryRowContext(ctx, `
SELECT EXISTS (
    SELECT 1 FROM agent_configs AS ac
    WHERE ac.customer_id = cc.customer_id
      AND ac.deleted_at IS NULL
      AND ac.connectors @> jsonb_build_array(jsonb_build_object(
          'connection', jsonb_build_object('type', 'fixed', 'connection_id', cc.id))))
FROM connector_connections AS cc
WHERE cc.customer_id = ? AND cc.id = ? AND cc.deleted_at IS NULL`, customerID, id).Scan(&referenced)
	if errors.Is(err, sql.ErrNoRows) {
		return false, fmt.Errorf("%w: %s", ErrNoConnectorConnection, id)
	}
	if err != nil {
		return false, fmt.Errorf("store: connector connection references: %w", err)
	}
	return referenced, nil
}

// CreateConnectorAuthorizationAttempt stores a sealed attempt for a live connection of the
// customer's, after deleting the customer's expired ones. Every read already ignores an
// expired attempt, so expiry is the earliest a row is dead and the cleanup needs no age of
// its own; a consumed one is deleted once it expires too.
func (s *Store) CreateConnectorAuthorizationAttempt(ctx context.Context, attempt *ConnectorAuthorizationAttempt) error {
	if attempt.ID == "" || attempt.CustomerID == "" || attempt.ConnectionID == "" || attempt.StateHash == "" {
		return errors.New("store: an attempt needs an id, a customer, a connection and a state hash")
	}
	switch attempt.Kind {
	case AttemptConsent, AttemptReconnect, AttemptStepUp, AttemptAdminConsent:
	default:
		return fmt.Errorf("store: attempt kind %q is not one of consent, reconnect, step_up, admin_consent", attempt.Kind)
	}
	if len(attempt.AttemptSealed) == 0 || attempt.KEKVersion < 1 {
		return errors.New("store: an attempt is sealed, under a key version of 1 or more")
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	if !attempt.ExpiresAt.After(now) {
		return errors.New("store: an attempt expires in the future")
	}
	if _, err := s.ConnectorConnection(ctx, attempt.CustomerID, attempt.ConnectionID); err != nil {
		return err
	}

	if _, err := s.db.NewDelete().Model((*ConnectorAuthorizationAttempt)(nil)).
		Where("customer_id = ?", attempt.CustomerID).
		Where("expires_at <= now()").
		Exec(ctx); err != nil {
		return fmt.Errorf("store: clean expired authorization attempts: %w", err)
	}
	attempt.ConsumedAt = nil
	attempt.CreatedAt = now
	if _, err := s.db.NewInsert().Model(attempt).Exec(ctx); err != nil {
		return fmt.Errorf("store: create authorization attempt: %w", err)
	}
	return nil
}

// ConnectorAuthorizationAttemptByState reads an open attempt by its OAuth state without
// consuming it, so the callback can check the browser that began it first. A callback
// carries no app credential, so the state, not a customer, is what finds the row.
func (s *Store) ConnectorAuthorizationAttemptByState(ctx context.Context, state string) (ConnectorAuthorizationAttempt, error) {
	if state == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: an oauth state is required")
	}
	return s.openAttempt(ctx, "caa.state_hash = ?", AuthorizationStateHash(state))
}

// ConnectorAuthorizationAttemptByID reads an open attempt by id, for the launch page and the
// browser handoff, which carry no app credential either.
func (s *Store) ConnectorAuthorizationAttemptByID(ctx context.Context, id string) (ConnectorAuthorizationAttempt, error) {
	if id == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: an attempt id is required")
	}
	return s.openAttempt(ctx, "caa.id = ?", id)
}

// ConsumeConnectorAuthorizationAttempt claims an open attempt by its OAuth state, once. It
// is one statement: Postgres locks the row for the first UPDATE, and every other caller's
// UPDATE then re-reads it, finds consumed_at set and matches nothing. So of any number of
// callbacks racing with the same state exactly one gets the attempt.
func (s *Store) ConsumeConnectorAuthorizationAttempt(ctx context.Context, state string) (ConnectorAuthorizationAttempt, error) {
	if state == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: an oauth state is required")
	}
	var attempt ConnectorAuthorizationAttempt
	err := s.db.NewRaw(`
UPDATE connector_authorization_attempts AS caa
SET consumed_at = now()
WHERE caa.state_hash = ?
  AND caa.consumed_at IS NULL
  AND caa.expires_at > now()
  AND EXISTS (SELECT 1 FROM connector_connections AS cc
              WHERE cc.id = caa.connection_id AND cc.customer_id = caa.customer_id AND cc.deleted_at IS NULL)
RETURNING caa.*`, AuthorizationStateHash(state)).Scan(ctx, &attempt)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorAuthorizationAttempt{}, ErrNoAuthorizationAttempt
	}
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: consume authorization attempt: %w", err)
	}
	return attempt, nil
}

// AuthorizationStateHash is how an OAuth state is stored and looked up: hashed, so the
// table never holds the bearer value a callback presents.
func AuthorizationStateHash(state string) string {
	hash := sha256.Sum256([]byte(state))
	return hex.EncodeToString(hash[:])
}

// openAttempt is the one attempt matching where that is unconsumed, unexpired and for a
// live connection.
func (s *Store) openAttempt(ctx context.Context, where string, arg any) (ConnectorAuthorizationAttempt, error) {
	var attempt ConnectorAuthorizationAttempt
	err := s.db.NewSelect().Model(&attempt).
		Where(where, arg).
		Where("caa.consumed_at IS NULL").
		Where("caa.expires_at > now()").
		Where("EXISTS (SELECT 1 FROM connector_connections AS cc WHERE cc.id = caa.connection_id AND cc.customer_id = caa.customer_id AND cc.deleted_at IS NULL)").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorAuthorizationAttempt{}, ErrNoAuthorizationAttempt
	}
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: authorization attempt: %w", err)
	}
	return attempt, nil
}

// validOwner is the owner rule the migration's connector_connections_owner CHECK holds.
func validOwner(ownerType, ownerID string) bool {
	return ownerType == OwnerApp && ownerID == "" || ownerType == OwnerUser && ownerID != ""
}
