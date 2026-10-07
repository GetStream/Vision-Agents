package store

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// ErrConnectorConnectionBound says an unforced delete left a live connection alone because
// an agent config of the customer's binds it as its fixed connection.
var ErrConnectorConnectionBound = errors.New("store: an agent config binds this connector connection")

// ErrUnregisteredScheme says a connection names a scheme no adapter is registered for.
var ErrUnregisteredScheme = errors.New("store: no such scheme is registered")

// ErrSchemeNotAllowed says a connection names a registered scheme its connector's manifest
// does not list. The manifest's schemes are the ones a connection may use
// (core.Manifest.Schemes), so a registered scheme alone is not enough.
var ErrSchemeNotAllowed = errors.New("store: the connector does not allow this scheme")

// ErrProviderUnitTaken says another live connection of the same connector already holds the
// provider unit id, whoever's it is. A unit takes the events of one connection only
// (connector_connections_provider_unit_idx), so the second is refused rather than share it.
var ErrProviderUnitTaken = errors.New("store: another connection of this connector holds the provider unit")

// ErrNoSharedWebhook says a connection's connector is not one events URL for every customer,
// so the URL already names the customer and the connection takes no provider unit id.
var ErrNoSharedWebhook = errors.New("store: the connector's events are not routed by provider unit")

// ErrConnectorConnectionNotConnected says a connection is pending, needs a reconnect or was
// disconnected, so it holds no provider unit: only a connected one does.
var ErrConnectorConnectionNotConnected = errors.New("store: the connector connection is not connected")

// providerUnitIndex is the unique index ErrProviderUnitTaken stands for
// (20261006194000_connector_connections_provider_unit.sql).
const providerUnitIndex = "connector_connections_provider_unit_idx"

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
	// ProviderUnitID is the routing key of a shared webhook, such as a WhatsApp
	// phone_number_id: empty unless SetConnectorConnectionProviderUnit wrote it, and emptied
	// when the connection stops being connected (saveAtRevision).
	ProviderUnitID string `bun:"provider_unit_id,nullzero"`
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

	// ConnectedAt is when a consent last stored credentials (core.CredentialState.ConnectedAt).
	// Nil until one did, and for a connection connected before the column existed.
	ConnectedAt *time.Time `bun:"connected_at"`
}

// ConnectorTool is one tool a connection offered when it was last checked, with the digest
// of the schema a ToolGrant pins.
type ConnectorTool struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"input_schema"`
	SchemaDigest string         `json:"schema_digest"`
	// NeedsScopes are the scopes a call of the tool needs, as the manifest says
	// (core.ToolSpec.NeedsScopes). Absent from rows a validate wrote before it existed, which
	// read as none.
	NeedsScopes []string `json:"needs_scopes,omitempty"`
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
	return s.createConnection(ctx, registry, connection, newID())
}

// ErrConnectorConnectionExists says a row, live or deleted, already has the id a
// CreateConnectorConnectionWithID was given.
var ErrConnectorConnectionExists = errors.New("store: a connector connection already has this id")

// CreateConnectorConnectionWithID is CreateConnectorConnection under an id the caller chose,
// for router plugins migrate, which derives it from the plugin login it moves so that a
// second run finds what the first made (T61 in acceleration/docs/connectors/subtasks.md on
// connectors/planning). An id any row already has, deleted or not, is
// ErrConnectorConnectionExists and writes nothing.
func (s *Store) CreateConnectorConnectionWithID(ctx context.Context, registry core.Registry, connection *ConnectorConnection, id string) error {
	if id == "" {
		return stack.Wrap(errors.New("store: a connection id is required"))
	}
	return s.createConnection(ctx, registry, connection, id)
}

// ConnectorConnectionEvenDeleted returns the customer's connection with id, deleted or not,
// so router plugins migrate can tell a connection it made and someone deleted since from one
// it never made.
func (s *Store) ConnectorConnectionEvenDeleted(ctx context.Context, customerID, id string) (ConnectorConnection, error) {
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
	}
	if err != nil {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("store: connector connection: %w", err))
	}
	return connection, nil
}

func (s *Store) createConnection(ctx context.Context, registry core.Registry, connection *ConnectorConnection, id string) error {
	if connection.CustomerID == "" || connection.ConnectorID == "" {
		return stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	if !validOwner(connection.OwnerType, connection.OwnerID) {
		return stack.Wrap(fmt.Errorf("store: owner %q with id %q: an app owner has no id and a user owner has one", connection.OwnerType, connection.OwnerID))
	}
	if _, found := registry.Schemes[connection.AuthScheme]; !found {
		return stack.Wrap(fmt.Errorf("%w: auth scheme %q", ErrUnregisteredScheme, connection.AuthScheme))
	}
	if _, found := registry.Schemes[connection.TLSScheme]; connection.TLSScheme != "" && !found {
		return stack.Wrap(fmt.Errorf("%w: tls scheme %q", ErrUnregisteredScheme, connection.TLSScheme))
	}
	// The stored credentials' AAD binds the connection id, which does not exist until this returns.
	if len(connection.CredentialsSealed) > 0 {
		return stack.Wrap(errors.New("store: credentials are saved onto a connection after it exists, not with it"))
	}
	// A unit routes another customer's events away once it is stored, so only a write that
	// checks the connector routes by it stores one.
	if connection.ProviderUnitID != "" {
		return stack.Wrap(errors.New("store: a provider unit is set on a connection after consent, not with it"))
	}
	// Definitions are never updated or deleted, so a revision found here stays.
	definition, err := s.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	if !slices.Contains(definition.Manifest.Schemes, connection.AuthScheme) {
		return stack.Wrap(fmt.Errorf("%w: %s revision %d does not list auth scheme %q", ErrSchemeNotAllowed,
			connection.ConnectorID, connection.DefinitionRevision, connection.AuthScheme))
	}
	if connection.TLSScheme != "" && !slices.Contains(definition.Manifest.Schemes, connection.TLSScheme) {
		return stack.Wrap(fmt.Errorf("%w: %s revision %d does not list tls scheme %q", ErrSchemeNotAllowed,
			connection.ConnectorID, connection.DefinitionRevision, connection.TLSScheme))
	}

	// Truncated to what Postgres keeps, so the row handed back is the row a read returns.
	now := time.Now().UTC().Truncate(time.Microsecond)
	connection.ID = id
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
	_, err = s.db.NewInsert().Model(connection).Exec(ctx)
	// The primary key, named by Postgres's default for a table's (CREATE TABLE in
	// 20261002193000_connector_connections.sql names none).
	if constraint(err) == "connector_connections_pkey" {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrConnectorConnectionExists, id))
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: create connector connection: %w", err))
	}
	return nil
}

// ConnectorConnection returns one live connection of the customer's.
func (s *Store) ConnectorConnection(ctx context.Context, customerID, id string) (ConnectorConnection, error) {
	if customerID == "" || id == "" {
		return ConnectorConnection{}, stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
	}
	if err != nil {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("store: connector connection: %w", err))
	}
	return connection, nil
}

// ConnectorConnectionByProviderUnit returns the one live connection of the connector that
// holds the provider unit id, whichever customer's it is: the event of a shared webhook names
// no customer, and this is how the Router finds it. The caller verifies the event first.
func (s *Store) ConnectorConnectionByProviderUnit(ctx context.Context, connectorID, providerUnitID string) (ConnectorConnection, error) {
	if connectorID == "" || providerUnitID == "" {
		return ConnectorConnection{}, stack.Wrap(errors.New("store: a connector and a provider unit id are required"))
	}
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("connector_id = ?", connectorID).
		Where("provider_unit_id = ?", providerUnitID).
		Where("deleted_at IS NULL").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("%w: %s unit %s", ErrNoConnectorConnection, connectorID, providerUnitID))
	}
	if err != nil {
		return ConnectorConnection{}, stack.Wrap(fmt.Errorf("store: connector connection by provider unit: %w", err))
	}
	return connection, nil
}

// SetConnectorConnectionProviderUnit stores the provider unit a consent proved is the
// customer's on one live, connected connection of theirs, replacing any it held. Of two
// connections of one connector, only the first to store a unit holds it: the second gets
// ErrProviderUnitTaken until the first is deleted or stops being connected. A connector whose
// events URL names the customer gets ErrNoSharedWebhook (sharedWebhook).
//
// Only a connected connection holds a unit, because a grant that ended proves nothing about
// the unit any more: a WhatsApp number can move to another business account, whose consent
// must then be able to take it (Meta, «Clients can migrate their business phone numbers
// between WhatsApp Business Accounts (WABAs)»; whether the number's id stays the same is
// unverified, the page does not say,
// https://developers.facebook.com/docs/whatsapp/business-management-api/guides/migrate-phone-to-different-waba).
func (s *Store) SetConnectorConnectionProviderUnit(ctx context.Context, customerID, id, providerUnitID string) error {
	if customerID == "" || id == "" || providerUnitID == "" {
		return stack.Wrap(errors.New("store: a customer, a connection id and a provider unit id are required"))
	}
	connection, err := s.ConnectorConnection(ctx, customerID, id)
	if err != nil {
		return err
	}
	// Definitions are never updated or deleted, so the revision the connection pins stays.
	definition, err := s.ConnectorDefinition(ctx, customerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	if !sharedWebhook(definition) {
		return stack.Wrap(fmt.Errorf("%w: %s revision %d", ErrNoSharedWebhook, connection.ConnectorID, connection.DefinitionRevision))
	}
	if connection.Status != ConnectionConnected {
		return stack.Wrap(fmt.Errorf("%w: %s is %s", ErrConnectorConnectionNotConnected, id, connection.Status))
	}
	result, err := s.db.NewUpdate().Model((*ConnectorConnection)(nil)).
		Set("provider_unit_id = ?", providerUnitID).
		Set("updated_at = ?", time.Now().UTC()).
		Where("cc.customer_id = ?", customerID).
		Where("cc.id = ?", id).
		Where("cc.deleted_at IS NULL").
		// Checked again here: a save that ends the grant may commit after the read above.
		// Under READ COMMITTED an UPDATE that waited on the row re-checks its WHERE against
		// the row as committed (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED).
		Where("cc.status = ?", ConnectionConnected).
		Exec(ctx)
	if constraint(err) == providerUnitIndex {
		return stack.Wrap(fmt.Errorf("%w: %s unit %s", ErrProviderUnitTaken, connection.ConnectorID, providerUnitID))
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set provider unit: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set provider unit: %w", err))
	}
	// Deleted or no longer connected since it was read.
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrConnectorConnectionChanged, id))
	}
	return nil
}

// sharedWebhook is whether a definition's events arrive at one URL for every customer, so
// only the provider unit in an event names the customer. That is a built-in channel verified
// with the operator's own secret (core.SecretOperator: one operator app, such as a WhatsApp
// Tech Provider app, for every customer) that reads a unit from each message. A channel
// verified with the customer's own provider app has its own events URL (architecture doc on
// connectors/planning, «Decisions, 2026-10-05», item 4), so two customers may hold one unit there: two Slack apps
// installed in one workspace. A custom definition is one customer's, so it does not speak
// for the operator's app.
func sharedWebhook(definition ConnectorDefinition) bool {
	channel := definition.Manifest.Channel
	return definition.CustomerID == BuiltinCustomer && channel != nil &&
		channel.Verifier.Secret == core.SecretOperator && channel.Messages.ProviderUnitID != ""
}

// SetConnectorConnectionTools stores the tools a validate found on one live connection of the
// customer's, with the digest of the list and when it was checked, replacing what it held. It
// writes only those three columns, as a credentials write leaves them out
// (credentialColumns), so neither puts back a stale copy of the other.
func (s *Store) SetConnectorConnectionTools(ctx context.Context, customerID, id string, tools []ConnectorTool, digest string, checkedAt time.Time) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	if tools == nil {
		tools = []ConnectorTool{}
	}
	raw, err := json.Marshal(tools)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: encode connector connection tools: %w", err))
	}
	checked := checkedAt.UTC().Truncate(time.Microsecond)
	result, err := s.db.NewUpdate().Model((*ConnectorConnection)(nil)).
		Set("cached_tools = ?::jsonb", string(raw)).
		Set("tools_digest = ?", digest).
		Set("tools_checked_at = ?", checked).
		Set("updated_at = ?", time.Now().UTC()).
		Where("cc.customer_id = ?", customerID).
		Where("cc.id = ?", id).
		Where("cc.deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set connector connection tools: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set connector connection tools: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
	}
	return nil
}

// ConnectorConnectionsByOwner lists one owner's live connections, newest first, one more
// than ConnectionLimit(filter.Limit).
func (s *Store) ConnectorConnectionsByOwner(ctx context.Context, customerID string, filter ConnectionFilter) ([]ConnectorConnection, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer id is required"))
	}
	if !validOwner(filter.OwnerType, filter.OwnerID) {
		return nil, stack.Wrap(fmt.Errorf("store: owner %q with id %q: an app owner has no id and a user owner has one", filter.OwnerType, filter.OwnerID))
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
		return nil, stack.Wrap(fmt.Errorf("store: list connector connections: %w", err))
	}
	return connections, nil
}

// DeleteConnectorConnection soft deletes a live connection and drops its credentials at once,
// so nothing can use it from here on, whether or not a config still binds it. It is the
// forced delete; DeleteUnboundConnectorConnection is the one that refuses a bound connection.
func (s *Store) DeleteConnectorConnection(ctx context.Context, customerID, id string) error {
	affected, err := softDeleteConnection(ctx, s.db, customerID, id, false)
	if err != nil {
		return err
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
	}
	return nil
}

// DeleteUnboundConnectorConnection soft deletes a live connection as DeleteConnectorConnection
// does, unless a live agent config of the customer's binds it as its fixed connection.
//
// It locks the connection FOR UPDATE first, and checks for a binding in the UPDATE after.
// A config write locks each connection it binds before it writes the binding
// (lockBoundConnections), so a bind in progress holds the delete at the lock until it
// commits. Under READ COMMITTED each statement sees the rows committed before it began
// (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED), so the
// UPDATE, begun after the wait, sees a binding that committed during it. Checked in the
// locking statement itself, it would not: a statement that waits for a row lock re-checks
// only that row.
func (s *Store) DeleteUnboundConnectorConnection(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var locked string
		err := tx.NewSelect().Model((*ConnectorConnection)(nil)).Column("cc.id").
			Where("cc.customer_id = ?", customerID).
			Where("cc.id = ?", id).
			Where("cc.deleted_at IS NULL").
			For("UPDATE").
			Scan(ctx, &locked)
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: %s", ErrNoConnectorConnection, id)
		}
		if err != nil {
			return fmt.Errorf("store: lock connector connection: %w", err)
		}
		affected, err := softDeleteConnection(ctx, tx, customerID, id, true)
		if err != nil {
			return err
		}
		// The row is live and locked by this transaction, so only a binding stops the UPDATE.
		if affected == 0 {
			return fmt.Errorf("%w: %s", ErrConnectorConnectionBound, id)
		}
		return nil
	}))
}

// boundByConfig is true for a connection cc that a live agent config of the same customer
// binds as its fixed connection. ConnectorConnectionReferenced and
// DeleteUnboundConnectorConnection both use it, so what one reports bound the other refuses.
//
// A binding is matched by containment on the shape the prototype stored
// ({"connection": {"type": "fixed", "connection_id": ...}}, ConnectorBinding in
// internal/store/models.go on codex/connector-support at cf62af0d), which T20 keeps. A
// customer has few configs and agent_configs_customer_idx finds them, so the column has no
// index of its own.
const boundByConfig = `EXISTS (
    SELECT 1 FROM agent_configs AS ac
    WHERE ac.customer_id = cc.customer_id
      AND ac.deleted_at IS NULL
      AND ac.connectors @> jsonb_build_array(jsonb_build_object(
          'connection', jsonb_build_object('type', 'fixed', 'connection_id', cc.id))))`

// softDeleteConnection marks a live connection deleted and drops its credentials in one
// statement, and when unbound is set only if no config binds it. It returns the rows it
// changed: one, or none.
func softDeleteConnection(ctx context.Context, db bun.IDB, customerID, id string, unbound bool) (int64, error) {
	if customerID == "" || id == "" {
		return 0, stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	now := time.Now().UTC()
	query := db.NewUpdate().Model((*ConnectorConnection)(nil)).
		Set("status = ?", ConnectionDisconnected).
		Set("credentials_sealed = ?", []byte{}).
		Set("credentials_kek_version = 0").
		Set("expires_at = NULL").
		Set("deleted_at = ?", now).
		Set("updated_at = ?", now).
		Where("cc.customer_id = ?", customerID).
		Where("cc.id = ?", id).
		Where("cc.deleted_at IS NULL")
	if unbound {
		query = query.Where("NOT " + boundByConfig)
	}
	result, err := query.Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: delete connector connection: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: delete connector connection: %w", err))
	}
	return affected, nil
}

// DeletedConnection is one connection DeleteUserConnectorConnections removed.
type DeletedConnection struct {
	ID          string
	ConnectorID string
	// HadGrant says it still held stored credentials, which the delete revoked.
	HadGrant bool
}

// DeleteUserConnectorConnections hard deletes every connection of one user of the
// customer's, live or already soft deleted, with their attempts and their invocations, so
// the user's id and their accounts' ids are gone (architecture doc, «Add» item 9: «hard
// delete of owner_id and account_id on request»). The audit rows of those connections stay,
// with their session, request and attempt ids emptied. A user with none is not an error.
//
// The rows are locked FOR UPDATE before they go, so a credentials write in flight commits
// first or finds the row gone (ErrConnectorConnectionChanged). A connection the user makes
// after the delete began is not one it removes.
func (s *Store) DeleteUserConnectorConnections(ctx context.Context, customerID, userID string) ([]DeletedConnection, error) {
	if customerID == "" || userID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a user id are required"))
	}
	var deleted []DeletedConnection
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var rows []struct {
			ID          string `bun:"id"`
			ConnectorID string `bun:"connector_id"`
			HadGrant    bool   `bun:"had_grant"`
		}
		err := tx.NewSelect().Model((*ConnectorConnection)(nil)).
			Column("cc.id", "cc.connector_id").
			ColumnExpr("cc.credentials_sealed <> ''::bytea AS had_grant").
			Where("cc.customer_id = ?", customerID).
			Where("cc.owner_type = ?", OwnerUser).
			Where("cc.owner_id = ?", userID).
			For("UPDATE").
			Scan(ctx, &rows)
		if err != nil {
			return fmt.Errorf("store: lock a user's connector connections: %w", err)
		}
		if len(rows) == 0 {
			return nil
		}
		ids := make([]string, 0, len(rows))
		for _, row := range rows {
			ids = append(ids, row.ID)
			deleted = append(deleted, DeletedConnection{ID: row.ID, ConnectorID: row.ConnectorID, HadGrant: row.HadGrant})
		}
		// Attempts reference their connection with no cascade (20261002193100), so they go
		// first. Invocations cascade.
		if _, err := tx.NewDelete().Model((*ConnectorAuthorizationAttempt)(nil)).
			Where("caa.connection_id IN (?)", bun.In(ids)).Exec(ctx); err != nil {
			return fmt.Errorf("store: delete a user's authorization attempts: %w", err)
		}
		if _, err := tx.NewDelete().Model((*ConnectorConnection)(nil)).
			Where("cc.customer_id = ?", customerID).
			Where("cc.id IN (?)", bun.In(ids)).Exec(ctx); err != nil {
			return fmt.Errorf("store: delete a user's connector connections: %w", err)
		}
		// The audit keeps the connections' rows, which name no user, but its correlation ids
		// lead back to one: a session id joins agent_sessions.user_id, and a request id and an
		// attempt id are found in the access log beside the user's requests.
		if _, err := tx.NewUpdate().Model((*ConnectorAuditEvent)(nil)).
			Set("session_id = ''").
			Set("request_id = ''").
			Set("attempt_id = ''").
			Where("ca.customer_id = ?", customerID).
			Where("ca.connection_id IN (?)", bun.In(ids)).Exec(ctx); err != nil {
			return fmt.Errorf("store: unlink a user's connector audit: %w", err)
		}
		return nil
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return deleted, nil
}

// ConnectionUse is one binding of a live agent config that names a connection as its fixed
// connection: what deleting the connection would break.
type ConnectionUse struct {
	ConfigID   string `bun:"config_id"`
	ConfigName string `bun:"config_name"`
	// Binding is the alias the config binds it under.
	Binding      string `bun:"binding"`
	ConnectionID string `bun:"connection_id"`
}

// ConnectorConnectionUses are the bindings of the customer's live agent configs that name
// each of ids as their fixed connection, by connection id, each list by config name and
// alias. A binding a session fills names no connection, so it is never one of them. It reads
// the shape boundByConfig matches.
func (s *Store) ConnectorConnectionUses(ctx context.Context, customerID string, ids []string) (map[string][]ConnectionUse, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer id is required"))
	}
	uses := map[string][]ConnectionUse{}
	if len(ids) == 0 {
		return uses, nil
	}
	var rows []ConnectionUse
	// A row whose connectors is not an array binds nothing, rather than failing the read: the
	// column is jsonb, and a writer that skips the store can put anything there.
	err := s.db.NewSelect().
		TableExpr("agent_configs AS ac").
		Join("CROSS JOIN LATERAL jsonb_array_elements(CASE WHEN jsonb_typeof(ac.connectors) = 'array' THEN ac.connectors ELSE '[]'::jsonb END) AS binding").
		ColumnExpr("ac.id AS config_id, ac.name AS config_name").
		ColumnExpr("binding ->> 'name' AS binding").
		ColumnExpr("binding -> 'connection' ->> 'connection_id' AS connection_id").
		Where("ac.customer_id = ?", customerID).
		Where("ac.deleted_at IS NULL").
		Where("binding -> 'connection' ->> 'type' = 'fixed'").
		Where("binding -> 'connection' ->> 'connection_id' IN (?)", bun.In(ids)).
		OrderExpr("ac.name, ac.id, binding ->> 'name'").
		Scan(ctx, &rows)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: connector connection uses: %w", err))
	}
	for _, row := range rows {
		uses[row.ConnectionID] = append(uses[row.ConnectionID], row)
	}
	return uses, nil
}

// ConnectorConnectionReferenced reports whether a live agent config of the customer's binds
// the live connection as its fixed connection, so deleting it would break that agent.
func (s *Store) ConnectorConnectionReferenced(ctx context.Context, customerID, id string) (bool, error) {
	if customerID == "" || id == "" {
		return false, stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	var referenced bool
	err := s.db.QueryRowContext(ctx, `
SELECT `+boundByConfig+`
FROM connector_connections AS cc
WHERE cc.customer_id = ? AND cc.id = ? AND cc.deleted_at IS NULL`, customerID, id).Scan(&referenced)
	if errors.Is(err, sql.ErrNoRows) {
		return false, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
	}
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: connector connection references: %w", err))
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

// HandOffConnectorAuthorizationAttempt replaces an open attempt's sealed blob with sealed,
// only while the row still holds previous, the blob the caller read. A seal has a fresh
// random nonce (auth.Sealer.SealWithAAD), so no two seals are equal: of any number of
// handoffs racing from the same read exactly one replaces it, as of racing callbacks one
// consumes it.
func (s *Store) HandOffConnectorAuthorizationAttempt(ctx context.Context, id string, previous, sealed []byte, kekVersion int) error {
	if id == "" || len(previous) == 0 || len(sealed) == 0 || kekVersion < 1 {
		return errors.New("store: a handoff needs an attempt id, the blob it read and a new one under a key version of 1 or more")
	}
	result, err := s.db.NewRaw(`
UPDATE connector_authorization_attempts AS caa
SET attempt_sealed = ?, kek_version = ?
WHERE caa.id = ?
  AND caa.attempt_sealed = ?
  AND caa.consumed_at IS NULL
  AND caa.expires_at > now()
  AND EXISTS (SELECT 1 FROM connector_connections AS cc
              WHERE cc.id = caa.connection_id AND cc.customer_id = caa.customer_id AND cc.deleted_at IS NULL)`,
		sealed, kekVersion, id, previous).Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: hand off authorization attempt: %w", err)
	}
	replaced, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: hand off authorization attempt: %w", err)
	}
	if replaced == 0 {
		return ErrNoAuthorizationAttempt
	}
	return nil
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
