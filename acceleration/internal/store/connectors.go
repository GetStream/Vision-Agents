package store

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"database/sql/driver"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

var ErrConnectorConnectionChanged = errors.New("store: connector connection changed while saving")
var ErrConnectorDefinitionNotFound = errors.New("store: connector definition not found")
var ErrConnectorConnectionNotFound = errors.New("store: there is no connector connection")

// ConnectorConnectionReferenced reports whether any live agent config still binds the
// app-owned connection. Legacy disconnect facades must not delete a connection shared by
// a second agent.
func (s *Store) ConnectorConnectionReferenced(ctx context.Context, customerID, connectionID string) (bool, error) {
	if customerID == "" || connectionID == "" {
		return false, errors.New("store: tenant and connector connection are required")
	}
	rows, err := s.db.QueryContext(ctx,
		"SELECT connectors FROM agent_configs WHERE customer_id = ? AND deleted_at IS NULL", customerID)
	if err != nil {
		return false, fmt.Errorf("store: list connector bindings: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			return false, fmt.Errorf("store: scan connector bindings: %w", err)
		}
		var bindings []ConnectorBinding
		if err := json.Unmarshal(raw, &bindings); err != nil {
			return false, fmt.Errorf("store: decode connector bindings: %w", err)
		}
		for _, binding := range bindings {
			if binding.Connection.Type == "fixed" && binding.Connection.ConnectionID == connectionID {
				return true, nil
			}
		}
	}
	if err := rows.Err(); err != nil {
		return false, fmt.Errorf("store: read connector bindings: %w", err)
	}
	return false, nil
}

// CreateConnectorConnection records a reusable account connection. Credentials are sealed
// by the API layer before they reach the store.
func (s *Store) CreateConnectorConnection(ctx context.Context, connection *ConnectorConnection) error {
	if connection.CustomerID == "" || connection.ConnectorID == "" || connection.Endpoint == "" {
		return errors.New("store: customer, connector and endpoint are required")
	}
	if connection.OwnerType != "app" && connection.OwnerType != "user" {
		return errors.New("store: connector owner must be app or user")
	}
	if connection.OwnerType == "app" && connection.OwnerID != "" ||
		connection.OwnerType == "user" && connection.OwnerID == "" {
		return errors.New("store: connector owner id does not match its owner type")
	}
	if connection.AuthType == "" {
		connection.AuthType = "oauth2"
	}
	switch connection.AuthType {
	case "oauth2", "none", "bearer":
		if connection.AuthHeader != "" {
			return errors.New("store: only API key connections may name an auth header")
		}
	case "api_key":
		if connection.AuthHeader == "" {
			return errors.New("store: API key connections need an auth header")
		}
	default:
		return errors.New("store: unsupported connector authentication mode")
	}
	now := time.Now().UTC()
	connection.ID = newID()
	connection.Status = ConnectorPending
	connection.Revision = 1
	connection.CredentialKEKVersion = 1
	connection.CreatedAt = now
	connection.UpdatedAt = now
	if connection.GrantedScopes == nil {
		connection.GrantedScopes = []string{}
	}
	if connection.CachedTools == nil {
		connection.CachedTools = []ConnectorTool{}
	}
	if connection.CredentialSealed == nil {
		connection.CredentialSealed = []byte{}
	}
	if _, err := s.db.NewInsert().Model(connection).Exec(ctx); err != nil {
		return fmt.Errorf("store: create connector connection: %w", err)
	}
	return nil
}

// ConnectorConnection returns one live connection owned by a customer.
func (s *Store) ConnectorConnection(ctx context.Context, customerID, id string) (ConnectorConnection, error) {
	var connection ConnectorConnection
	err := s.db.NewSelect().Model(&connection).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorConnection{}, unknownConnectorConnection(id)
	}
	if err != nil {
		return ConnectorConnection{}, fmt.Errorf("store: connector connection: %w", err)
	}
	return connection, nil
}

// ConnectorConnections lists an app's live connections, newest first.
func (s *Store) ConnectorConnections(ctx context.Context, customerID string) ([]ConnectorConnection, error) {
	if customerID == "" {
		return nil, errors.New("store: customer is required")
	}
	var connections []ConnectorConnection
	err := s.db.NewSelect().Model(&connections).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: connector connections: %w", err)
	}
	return connections, nil
}

// CreateConnectorDefinition stores an immutable tenant-owned MCP endpoint and auth policy.
func (s *Store) CreateConnectorDefinition(ctx context.Context, definition *ConnectorDefinition) error {
	if definition.CustomerID == "" || definition.ID == "" || definition.Name == "" || definition.Endpoint == "" {
		return errors.New("store: connector definition requires a tenant, id, name and endpoint")
	}
	if definition.Category == "" {
		definition.Category = "Custom"
	}
	now := time.Now().UTC()
	definition.CreatedAt = now
	definition.UpdatedAt = now
	if _, err := s.db.NewInsert().Model(definition).Exec(ctx); err != nil {
		return fmt.Errorf("store: create connector definition: %w", err)
	}
	return nil
}

// ConnectorDefinition returns one connector definition visible to the app.
func (s *Store) ConnectorDefinition(ctx context.Context, customerID, id string) (ConnectorDefinition, error) {
	var definition ConnectorDefinition
	err := s.db.NewSelect().Model(&definition).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, unknownConnectorDefinition(id)
	}
	if err != nil {
		return ConnectorDefinition{}, fmt.Errorf("store: connector definition: %w", err)
	}
	return definition, nil
}

// ConnectorDefinitions lists the app's immutable custom connector definitions.
func (s *Store) ConnectorDefinitions(ctx context.Context, customerID string) ([]ConnectorDefinition, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	var definitions []ConnectorDefinition
	err := s.db.NewSelect().Model(&definitions).
		Where("customer_id = ?", customerID).
		Order("name ASC", "id ASC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: connector definitions: %w", err)
	}
	return definitions, nil
}

// SaveConnectorConnection stores a credential bundle or refreshed discovery metadata.
func (s *Store) SaveConnectorConnection(ctx context.Context, connection *ConnectorConnection) error {
	return s.SaveConnectorConnectionAtRevision(ctx, connection, connection.Revision)
}

// SaveConnectorConnectionAtRevision updates metadata and credentials only when the stored
// grant is still at expectedRevision. Credential changes should advance connection.Revision.
func (s *Store) SaveConnectorConnectionAtRevision(
	ctx context.Context,
	connection *ConnectorConnection,
	expectedRevision int,
) error {
	if connection.ID == "" {
		return errors.New("store: connector connection id is required")
	}
	if expectedRevision < 1 {
		return errors.New("store: connector connection revision is required")
	}
	connection.UpdatedAt = time.Now().UTC()
	return s.withConnectorConnectionLock(ctx, connection.CustomerID, connection.ID, func(conn bun.Conn) error {
		result, err := s.db.NewUpdate().Model(connection).
			Column("status", "account_id", "granted_scopes", "revision", "credential_sealed", "credential_kek_version",
				"expires_at", "cached_tools", "tools_digest", "tools_checked_at", "last_error", "updated_at").
			Where("id = ?", connection.ID).
			Where("customer_id = ?", connection.CustomerID).
			Where("revision = ?", expectedRevision).
			Where("deleted_at IS NULL").
			Conn(conn).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: save connector connection: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: save connector connection: %w", err)
		}
		if affected == 0 {
			return ErrConnectorConnectionChanged
		}
		return nil
	})
}

// WithLockedConnectorConnection reloads one connection under a Postgres advisory lock. The
// callback can checkpoint before an external side effect without releasing the lock or
// keeping a transaction open, then return changed=true to persist the final bundle.
func (s *Store) WithLockedConnectorConnection(
	ctx context.Context,
	customerID, id string,
	update func(*ConnectorConnection, func() error) (bool, error),
) error {
	return s.withConnectorConnectionLock(ctx, customerID, id, func(conn bun.Conn) error {
		var connection ConnectorConnection
		err := s.db.NewSelect().Model(&connection).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Where("deleted_at IS NULL").
			Limit(1).
			Conn(conn).
			Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			return unknownConnectorConnection(id)
		}
		if err != nil {
			return fmt.Errorf("store: load locked connector connection: %w", err)
		}
		expectedRevision := connection.Revision
		checkpoint := func() error {
			connection.UpdatedAt = time.Now().UTC()
			result, err := s.db.NewUpdate().Model(&connection).
				Column("status", "granted_scopes", "revision", "credential_sealed", "credential_kek_version",
					"expires_at", "cached_tools", "tools_digest", "tools_checked_at", "last_error", "updated_at").
				Where("id = ?", connection.ID).
				Where("customer_id = ?", customerID).
				Where("revision = ?", expectedRevision).
				Where("deleted_at IS NULL").
				Conn(conn).
				Exec(ctx)
			if err != nil {
				return fmt.Errorf("store: update connector under advisory lock: %w", err)
			}
			applied, err := result.RowsAffected()
			if err != nil {
				return fmt.Errorf("store: update connector under advisory lock: %w", err)
			}
			if applied == 0 {
				return ErrConnectorConnectionChanged
			}
			expectedRevision = connection.Revision
			return nil
		}
		changed, err := update(&connection, checkpoint)
		if err != nil {
			return err
		}
		if !changed {
			return nil
		}
		return checkpoint()
	})
}

// DeleteConnectorConnection immediately blocks future use and removes the local credential.
func (s *Store) DeleteConnectorConnection(ctx context.Context, customerID, id string) error {
	return s.withConnectorConnectionLock(ctx, customerID, id, func(conn bun.Conn) error {
		now := time.Now().UTC()
		result, err := s.db.NewUpdate().Model((*ConnectorConnection)(nil)).
			Set("status = ?", ConnectorDisconnected).
			Set("credential_sealed = ?", []byte{}).
			Set("expires_at = NULL").
			Set("last_error = ''").
			Set("deleted_at = ?", now).
			Set("updated_at = ?", now).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Where("deleted_at IS NULL").
			Conn(conn).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: delete connector connection: %w", err)
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("store: delete connector connection: %w", err)
		}
		if affected == 0 {
			return unknownConnectorConnection(id)
		}
		return nil
	})
}

func (s *Store) withConnectorConnectionLock(ctx context.Context, customerID, id string, run func(bun.Conn) error) error {
	if customerID == "" || id == "" {
		return errors.New("store: connector lock requires a tenant and connection id")
	}
	digest := sha256.Sum256([]byte(customerID + "\x00" + id))
	lockKey := int64(binary.BigEndian.Uint64(digest[:8]))
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("store: open connector coordination connection: %w", err)
	}
	defer conn.Close()
	var locked bool
	if err := conn.QueryRowContext(ctx, "SELECT pg_advisory_lock(?) IS NULL", lockKey).Scan(&locked); err != nil {
		return fmt.Errorf("store: acquire connector advisory lock: %w", err)
	}
	defer func() {
		unlockCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		var unlocked bool
		if err := conn.QueryRowContext(unlockCtx, "SELECT pg_advisory_unlock(?) IS NULL", lockKey).Scan(&unlocked); err != nil {
			_ = conn.Raw(func(any) error { return driver.ErrBadConn })
		}
	}()
	return run(conn)
}

// CreateConnectorAuthorizationAttempt writes an encrypted, expiring OAuth callback state.
func (s *Store) CreateConnectorAuthorizationAttempt(ctx context.Context, attempt *ConnectorAuthorizationAttempt) error {
	if attempt.ID == "" || attempt.CustomerID == "" || attempt.ConnectionID == "" || attempt.StateHash == "" || attempt.KEKVersion < 1 {
		return errors.New("store: authorization attempt identity is required")
	}
	attempt.CreatedAt = time.Now().UTC()
	if _, err := s.db.NewDelete().Model((*ConnectorAuthorizationAttempt)(nil)).
		Where("customer_id = ?", attempt.CustomerID).
		Where("(expires_at <= ? OR consumed_at IS NOT NULL)", attempt.CreatedAt).
		Exec(ctx); err != nil {
		return fmt.Errorf("store: clean expired connector authorization attempts: %w", err)
	}
	if _, err := s.db.NewInsert().Model(attempt).Exec(ctx); err != nil {
		return fmt.Errorf("store: create connector authorization attempt: %w", err)
	}
	return nil
}

// ConnectorAuthorizationAttemptByState reads an unexpired attempt without consuming it,
// so the callback can verify the initiating browser before atomically claiming the state.
func (s *Store) ConnectorAuthorizationAttemptByState(ctx context.Context, state string) (ConnectorAuthorizationAttempt, error) {
	if state == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: oauth state is required")
	}
	var attempt ConnectorAuthorizationAttempt
	err := s.db.NewSelect().Model(&attempt).
		Where("state_hash = ?", OAuthStateHash(state)).
		Where("consumed_at IS NULL").
		Where("expires_at > ?", time.Now().UTC()).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorAuthorizationAttempt{}, errors.New("store: authorization state is absent, expired or already used")
	}
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: find connector authorization attempt: %w", err)
	}
	return attempt, nil
}

// ConnectorAuthorizationAttemptByID reads an unexpired attempt for the browser handoff.
func (s *Store) ConnectorAuthorizationAttemptByID(ctx context.Context, id string) (ConnectorAuthorizationAttempt, error) {
	if id == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: authorization attempt ID is required")
	}
	var attempt ConnectorAuthorizationAttempt
	err := s.db.NewSelect().Model(&attempt).
		Where("id = ?", id).
		Where("consumed_at IS NULL").
		Where("expires_at > ?", time.Now().UTC()).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorAuthorizationAttempt{}, errors.New("store: authorization attempt is absent, expired or already used")
	}
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: find connector authorization attempt: %w", err)
	}
	return attempt, nil
}

// ConsumeConnectorAuthorizationAttempt atomically consumes an unexpired state exactly once.
func (s *Store) ConsumeConnectorAuthorizationAttempt(ctx context.Context, state string) (ConnectorAuthorizationAttempt, error) {
	if state == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: oauth state is required")
	}
	stateHash := OAuthStateHash(state)
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: begin authorization state consumption: %w", err)
	}
	defer tx.Rollback()
	var attempt ConnectorAuthorizationAttempt
	err = tx.NewSelect().Model(&attempt).
		Where("state_hash = ?", stateHash).
		Where("consumed_at IS NULL").
		Where("expires_at > ?", time.Now().UTC()).
		For("UPDATE").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorAuthorizationAttempt{}, errors.New("store: authorization state is absent, expired or already used")
	}
	if err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: consume connector authorization attempt: %w", err)
	}
	if attempt.ID == "" {
		return ConnectorAuthorizationAttempt{}, errors.New("store: authorization state is absent, expired or already used")
	}
	if _, err := tx.NewDelete().Model(&attempt).Where("id = ?", attempt.ID).Exec(ctx); err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: consume connector authorization attempt: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return ConnectorAuthorizationAttempt{}, fmt.Errorf("store: commit authorization state consumption: %w", err)
	}
	return attempt, nil
}

// OAuthStateHash makes callback state searchable without storing the bearer value itself.
func OAuthStateHash(state string) string {
	hash := sha256.Sum256([]byte(state))
	return hex.EncodeToString(hash[:])
}

const (
	ConnectorPending      = "pending"
	ConnectorConnected    = "connected"
	ConnectorNeedsReauth  = "needs_reauthorization"
	ConnectorDisconnected = "disconnected"
	ConnectorFailed       = "failed"
)

func unknownConnectorConnection(id string) error {
	return fmt.Errorf("%w %s", ErrConnectorConnectionNotFound, id)
}

func unknownConnectorDefinition(id string) error {
	return fmt.Errorf("%w: %s", ErrConnectorDefinitionNotFound, id)
}
