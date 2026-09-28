// Package connectorimport transfers the supported app-owned accounts from the retired
// per-agent connector schema before that schema is removed.
package connectorimport

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/uptrace/bun"
)

const identityReviewMessage = "Imported account needs reauthorization to verify its identity and OAuth configuration"

// Report summarizes an import without exposing account identifiers or credential data.
type Report struct {
	ImportedConnections int
	ExistingConnections int
	BindingsAdded       int
	Unsupported         int
	Inactive            int
}

type previousLogin struct {
	ID            string
	CustomerID    string
	ConfigID      string
	ConnectorID   string
	InstanceURL   string
	AccessToken   string
	RefreshToken  string
	ExpiresAt     sql.NullTime
	Status        string
	ClientID      string
	TokenEndpoint string
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// ImportSupportedAccounts copies connected logins for the built-in connector catalog into
// encrypted reusable connections. Imported accounts require reauthorization and receive no
// tool grants; the old token is retained only inside the encrypted credential bundle.
func ImportSupportedAccounts(ctx context.Context, db *bun.DB, sealer *auth.Sealer) (Report, error) {
	if db == nil {
		return Report{}, errors.New("connectorimport: database is required")
	}
	rows, err := db.QueryContext(ctx, `
		SELECT id, customer_id, config_id, plugin_id, instance_url, access_token,
			refresh_token, expires_at, status, client_id, token_endpoint, created_at, updated_at
		FROM agent_plugin_connections
		WHERE deleted_at IS NULL
		ORDER BY id`)
	if err != nil {
		return Report{}, fmt.Errorf("connectorimport: read connected accounts: %w", err)
	}
	defer rows.Close()

	var report Report
	for rows.Next() {
		var previous previousLogin
		if err := rows.Scan(
			&previous.ID, &previous.CustomerID, &previous.ConfigID, &previous.ConnectorID,
			&previous.InstanceURL, &previous.AccessToken, &previous.RefreshToken,
			&previous.ExpiresAt, &previous.Status, &previous.ClientID, &previous.TokenEndpoint,
			&previous.CreatedAt, &previous.UpdatedAt,
		); err != nil {
			return report, fmt.Errorf("connectorimport: scan connected account: %w", err)
		}
		if _, supported := mcp.Lookup(previous.ConnectorID); !supported {
			report.Unsupported++
			continue
		}
		if previous.Status != store.ConnectorConnected || previous.AccessToken == "" {
			report.Inactive++
			continue
		}
		if sealer == nil {
			return report, errors.New("connectorimport: ROUTER_AUTH_KEK is required to encrypt connected accounts before removing plaintext storage")
		}
		imported, bindingAdded, err := importOne(ctx, db, sealer, previous)
		if err != nil {
			return report, err
		}
		if imported {
			report.ImportedConnections++
		} else {
			report.ExistingConnections++
		}
		if bindingAdded {
			report.BindingsAdded++
		}
	}
	if err := rows.Err(); err != nil {
		return report, fmt.Errorf("connectorimport: read connected accounts: %w", err)
	}
	return report, nil
}

func importOne(ctx context.Context, db *bun.DB, sealer *auth.Sealer, previous previousLogin) (bool, bool, error) {
	connector, exists := mcp.Lookup(previous.ConnectorID)
	if !exists {
		return false, false, fmt.Errorf("connectorimport: connector %q is not in the supported catalog", previous.ConnectorID)
	}
	instance := previousInstance(connector.ID, previous.InstanceURL)
	endpoint, err := connector.Endpoint(instance)
	if err != nil {
		return false, false, fmt.Errorf("connectorimport: resolve %s endpoint: %w", connector.ID, err)
	}
	credentials := connectors.Credentials{
		AuthType:      connectors.AuthOAuth2,
		AccessToken:   previous.AccessToken,
		RefreshToken:  previous.RefreshToken,
		OAuthClientID: previous.ClientID,
		TokenEndpoint: previous.TokenEndpoint,
	}
	sealed, err := connectors.SealCredentials(sealer, previous.CustomerID, previous.ID, 1, credentials)
	if err != nil {
		return false, false, fmt.Errorf("connectorimport: encrypt %s account: %w", connector.ID, err)
	}
	opened, err := connectors.OpenCredentials(sealer, previous.CustomerID, previous.ID, 1, sealer.CurrentVersion(), sealed)
	if err != nil || opened.AccessToken != credentials.AccessToken || opened.RefreshToken != credentials.RefreshToken {
		return false, false, fmt.Errorf("connectorimport: verify encrypted %s account before removing plaintext storage", connector.ID)
	}
	expiresAt := nullableTime(previous.ExpiresAt)
	connection := store.ConnectorConnection{
		ID:                   previous.ID,
		CustomerID:           previous.CustomerID,
		ConnectorID:          connector.ID,
		OwnerType:            "app",
		Endpoint:             endpoint,
		Instance:             instance,
		AuthType:             connectors.AuthOAuth2,
		Status:               store.ConnectorNeedsReauth,
		GrantedScopes:        []string{},
		Revision:             1,
		CredentialSealed:     sealed,
		CredentialKEKVersion: sealer.CurrentVersion(),
		ExpiresAt:            expiresAt,
		CachedTools:          []store.ConnectorTool{},
		LastError:            identityReviewMessage,
		CreatedAt:            previous.CreatedAt,
		UpdatedAt:            previous.UpdatedAt,
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return false, false, fmt.Errorf("connectorimport: begin account import: %w", err)
	}
	defer tx.Rollback()
	result, err := tx.NewInsert().Model(&connection).On("CONFLICT (id) DO NOTHING").Exec(ctx)
	if err != nil {
		return false, false, fmt.Errorf("connectorimport: insert %s account: %w", connector.ID, err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, false, fmt.Errorf("connectorimport: inspect %s account insert: %w", connector.ID, err)
	}
	if affected == 0 {
		var customerID, connectorID, ownerType string
		if err := tx.QueryRowContext(ctx,
			"SELECT customer_id, connector_id, owner_type FROM connector_connections WHERE id = ?",
			previous.ID,
		).Scan(&customerID, &connectorID, &ownerType); err != nil {
			return false, false, fmt.Errorf("connectorimport: inspect existing account: %w", err)
		}
		if customerID != previous.CustomerID || connectorID != connector.ID || ownerType != "app" {
			return false, false, fmt.Errorf("connectorimport: connection id %q already belongs to another account", previous.ID)
		}
	}

	bindingAdded, err := importBinding(ctx, tx, previous)
	if err != nil {
		return false, false, err
	}
	if err := tx.Commit(); err != nil {
		return false, false, fmt.Errorf("connectorimport: commit %s account: %w", connector.ID, err)
	}
	return affected != 0, bindingAdded, nil
}

func importBinding(ctx context.Context, tx bun.Tx, previous previousLogin) (bool, error) {
	var rawPlugins, rawConnectors []byte
	err := tx.QueryRowContext(ctx,
		"SELECT plugins, connectors FROM agent_configs WHERE id = ? AND customer_id = ? AND deleted_at IS NULL FOR UPDATE",
		previous.ConfigID, previous.CustomerID,
	).Scan(&rawPlugins, &rawConnectors)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("connectorimport: load source agent config: %w", err)
	}
	var selected []string
	if err := json.Unmarshal(rawPlugins, &selected); err != nil {
		return false, fmt.Errorf("connectorimport: decode source connector selection: %w", err)
	}
	selectedBefore := false
	for _, id := range selected {
		if id == previous.ConnectorID {
			selectedBefore = true
			break
		}
	}
	if !selectedBefore {
		return false, nil
	}
	var bindings []store.ConnectorBinding
	if err := json.Unmarshal(rawConnectors, &bindings); err != nil {
		return false, fmt.Errorf("connectorimport: decode current connector bindings: %w", err)
	}
	for _, binding := range bindings {
		if binding.Name == previous.ConnectorID {
			return false, nil
		}
	}
	bindings = append(bindings, store.ConnectorBinding{
		Name:        previous.ConnectorID,
		ConnectorID: previous.ConnectorID,
		Connection:  store.ConnectionBinding{Type: "fixed", ConnectionID: previous.ID},
		Tools:       []store.ToolGrant{},
	})
	encoded, err := json.Marshal(bindings)
	if err != nil {
		return false, fmt.Errorf("connectorimport: encode connector bindings: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		"UPDATE agent_configs SET connectors = ?, updated_at = ? WHERE id = ? AND customer_id = ?",
		string(encoded), time.Now().UTC(), previous.ConfigID, previous.CustomerID,
	); err != nil {
		return false, fmt.Errorf("connectorimport: save imported connector binding: %w", err)
	}
	return true, nil
}

func previousInstance(connectorID, value string) string {
	if connectorID != "salesforce" {
		return value
	}
	if containsSandbox(value) {
		return "sandbox"
	}
	return "production"
}

func containsSandbox(value string) bool {
	return strings.Contains(strings.ToLower(value), "sandbox")
}

func nullableTime(value sql.NullTime) *time.Time {
	if !value.Valid {
		return nil
	}
	return &value.Time
}
