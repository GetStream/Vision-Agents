//go:build integration

package connectorimport

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/stretchr/testify/require"
)

func TestImportSupportedAccounts(t *testing.T) {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN must be set")
	}

	ctx := context.Background()
	db, err := store.Open(dsn)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	require.NoError(t, db.Ping(ctx))
	var database string
	require.NoError(t, db.DB().QueryRowContext(ctx, "SELECT current_database()").Scan(&database))
	require.True(t, strings.HasSuffix(database, "_test"), "connector import integration test must use a test database")
	require.NoError(t, db.Migrate(ctx))

	_, err = db.DB().ExecContext(ctx, `
		ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS plugins JSONB NOT NULL DEFAULT '[]';
		CREATE TABLE IF NOT EXISTS agent_plugin_connections (
			id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, config_id TEXT NOT NULL, plugin_id TEXT NOT NULL,
			instance_url TEXT NOT NULL DEFAULT '', access_token TEXT NOT NULL DEFAULT '',
			refresh_token TEXT NOT NULL DEFAULT '', expires_at TIMESTAMPTZ, status TEXT NOT NULL DEFAULT 'pending',
			oauth_state TEXT NOT NULL DEFAULT '', code_verifier TEXT NOT NULL DEFAULT '',
			client_id TEXT NOT NULL DEFAULT '', token_endpoint TEXT NOT NULL DEFAULT '',
			created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), deleted_at TIMESTAMPTZ
		)`)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, cleanupErr := db.DB().ExecContext(ctx, "DROP TABLE IF EXISTS agent_plugin_connections")
		require.NoError(t, cleanupErr)
		_, cleanupErr = db.DB().ExecContext(ctx, "ALTER TABLE agent_configs DROP COLUMN IF EXISTS plugins")
		require.NoError(t, cleanupErr)
	})

	stamp := time.Now().UnixNano()
	customerID := fmt.Sprintf("connector-import-test-%d", stamp)
	connectionID := fmt.Sprintf("connector-import-account-%d", stamp)
	unsupportedID := fmt.Sprintf("connector-import-unsupported-%d", stamp)
	config := store.AgentConfig{CustomerID: customerID, Name: fmt.Sprintf("connector-import-%d", stamp)}
	require.NoError(t, db.CreateAgentConfig(ctx, &config))
	_, err = db.DB().ExecContext(ctx, "UPDATE agent_configs SET plugins = ? WHERE id = ?", `["slack"]`, config.ID)
	require.NoError(t, err)
	now := time.Now().UTC()
	_, err = db.DB().ExecContext(ctx, `
		INSERT INTO agent_plugin_connections
			(id, customer_id, config_id, plugin_id, access_token, refresh_token, status, client_id, token_endpoint, created_at, updated_at)
		VALUES (?, ?, ?, 'slack', 'access-value', 'refresh-value', 'connected', 'old-client', 'https://slack.com/api/oauth.v2.user.access', ?, ?),
		       (?, ?, ?, 'shopify', 'unsupported-token', 'unsupported-refresh', 'connected', 'old-client', 'https://shopify.example/token', ?, ?)`,
		connectionID, customerID, config.ID, now, now,
		unsupportedID, customerID, config.ID, now, now,
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, cleanupErr := db.DB().ExecContext(ctx, "DELETE FROM agent_plugin_connections WHERE customer_id = ?", customerID)
		require.NoError(t, cleanupErr)
		_, cleanupErr = db.DB().ExecContext(ctx, "DELETE FROM connector_connections WHERE customer_id = ?", customerID)
		require.NoError(t, cleanupErr)
		_, cleanupErr = db.DB().ExecContext(ctx, "DELETE FROM agent_configs WHERE customer_id = ?", customerID)
		require.NoError(t, cleanupErr)
	})

	sealer, err := auth.NewSealer("connector-import-integration-key")
	require.NoError(t, err)
	_, err = ImportSupportedAccounts(ctx, db.DB(), nil)
	require.ErrorContains(t, err, "ROUTER_AUTH_KEK is required")
	report, err := ImportSupportedAccounts(ctx, db.DB(), sealer)
	require.NoError(t, err)
	require.Equal(t, Report{ImportedConnections: 1, BindingsAdded: 1, Unsupported: 1}, report)

	connection, err := db.ConnectorConnection(ctx, customerID, connectionID)
	require.NoError(t, err)
	require.Equal(t, store.ConnectorNeedsReauth, connection.Status)
	require.Empty(t, connection.GrantedScopes)
	require.Empty(t, connection.CachedTools)
	require.NotContains(t, string(connection.CredentialSealed), "access-value")
	credentials, err := connectors.OpenCredentials(sealer, customerID, connectionID, connection.Revision,
		connection.CredentialKEKVersion, connection.CredentialSealed)
	require.NoError(t, err)
	require.Equal(t, "access-value", credentials.AccessToken)
	require.Equal(t, "refresh-value", credentials.RefreshToken)

	stored, err := db.AgentConfig(ctx, customerID, config.ID)
	require.NoError(t, err)
	require.Len(t, stored.Connectors, 1)
	require.Equal(t, "slack", stored.Connectors[0].Name)
	require.Equal(t, connectionID, stored.Connectors[0].Connection.ConnectionID)
	require.Empty(t, stored.Connectors[0].Tools)
	encoded, err := json.Marshal(stored.Connectors[0].Tools)
	require.NoError(t, err)
	require.JSONEq(t, "[]", string(encoded))

	report, err = ImportSupportedAccounts(ctx, db.DB(), sealer)
	require.NoError(t, err)
	require.Equal(t, Report{ExistingConnections: 1, Unsupported: 1}, report)
}
