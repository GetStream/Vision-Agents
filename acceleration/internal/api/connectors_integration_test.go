//go:build integration

package api

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

func TestForkUsesCurrentConnectorGrantsAndOnlyCurrentSessionSelections(t *testing.T) {
	require := require.New(t)
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN must be set")
	}
	ctx := context.Background()
	db, err := store.Open(dsn)
	require.NoError(err)
	t.Cleanup(func() { require.NoError(db.Close()) })
	require.NoError(db.Ping(ctx))
	var database string
	require.NoError(db.DB().QueryRowContext(ctx, "SELECT current_database()").Scan(&database))
	require.True(strings.HasSuffix(database, "_test"), "refusing to update connector grants in %s", database)
	require.NoError(db.Migrate(ctx))

	customerID := "connector-fork-api-" + time.Now().UTC().Format("20060102150405.000000000")
	config := store.AgentConfig{
		CustomerID: customerID,
		Name:       "CRM",
		Connectors: []store.ConnectorBinding{
			{
				Name: "crm", ConnectorID: "salesforce",
				Connection: store.ConnectionBinding{Type: "session"},
				Tools:      []store.ToolGrant{{Name: "query_records", SchemaDigest: "current-schema"}},
			},
			{
				Name: "calendar", ConnectorID: "calendly",
				Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: "app-calendar"},
				Tools:      []store.ToolGrant{{Name: "list_events", SchemaDigest: "calendar-schema"}},
			},
		},
	}
	require.NoError(db.CreateAgentConfig(ctx, &config))
	t.Cleanup(func() { require.NoError(db.DeleteAgentConfig(ctx, customerID, config.ID)) })

	parent := store.AgentSession{
		ID:         "connector-fork-" + time.Now().UTC().Format("150405.000000000"),
		CustomerID: customerID,
		ConfigID:   config.ID,
		AgentName:  config.Name,
		ConnectorSelections: []store.SessionConnectorSelection{
			{Name: "crm", ConnectionID: "user-salesforce"},
			{Name: "old", ConnectionID: "user-slack"},
			{Name: "calendar", ConnectionID: "user-calendar"},
		},
	}
	require.NoError(db.SaveSession(ctx, &parent))
	t.Cleanup(func() {
		_, err := db.DB().ExecContext(ctx, "DELETE FROM agent_sessions WHERE id = ?", parent.ID)
		require.NoError(err)
	})
	stored, err := db.StoredSession(ctx, customerID, parent.ID)
	require.NoError(err)
	spec, err := forkSpec(session.Found{Stored: &stored}, ForkSessionRequest{}, nil)
	require.NoError(err)

	require.NoError((&Server{store: db}).revalidateForkConnectors(ctx, customerID, &spec))
	require.Equal(config.Connectors, spec.ConnectorBindings)
	require.Equal([]session.ConnectorSelection{{Name: "crm", ConnectionID: "user-salesforce"}}, spec.ConnectorSelections,
		"a fork keeps the caller's account choice only for a binding that is still session-selected")
}
