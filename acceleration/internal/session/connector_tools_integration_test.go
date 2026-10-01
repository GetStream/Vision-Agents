//go:build integration

package session

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

func TestConnectorSessionSelectionIsScopedToVerifiedCaller(t *testing.T) {
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
	require.True(strings.HasSuffix(database, "_test"), "refusing to write connector sessions to %s", database)
	require.NoError(db.Migrate(ctx))

	sealer, err := auth.NewSealer("connector-session-owner-test-key")
	require.NoError(err)
	customerID := "connector-session-owner-" + time.Now().UTC().Format("20060102150405.000000000")
	credentials := map[string]string{"alice": "alice-crm-token", "bob": "bob-crm-token"}
	connectionIDs := make(map[string]string, len(credentials))
	for userID, token := range credentials {
		connection := store.ConnectorConnection{
			CustomerID:  customerID,
			ConnectorID: "custom_crm",
			OwnerType:   "user",
			OwnerID:     userID,
			Endpoint:    "https://127.0.0.1/mcp",
			AuthType:    connectors.AuthBearer,
		}
		require.NoError(db.CreateConnectorConnection(ctx, &connection))
		connectionIDs[userID] = connection.ID
		connectionID := connection.ID
		t.Cleanup(func() { require.NoError(db.DeleteConnectorConnection(ctx, customerID, connectionID)) })
		connection.Revision++
		connection.Status = store.ConnectorConnected
		connection.CredentialKEKVersion = sealer.CurrentVersion()
		connection.CredentialSealed, err = connectors.SealCredentials(sealer, customerID, connection.ID, connection.Revision,
			connectors.Credentials{AuthType: connectors.AuthBearer, AccessToken: token})
		require.NoError(err)
		require.NoError(db.SaveConnectorConnectionAtRevision(ctx, &connection, connection.Revision-1))
	}
	config := store.AgentConfig{
		CustomerID: customerID,
		Name:       "connector-owner-selection",
		Connectors: []store.ConnectorBinding{{
			Name: "crm", ConnectorID: "custom_crm",
			Connection: store.ConnectionBinding{Type: "session"},
		}},
	}
	require.NoError(db.CreateAgentConfig(ctx, &config))
	t.Cleanup(func() { require.NoError(db.DeleteAgentConfig(ctx, customerID, config.ID)) })

	for userID, expectedToken := range credentials {
		connectionID := connectionIDs[userID]
		resolved, err := connectors.ResolveCredentials(ctx, db, sealer, customerID, connectionID, nil)
		require.NoError(err)
		require.Equal(expectedToken, resolved.AccessToken)
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://127.0.0.1/mcp", nil)
		require.NoError(err)
		require.NoError(connectors.AuthorizeRequest(ctx, db, sealer, customerID, connectionID, (*mcp.OAuthClient)(nil), request))
		require.Equal("Bearer "+expectedToken, request.Header.Get("Authorization"))

		_, _, _, attachErr := attachConnectors(ctx, Spec{
			CustomerID:          customerID,
			ConfigID:            config.ID,
			Caller:              routing.Caller{UserID: userID},
			CallerKind:          auth.KindAuthenticated,
			ConnectorBindings:   config.Connectors,
			ConnectorSelections: []ConnectorSelection{{Name: "crm", ConnectionID: connectionID}},
		}, db, sealer, slog.New(slog.NewTextHandler(io.Discard, nil)), nil)
		require.NoError(attachErr, "the authenticated owner may select their own connection")
	}

	for _, callerKind := range []auth.Kind{auth.KindAnonymous, auth.KindGuest} {
		_, _, _, attachErr := attachConnectors(ctx, Spec{
			CustomerID:          customerID,
			ConfigID:            config.ID,
			Caller:              routing.Caller{UserID: "alice"},
			CallerKind:          callerKind,
			ConnectorBindings:   config.Connectors,
			ConnectorSelections: []ConnectorSelection{{Name: "crm", ConnectionID: connectionIDs["alice"]}},
		}, db, sealer, slog.New(slog.NewTextHandler(io.Discard, nil)), nil)
		require.ErrorContains(attachErr, "does not belong to the verified caller")
	}

	_, _, _, err = attachConnectors(ctx, Spec{
		CustomerID:          customerID,
		ConfigID:            config.ID,
		Caller:              routing.Caller{UserID: "bob"},
		CallerKind:          auth.KindAuthenticated,
		ConnectorBindings:   config.Connectors,
		ConnectorSelections: []ConnectorSelection{{Name: "crm", ConnectionID: connectionIDs["alice"]}},
	}, db, sealer, slog.New(slog.NewTextHandler(io.Discard, nil)), nil)
	require.ErrorContains(err, "does not belong to the verified caller")
}

func TestSessionSelectsBetweenTwoAccountsAndDisconnectBlocksAnOpenRuntime(t *testing.T) {
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
	require.True(strings.HasSuffix(database, "_test"), "refusing to write connector sessions to %s", database)
	require.NoError(db.Migrate(ctx))

	primaryServer := newConnectorMCPTestServer(t, "Bearer primary-account-token", "primary account")
	secondaryServer := newConnectorMCPTestServer(t, "Bearer secondary-account-token", "secondary account")
	t.Cleanup(primaryServer.Close)
	t.Cleanup(secondaryServer.Close)
	customerID := "connector-multi-account-" + time.Now().UTC().Format("20060102150405.000000000")
	sealer, err := auth.NewSealer("connector-multi-account-test-key")
	require.NoError(err)
	connectionIDs := make(map[string]string, 2)
	for _, account := range []struct {
		alias    string
		endpoint string
		token    string
	}{
		{alias: "primary", endpoint: primaryServer.URL, token: "primary-account-token"},
		{alias: "secondary", endpoint: secondaryServer.URL, token: "secondary-account-token"},
	} {
		connection := store.ConnectorConnection{
			CustomerID:  customerID,
			ConnectorID: "linear",
			OwnerType:   "user",
			OwnerID:     "alice",
			Endpoint:    account.endpoint,
			AuthType:    connectors.AuthBearer,
			Label:       account.alias,
		}
		require.NoError(db.CreateConnectorConnection(ctx, &connection))
		connectionIDs[account.alias] = connection.ID
		connection.Revision++
		connection.Status = store.ConnectorConnected
		connection.CredentialKEKVersion = sealer.CurrentVersion()
		connection.CredentialSealed, err = connectors.SealCredentials(sealer, customerID, connection.ID, connection.Revision,
			connectors.Credentials{AuthType: connectors.AuthBearer, AccessToken: account.token})
		require.NoError(err)
		require.NoError(db.SaveConnectorConnectionAtRevision(ctx, &connection, connection.Revision-1))
	}
	t.Cleanup(func() {
		_, cleanupErr := db.DB().NewDelete().Model((*store.ConnectorConnection)(nil)).Where("customer_id = ?", customerID).Exec(ctx)
		require.NoError(cleanupErr)
	})

	tool := &mcpsdk.Tool{
		Name: "whoami", Description: "identify the connected account",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
	}
	digest, err := mcp.ToolSchemaDigest(tool)
	require.NoError(err)
	binding := func(alias string) store.ConnectorBinding {
		return store.ConnectorBinding{
			Name:        alias,
			ConnectorID: "linear",
			Connection:  store.ConnectionBinding{Type: "session"},
			Required:    true,
			Tools:       []store.ToolGrant{{Name: tool.Name, SchemaDigest: digest}},
		}
	}
	config := store.AgentConfig{
		CustomerID: customerID,
		Name:       "connector-multi-account",
		Connectors: []store.ConnectorBinding{binding("primary"), binding("secondary")},
	}
	require.NoError(db.CreateAgentConfig(ctx, &config))
	t.Cleanup(func() { require.NoError(db.DeleteAgentConfig(ctx, customerID, config.ID)) })
	config, err = db.AgentConfig(ctx, customerID, config.ID)
	require.NoError(err)
	runtime, tools, unavailable, err := attachConnectors(ctx, Spec{
		CustomerID:        customerID,
		ConfigID:          config.ID,
		Caller:            routing.Caller{UserID: "alice"},
		CallerKind:        auth.KindAuthenticated,
		ConnectorBindings: config.Connectors,
		ConnectorSelections: []ConnectorSelection{
			{Name: "primary", ConnectionID: connectionIDs["primary"]},
			{Name: "secondary", ConnectionID: connectionIDs["secondary"]},
		},
	}, db, sealer, slog.New(slog.NewTextHandler(io.Discard, nil)), &http.Client{Transport: http.DefaultTransport})
	require.NoError(err)
	require.NotNil(runtime)
	t.Cleanup(runtime.Close)
	require.Empty(unavailable)
	require.Len(tools, 2)
	require.ElementsMatch([]string{"primary__whoami", "secondary__whoami"}, []string{tools[0].Name, tools[1].Name})

	primary, err := runtime.Call(ctx, llm.ToolCall{Name: "primary__whoami", Arguments: "{}"})
	require.NoError(err)
	require.Equal("primary account", primary)
	require.Equal(int32(1), primaryServer.calls.Load())
	require.Zero(secondaryServer.calls.Load())

	config.Connectors = []store.ConnectorBinding{binding("primary"), binding("secondary")}
	config.Connectors[0].Tools = []store.ToolGrant{}
	require.NoError(db.UpdateAgentConfig(ctx, &config))
	_, err = runtime.Call(ctx, llm.ToolCall{Name: "primary__whoami", Arguments: "{}"})
	require.ErrorContains(err, "tool whoami is no longer granted")
	require.Equal(int32(1), primaryServer.calls.Load(), "a revoked grant must not reach the provider")

	config.Connectors = []store.ConnectorBinding{binding("primary"), binding("secondary")}
	require.NoError(db.UpdateAgentConfig(ctx, &config))
	primary, err = runtime.Call(ctx, llm.ToolCall{Name: "primary__whoami", Arguments: "{}"})
	require.NoError(err)
	require.Equal("primary account", primary)
	require.Equal(int32(2), primaryServer.calls.Load())
	_, err = db.DB().NewUpdate().Model((*store.ConnectorConnection)(nil)).
		Set("endpoint = ?", "https://changed.example/mcp").
		Where("customer_id = ?", customerID).
		Where("id = ?", connectionIDs["primary"]).
		Exec(ctx)
	require.NoError(err)
	_, err = runtime.Call(ctx, llm.ToolCall{Name: "primary__whoami", Arguments: "{}"})
	require.ErrorContains(err, "account endpoint changed while the session was open")
	require.Equal(int32(2), primaryServer.calls.Load(), "a changed endpoint must not receive the old connection's token")
	_, err = db.DB().NewUpdate().Model((*store.ConnectorConnection)(nil)).
		Set("endpoint = ?", primaryServer.URL).
		Where("customer_id = ?", customerID).
		Where("id = ?", connectionIDs["primary"]).
		Exec(ctx)
	require.NoError(err)

	require.NoError(db.DeleteConnectorConnection(ctx, customerID, connectionIDs["primary"]))
	_, err = runtime.Call(ctx, llm.ToolCall{Name: "primary__whoami", Arguments: "{}"})
	require.Error(err, "an open runtime must not dispatch after its selected account is disconnected")
	require.Equal(int32(2), primaryServer.calls.Load(), "the provider must not receive the post-disconnect tool call")
	secondary, err := runtime.Call(ctx, llm.ToolCall{Name: "secondary__whoami", Arguments: "{}"})
	require.NoError(err, "disconnecting one account must not make the runtime fall back to another")
	require.Equal("secondary account", secondary)
	require.Equal(int32(1), secondaryServer.calls.Load())
}

func TestRequiredConnectorNeedsAReadyAccountWhileOptionalConnectorIsOmitted(t *testing.T) {
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
	require.True(strings.HasSuffix(database, "_test"), "refusing to write connector readiness to %s", database)
	require.NoError(db.Migrate(ctx))

	customerID := "connector-readiness-" + time.Now().UTC().Format("20060102150405.000000000")
	connection := store.ConnectorConnection{
		CustomerID:  customerID,
		ConnectorID: "linear",
		OwnerType:   "app",
		Endpoint:    "https://mcp.linear.app/mcp",
		AuthType:    connectors.AuthOAuth2,
		Status:      store.ConnectorNeedsReauth,
	}
	require.NoError(db.CreateConnectorConnection(ctx, &connection))
	t.Cleanup(func() { require.NoError(db.DeleteConnectorConnection(ctx, customerID, connection.ID)) })
	connection.Status = store.ConnectorNeedsReauth
	require.NoError(db.SaveConnectorConnectionAtRevision(ctx, &connection, connection.Revision))
	binding := store.ConnectorBinding{
		Name: "linear", ConnectorID: "linear",
		Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: connection.ID},
	}
	config := store.AgentConfig{
		CustomerID: customerID, Name: "connector-readiness", Connectors: []store.ConnectorBinding{binding},
	}
	require.NoError(db.CreateAgentConfig(ctx, &config))
	t.Cleanup(func() { require.NoError(db.DeleteAgentConfig(ctx, customerID, config.ID)) })

	sealer, err := auth.NewSealer("connector-readiness-test-key")
	require.NoError(err)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	makeSpec := func(required bool) Spec {
		return Spec{
			CustomerID: customerID,
			ConfigID:   config.ID,
			ConnectorBindings: []store.ConnectorBinding{{
				Name: binding.Name, ConnectorID: binding.ConnectorID,
				Connection: binding.Connection, Required: required,
			}},
		}
	}

	_, _, _, err = attachConnectors(ctx, makeSpec(true), db, sealer, logger, nil)
	require.ErrorContains(err, "required connector linear needs reauthorization")

	runtime, tools, unavailable, err := attachConnectors(ctx, makeSpec(false), db, sealer, logger, nil)
	require.NoError(err)
	require.Nil(runtime)
	require.Empty(tools)
	require.Equal([]ConnectorUnavailable{{
		Name: "linear", ConnectorID: "linear", Reason: connectorNeedsReauthorization,
	}}, unavailable)
}

type connectorMCPTestServer struct {
	*httptest.Server
	calls atomic.Int32
}

func newConnectorMCPTestServer(t *testing.T, authorization, account string) *connectorMCPTestServer {
	t.Helper()
	fixture := &connectorMCPTestServer{}
	fixture.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != authorization {
			t.Errorf("MCP authorization = %q, want %q", got, authorization)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode MCP request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		id := request["id"]
		switch request["method"] {
		case "server/discover":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": id,
				"error": map[string]any{"code": -32601, "message": "method not found"},
			})
		case "initialize":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": id,
				"result": map[string]any{
					"protocolVersion": "2025-11-25",
					"capabilities":    map[string]any{"tools": map[string]any{}},
					"serverInfo":      map[string]any{"name": "connector-test", "version": "1"},
				},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": id,
				"result": map[string]any{"tools": []any{map[string]any{
					"name": "whoami", "description": "identify the connected account",
					"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
				}}},
			})
		case "tools/call":
			fixture.calls.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": id,
				"result": map[string]any{"content": []any{map[string]any{"type": "text", "text": account}}},
			})
		default:
			t.Errorf("unexpected MCP method %v", request["method"])
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	return fixture
}
