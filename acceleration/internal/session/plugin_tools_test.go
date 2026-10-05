//go:build integration

package session

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// UserPluginsSuite covers the tools of a plugin each end user connects themselves, against
// Postgres and a provider that plays both the OAuth server and the MCP server.
type UserPluginsSuite struct {
	suite.Suite
	store    *store.Store
	provider *httptest.Server
	// accepted is the one token the provider's MCP server takes.
	accepted atomic.Value
	runner   *userPluginRunner
	calendar plugins.Plugin
}

func TestUserPluginsSuite(t *testing.T) {
	suite.Run(t, new(UserPluginsSuite))
}

func (s *UserPluginsSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(context.Background()))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"authorization_endpoint": "https://accounts.example/auth",
			"token_endpoint":         "https://accounts.example/token",
			"registration_endpoint":  "http://" + r.Host + "/register",
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"client_id": "registered"})
	})
	mux.HandleFunc("/mcp", s.serveMCP)
	s.provider = httptest.NewServer(mux)
	s.T().Cleanup(s.provider.Close)
}

func (s *UserPluginsSuite) SetupTest() {
	s.accepted.Store("good-token")
	s.calendar = plugins.Plugin{ID: "google_calendar", Name: "Google Calendar", URL: s.provider.URL + "/mcp"}
	s.runner = &userPluginRunner{
		customerID: "customer-" + uuid.NewString(),
		configID:   uuid.NewString(),
		userID:     "alice",
		named:      map[string]plugins.Plugin{"google_calendar": s.calendar},
		db:         s.store,
		auth:       &plugins.Auth{HTTP: s.provider.Client(), PublicURL: "https://router.example"},
		logger:     slog.New(slog.DiscardHandler),
		open:       map[string]*plugins.Runtime{},
	}
	s.T().Cleanup(s.runner.Close)
}

func (s *UserPluginsSuite) TestAUserWhoNeverConnectedIsAskedTo() {
	result := s.run("google_calendar__list_tools", "")

	s.asksToConnect(result)
}

func (s *UserPluginsSuite) TestAConnectedUserReachesTheirAccount() {
	s.connect("good-token")

	listed := s.run("google_calendar__list_tools", "")
	answered := s.run("google_calendar__call_tool", `{"tool":"list_events","arguments":{}}`)

	s.JSONEq(`{"tools":[{"name":"list_events","description":"Events today","input_schema":{"type":"object"}}]}`, listed)
	s.Equal("standup at 10", answered)
}

func (s *UserPluginsSuite) TestAToolTheAgentLeftOutIsNeitherListedNorRun() {
	s.calendar.Tools = []string{"free_busy"}
	s.runner.named["google_calendar"] = s.calendar
	s.connect("good-token")

	listed := s.run("google_calendar__list_tools", "")
	_, err := s.runner.Run(context.Background(), llm.ToolCall{
		ID: uuid.NewString(), Name: "google_calendar__call_tool", Arguments: `{"tool":"list_events","arguments":{}}`,
	})

	s.JSONEq(`{"tools":[]}`, listed)
	s.Error(err)
}

func (s *UserPluginsSuite) TestALoginTheProviderRefusesIsAskedForAgain() {
	s.connect("revoked-token")

	result := s.run("google_calendar__list_tools", "")

	s.asksToConnect(result)
	s.forgotten()
}

func (s *UserPluginsSuite) TestALoginRevokedMidConversationIsAskedForAgain() {
	s.connect("good-token")
	s.run("google_calendar__list_tools", "")
	s.accepted.Store("a-token-alice-does-not-have")

	result := s.run("google_calendar__call_tool", `{"tool":"list_events","arguments":{}}`)

	s.asksToConnect(result)
	s.forgotten()
	s.Empty(s.runner.open, "the session that held the refused login is closed")
}

// run calls one of the user plugin's tools the way the model would.
func (s *UserPluginsSuite) run(name, arguments string) string {
	parts, err := s.runner.Run(context.Background(), llm.ToolCall{ID: uuid.NewString(), Name: name, Arguments: arguments})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	return parts[0].Text
}

// connect stores Alice's login as the callback would have.
func (s *UserPluginsSuite) connect(token string) {
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &store.PluginConnection{
		CustomerID: s.runner.customerID, ConfigID: s.runner.configID, PluginID: "google_calendar",
		UserID: "alice", Status: store.PluginConnected, AccessToken: token,
	}))
}

func (s *UserPluginsSuite) asksToConnect(result string) {
	found, ok := plugins.RequestedAuthorization("google_calendar__list_tools", result)
	s.Require().True(ok, result)
	s.True(strings.HasPrefix(found.AuthorizeURL, "https://accounts.example/auth?"), found.AuthorizeURL)
	s.Equal("Connect Google Calendar", found.Title)
}

// forgotten checks the refused login was replaced by a fresh attempt holding no token.
func (s *UserPluginsSuite) forgotten() {
	conn, err := s.store.UserPluginConnection(context.Background(),
		s.runner.customerID, s.runner.configID, "alice", "google_calendar")
	s.Require().NoError(err)
	s.Equal(store.PluginPending, conn.Status)
	s.Empty(conn.AccessToken)
	s.NotEmpty(conn.OAuthState)
}

func (s *UserPluginsSuite) serveMCP(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer "+s.accepted.Load().(string) {
		w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
		http.Error(w, `{"error":"invalid_token"}`, http.StatusUnauthorized)
		return
	}
	var body struct {
		ID     int    `json:"id"`
		Method string `json:"method"`
	}
	s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
	var result any
	switch body.Method {
	case "notifications/initialized":
		w.WriteHeader(http.StatusAccepted)
		return
	case "initialize":
		result = map[string]any{"protocolVersion": "2025-03-26"}
	case "tools/list":
		result = map[string]any{"tools": []map[string]any{{
			"name": "list_events", "description": "Events today", "inputSchema": map[string]any{"type": "object"},
		}}}
	case "tools/call":
		result = map[string]any{"content": []map[string]any{{"type": "text", "text": "standup at 10"}}}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": body.ID, "result": result})
}
