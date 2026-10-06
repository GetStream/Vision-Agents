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

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// UserPluginsSuite covers the tools of a plugin each end user connects themselves, and of a
// server named by URL that logs in, against Postgres and a provider that plays both the
// OAuth server and the MCP server.
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
	mux.HandleFunc("/open", s.serveMCP)
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

func (s *UserPluginsSuite) TestAUsersLoginIsMadeWithTheAgentsOwnClient() {
	secrets, err := auth.NewSealer("a passphrase")
	s.Require().NoError(err)
	owner := plugins.Owner{CustomerID: s.runner.customerID, ConfigID: s.runner.configID}
	sealed, err := SealPluginClientSecret(secrets, owner, "google_calendar", "acme-secret")
	s.Require().NoError(err)
	s.Require().NoError(s.store.SavePluginClient(context.Background(), &store.PluginClient{
		CustomerID: owner.CustomerID, ConfigID: owner.ConfigID, PluginID: "google_calendar",
		ClientID: "acme-client", SecretSealed: sealed, SecretKEKVersion: secrets.CurrentVersion(),
	}))
	s.runner.auth.Clients = PluginClients(s.store, secrets)

	result := s.run("google_calendar__list_tools", "")

	s.asksToConnect(result)
	pending, err := s.store.UserPluginConnection(context.Background(), owner.CustomerID, owner.ConfigID, "alice", "google_calendar")
	s.Require().NoError(err)
	s.Equal("acme-client", pending.ClientID, "the agent's client, not one registered on the fly")
	client, found, err := s.runner.auth.Clients(context.Background(), owner, "google_calendar")
	s.Require().NoError(err)
	s.True(found)
	s.Equal(plugins.Client{ID: "acme-client", Secret: "acme-secret"}, client)
}

func (s *UserPluginsSuite) TestAUserIsToldAPluginTheAgentHasNoClientForIsNotAvailable() {
	// A provider such as Google registers no client on the fly.
	google := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"authorization_endpoint": "https://accounts.example/auth",
			"token_endpoint":         "https://accounts.example/token",
		})
	}))
	defer google.Close()
	s.runner.auth.HTTP = google.Client()
	s.calendar.URL = google.URL + "/mcp"
	s.runner.named["google_calendar"] = s.calendar

	result := s.run("google_calendar__list_tools", "")

	s.JSONEq(plugins.UnavailableResult(s.calendar), result)
	_, ok := plugins.RequestedAuthorization("google_calendar__list_tools", result,
		Logins(Spec{UserPlugins: []store.PluginEntry{{Name: "google_calendar"}}}))
	s.False(ok, "there is nothing for the user to press")
}

func (s *UserPluginsSuite) TestTheAppsLoginToAPluginEachUserConnectsIsNotHandedToEverySession() {
	s.login("", s.calendar, "", "good-token")
	spec := Spec{
		CustomerID:  s.runner.customerID,
		ConfigID:    s.runner.configID,
		UserPlugins: []store.PluginEntry{{Name: "google_calendar"}},
	}

	runtime, tools, unconnected := attachPlugins(context.Background(), spec, s.store, &plugins.Auth{HTTP: s.provider.Client()}, slog.New(slog.DiscardHandler))

	s.Nil(runtime)
	s.Empty(tools)
	s.Empty(unconnected)
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

func (s *UserPluginsSuite) TestAServerNamedByURLAsksEachUserToLogInThere() {
	notes := s.server(store.MCPServer{User: true})
	s.runner.named["notes"] = notes

	result := s.run("notes__list_tools", "")

	found, ok := plugins.RequestedAuthorization("notes__list_tools", result, Logins(Spec{MCPServers: []store.MCPServer{{Name: "notes", User: true}}}))
	s.Require().True(ok, result)
	s.Equal("Connect notes", found.Title)
	s.Empty(found.ThumbURL, "the server has no logo of ours to show")
	pending, err := s.store.UserPluginConnection(context.Background(), s.runner.customerID, s.runner.configID, "alice", "notes")
	s.Require().NoError(err)
	s.Equal(notes.URL, pending.InstanceURL, "the login is only good where it was made")
}

func (s *UserPluginsSuite) TestAUserLoggedIntoAServerNamedByURLReachesItWithTheirToken() {
	notes := s.server(store.MCPServer{User: true})
	s.runner.named["notes"] = notes
	s.login("alice", notes, notes.URL, "good-token")

	listed := s.run("notes__list_tools", "")
	answered := s.run("notes__call_tool", `{"tool":"list_events","arguments":{}}`)

	s.JSONEq(`{"tools":[{"name":"list_events","description":"Events today","input_schema":{"type":"object"}}]}`, listed)
	s.Equal("standup at 10", answered)
}

func (s *UserPluginsSuite) TestALoginMadeWhereAServerUsedToBeIsAskedForAgain() {
	notes := s.server(store.MCPServer{User: true})
	s.runner.named["notes"] = notes
	s.login("alice", notes, "https://somewhere-else.example/mcp", "good-token")

	result := s.run("notes__list_tools", "")

	_, ok := plugins.RequestedAuthorization("notes__list_tools", result, Logins(Spec{MCPServers: []store.MCPServer{{Name: "notes", User: true}}}))
	s.True(ok, "the token is never sent to a server it was not granted for")
}

func (s *UserPluginsSuite) TestAServerThatNeedsALoginIsOpenedWithTheAppsTokenWithoutScopes() {
	notes := s.server(store.MCPServer{})
	s.login("", notes, notes.URL, "good-token")

	runtime, tools, unconnected := s.attach(store.MCPServer{Name: notes.ID, URL: notes.URL})
	s.Require().NotNil(runtime, "the server's 401 says it needs a login, which the app made")
	defer runtime.Close()
	answered, err := runtime.Call(context.Background(), llm.ToolCall{ID: uuid.NewString(), Name: "notes__list_events", Arguments: `{}`})

	s.Equal([]string{"notes__list_events"}, toolNames(tools))
	s.Empty(unconnected)
	s.Require().NoError(err)
	s.Equal("standup at 10", answered)
}

func (s *UserPluginsSuite) TestAServerTheAppLoggedIntoOffersOnlyTheToolsTheAgentAllows() {
	notes := s.server(store.MCPServer{})
	s.login("", notes, notes.URL, "good-token")

	runtime, tools, _ := s.attach(store.MCPServer{Name: notes.ID, URL: notes.URL, NeedsLogin: &needsLogin, Tools: []string{"free_busy"}})
	s.Require().NotNil(runtime, "the server opened with the login")
	defer runtime.Close()

	s.Empty(tools)
	s.False(runtime.Owns("notes__list_events"))
}

func (s *UserPluginsSuite) TestAServerThatNeedsNoLoginIsOpenedWithNone() {
	runtime, tools, unconnected := s.attach(store.MCPServer{Name: "menus", URL: s.provider.URL + "/open"})
	s.Require().NotNil(runtime)
	defer runtime.Close()
	answered, err := runtime.Call(context.Background(), llm.ToolCall{ID: uuid.NewString(), Name: "menus__list_events", Arguments: `{}`})

	s.Equal([]string{"menus__list_events"}, toolNames(tools))
	s.Empty(unconnected)
	s.Require().NoError(err)
	s.Equal("standup at 10", answered)
}

func (s *UserPluginsSuite) TestAServerTheAppHasNotLoggedIntoFailsWithHowToConnectIt() {
	notes := s.server(store.MCPServer{})
	spec := Spec{CustomerID: s.runner.customerID, ConfigID: s.runner.configID, MCPServers: []store.MCPServer{{Name: notes.ID, URL: notes.URL}}}

	runtime, tools, unconnected := attachPlugins(context.Background(), spec, s.store, &plugins.Auth{HTTP: s.provider.Client()}, slog.New(slog.DiscardHandler))
	_, err := (&pluginRunner{mcp: runtime, unconnected: unconnected}).Run(context.Background(),
		llm.ToolCall{ID: uuid.NewString(), Name: "notes__list_tools", Arguments: `{}`})

	s.Nil(runtime)
	s.Empty(tools)
	s.Equal([]string{"notes__list_tools"}, toolNames(unconnectedTools(unconnected)))
	s.EqualError(err, "notes is not connected: connect notes on the dashboard")
}

func (s *UserPluginsSuite) TestOnlyTheServersEachUserLogsIntoMayAskForALoginInTheChat() {
	spec := Spec{
		AgentPlugins: []store.PluginEntry{{Name: "sentry"}},
		UserPlugins:  []store.PluginEntry{{Name: "google_calendar"}},
		MCPServers: []store.MCPServer{
			{Name: "notes", URL: "https://notes.example.com/mcp", User: true},
			{Name: "crm", URL: "https://crm.example.com/mcp", NeedsLogin: &needsLogin},
			{Name: "tablejourney", URL: "https://tablejourney.com/mcp"},
		},
	}
	sentry, ok := plugins.Lookup("sentry")
	s.Require().True(ok)

	s.Equal([]string{"google_calendar", "notes"}, Logins(spec))
	for tool, card := range map[string]string{
		"sentry__list_issues":     plugins.AuthorizationResult(sentry, "https://sentry.io/oauth/authorize", ""),
		"crm__call_tool":          plugins.AuthorizationResult(ServerPlugin(spec.MCPServers[1]), "https://crm.example.com/authorize", ""),
		"tablejourney__call_tool": plugins.AuthorizationResult(ServerPlugin(spec.MCPServers[2]), "https://tablejourney.com/authorize", ""),
	} {
		_, ok := plugins.RequestedAuthorization(tool, card, Logins(spec))
		s.False(ok, "%s: the app logs in on the dashboard, so no card of its comes through the chat", tool)
	}
}

// needsLogin is what a server that said at save it requires a login has stored.
var needsLogin = true

// server is the provider's MCP server named by URL as notes, logging in as server says.
func (s *UserPluginsSuite) server(server store.MCPServer) plugins.Plugin {
	server.Name, server.URL = "notes", s.provider.URL+"/mcp"
	return ServerPlugin(server)
}

// attach opens the config's MCP servers as a session starting would.
func (s *UserPluginsSuite) attach(server store.MCPServer) (*plugins.Runtime, []harness.Tool, []string) {
	spec := Spec{CustomerID: s.runner.customerID, ConfigID: s.runner.configID, MCPServers: []store.MCPServer{server}}
	return attachPlugins(context.Background(), spec, s.store, &plugins.Auth{HTTP: s.provider.Client()}, slog.New(slog.DiscardHandler))
}

// login stores a login to plugin made at endpoint, the app's for an empty user.
func (s *UserPluginsSuite) login(userID string, plugin plugins.Plugin, endpoint, token string) {
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &store.PluginConnection{
		CustomerID: s.runner.customerID, ConfigID: s.runner.configID, PluginID: plugin.ID,
		UserID: userID, InstanceURL: endpoint, Status: store.PluginConnected, AccessToken: token,
	}))
}

func toolNames(tools []harness.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	return names
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
	found, ok := plugins.RequestedAuthorization("google_calendar__list_tools", result,
		Logins(Spec{UserPlugins: []store.PluginEntry{{Name: "google_calendar"}}}))
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
	if r.URL.Path == "/mcp" && r.Header.Get("Authorization") != "Bearer "+s.accepted.Load().(string) {
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
