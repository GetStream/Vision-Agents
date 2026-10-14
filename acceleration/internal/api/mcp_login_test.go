//go:build integration

package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

const (
	crmURL   = "https://crm.example.com/mcp"
	notesURL = "https://notes.example.com/mcp"
	// plainURL is a server that answers and advertises no login.
	plainURL = "https://plain.example.com/open"
)

// MCPLoginSuite covers the MCP servers a config names by URL that log in: what is saved,
// and the app's login, made from the dashboard as a plugin's is. One server stands in for
// every host, playing the MCP server, its protected-resource metadata and its
// authorization server.
type MCPLoginSuite struct {
	RouterSuite

	// logged is what the router logged, for the tests of the deprecation it warns of.
	logged *lockedLog
}

func TestMCPLoginSuite(t *testing.T) {
	runSuite(t, new(MCPLoginSuite))
}

func (s *MCPLoginSuite) SetupSuite() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"resource":              "https://" + r.Host + "/mcp",
			"authorization_servers": []string{"https://" + r.Host + "/as"},
			"scopes_supported":      []string{"contacts.read", "contacts.write"},
		})
	})
	mux.HandleFunc("GET /as/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"authorization_endpoint": "https://" + r.Host + "/as/authorize",
			"token_endpoint":         "https://" + r.Host + "/as/token",
			"registration_endpoint":  "https://" + r.Host + "/as/register",
		})
	})
	mux.HandleFunc("POST /as/register", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"client_id": "registered-" + r.Host})
	})
	mux.HandleFunc("POST /as/token", func(w http.ResponseWriter, r *http.Request) {
		if r.FormValue("code") != "the-code" || r.FormValue("code_verifier") == "" {
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "crm-token", "expires_in": 3600})
	})
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "log in", http.StatusUnauthorized)
	})
	s.pluginMCP = httptest.NewTLSServer(mux)
	s.T().Cleanup(s.pluginMCP.Close)
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *MCPLoginSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *MCPLoginSuite) TestAServerThatLogsInIsSavedWithItsScopesAndWhoLogsIn() {
	needs := pointerTo(true)
	created := s.create([]map[string]any{
		{"name": "crm", "url": crmURL, "scopes": []string{"contacts.read"}, "tools": []string{"search_*"}},
		{"name": "notes", "url": notesURL, "user": true},
	})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"mcp_servers": []map[string]any{
			{"name": "crm", "url": crmURL, "scopes": []string{"contacts.read", "contacts.write"}},
			{"name": "notes", "url": notesURL, "user": true, "scopes": []string{"contacts.read"}},
		}}, &patched))

	s.Equal([]McpServer{
		{Name: "crm", Url: crmURL, Scopes: pointerTo([]string{"contacts.read"}), Tools: pointerTo([]string{"search_*"}), NeedsLogin: needs},
		{Name: "notes", Url: notesURL, User: pointerTo(true), NeedsLogin: needs},
	}, value(created.McpServers))
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal([]McpServer{
		{Name: "crm", Url: crmURL, Scopes: pointerTo([]string{"contacts.read", "contacts.write"}), NeedsLogin: needs},
		{Name: "notes", Url: notesURL, Scopes: pointerTo([]string{"contacts.read"}), User: pointerTo(true), NeedsLogin: needs},
	}, value(read.McpServers))
}

func (s *MCPLoginSuite) TestMoreScopesThanALoginAsksForAreRefused() {
	scopes := make([]string, 33)
	for i := range scopes {
		scopes[i] = fmt.Sprintf("scope.%d", i)
	}

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "crm-" + s.utils.uuid(), "mcp_servers": []map[string]any{{"name": "crm", "url": crmURL, "scopes": scopes}},
	})

	s.Equal(http.StatusBadRequest, status)
}

func (s *MCPLoginSuite) TestAServerThatAdvertisesNoLoginCannotBeSavedAsOneThatLogsIn() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "plain-" + s.utils.uuid(), "mcp_servers": []map[string]any{{"name": "plain", "url": plainURL, "user": true}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "plain sets scopes or user, but advertises no OAuth login")
}

func (s *MCPLoginSuite) TestAServerThatNeedsNoLoginIsNotAskedForOne() {
	created := s.create([]map[string]any{{"name": "plain", "url": plainURL}})

	s.Equal([]McpServer{{Name: "plain", Url: plainURL, NeedsLogin: pointerTo(false)}}, value(created.McpServers))
	s.Empty(s.logins(created.Id), "the app has nothing to log into")
}

func (s *MCPLoginSuite) TestAServerThatNeedsALoginIsTheAppsToMakeWithoutScopes() {
	created := s.create([]map[string]any{{"name": "crm", "url": crmURL}})
	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusNotConnected}},
		s.logins(created.Id), "the server said it needs a login, so the dashboard offers to connect it")

	authorize := s.authorize(created.Id, "crm")
	s.Equal("contacts.read contacts.write", authorize.Query().Get("scope"), "what the server advertises")
	s.Equal(http.StatusFound, s.callback(authorize.Query().Get("state")))

	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusConnected}}, s.logins(created.Id))
}

func (s *MCPLoginSuite) TestTheAppLogsIntoAServerNamedByURLAsItDoesIntoAPlugin() {
	created := s.create([]map[string]any{
		{"name": "crm", "url": crmURL, "scopes": []string{"contacts.read"}},
		{"name": "notes", "url": notesURL, "user": true},
	})
	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusNotConnected}},
		s.logins(created.Id), "each user logs into notes, so the app has only crm to finish")

	authorize := s.authorize(created.Id, "crm")
	s.Equal("crm.example.com", authorize.Host)
	s.Equal("/as/authorize", authorize.Path)
	s.Equal("registered-crm.example.com", authorize.Query().Get("client_id"))
	s.Equal("contacts.read", authorize.Query().Get("scope"))
	s.Equal(crmURL, authorize.Query().Get("resource"))
	s.Equal("S256", authorize.Query().Get("code_challenge_method"))
	s.Equal(http.StatusFound, s.callback(authorize.Query().Get("state")))

	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusConnected}}, s.logins(created.Id))
	login, err := s.store.PluginLogins(s.T().Context(), s.customerID(), created.Id, "crm")
	s.Require().NoError(err)
	s.Require().Len(login, 1)
	s.Equal("crm-token", login[0].AccessToken)
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Nil(read.Plugins, "the config names the server under mcp_servers already")
}

func (s *MCPLoginSuite) TestTheAppsLoginToAServerNamedByURLCanBeDropped() {
	created := s.create([]map[string]any{{"name": "crm", "url": crmURL, "scopes": []string{"contacts.read"}}})
	s.Require().Equal(http.StatusFound, s.callback(s.authorize(created.Id, "crm").Query().Get("state")))

	status, _ := s.serverClient.call(http.MethodDelete, "/v1/agents/configs/"+created.Id+"/plugins/crm", nil)

	s.Equal(http.StatusNoContent, status)
	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusNotConnected}}, s.logins(created.Id))
}

func (s *MCPLoginSuite) TestTheAppsLoginWarnsOfTheDeprecationWhereItStartsAndFinishes() {
	created := s.create([]map[string]any{{"name": "crm", "url": crmURL}})

	authorize := s.authorize(created.Id, "crm")
	s.Equal(http.StatusFound, s.callback(authorize.Query().Get("state")))

	started := deprecations(s.logged, plugins.PathLogin, created.Id)
	s.Require().Len(started, 1)
	s.Contains(started[0], "customer="+s.customerID()+" config="+created.Id+" plugin=crm via=mcp_servers")
	callback := deprecations(s.logged, plugins.PathCallback, created.Id)
	s.Require().Len(callback, 1)
	s.Contains(callback[0], "plugin=crm via=mcp_servers")
	s.NotContains(s.logged.String(), "crm-token")
	s.Equal([]PluginConnection{{PluginId: "crm", Name: "crm", Status: PluginConnectionStatusConnected}}, s.logins(created.Id))
}

func (s *MCPLoginSuite) TestDroppingTheAppsLoginToAServerWarnsOfTheDeprecationAsAnMCPServer() {
	created := s.create([]map[string]any{{"name": "crm", "url": crmURL}})
	s.Require().Equal(http.StatusFound, s.callback(s.authorize(created.Id, "crm").Query().Get("state")))

	status, _ := s.serverClient.call(http.MethodDelete, "/v1/agents/configs/"+created.Id+"/plugins/crm", nil)

	s.Equal(http.StatusNoContent, status)
	lines := deprecations(s.logged, plugins.PathDisconnect, created.Id)
	s.Require().Len(lines, 1)
	s.Contains(lines[0], "customer="+s.customerID()+" config="+created.Id+" plugin=crm via=mcp_servers")
}

func (s *MCPLoginSuite) TestALoginThatIsRefusedWarnsOfNothing() {
	created := s.create([]map[string]any{{"name": "plain", "url": plainURL}})

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs/"+created.Id+"/plugins/plain/authorize", nil)

	s.Equal(http.StatusNotFound, status)
	s.Empty(deprecations(s.logged, plugins.PathLogin, created.Id))
}

func (s *MCPLoginSuite) TestAServerEachUserLogsIntoIsNotConnectedForTheApp() {
	created := s.create([]map[string]any{{"name": "notes", "url": notesURL, "user": true}})

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+created.Id+"/plugins/notes/authorize", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "each end user")
}

func (s *MCPLoginSuite) TestAServerThatNeedsNoLoginCannotBeLoggedInto() {
	created := s.create([]map[string]any{{"name": "plain", "url": plainURL}})

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+created.Id+"/plugins/plain/authorize", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownPlugin.Message)
}

// create saves a config naming servers.
func (s *MCPLoginSuite) create(servers []map[string]any) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "crm-" + s.utils.uuid(), "mcp_servers": servers}, &created))
	return created
}

// logins are the config's logins as the dashboard lists them.
func (s *MCPLoginSuite) logins(configID string) []PluginConnection {
	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+configID+"/plugins", nil, &connections))
	return connections
}

// authorize starts the app's login to a server and returns the URL the browser would open.
func (s *MCPLoginSuite) authorize(configID, server string) *url.URL {
	var started PluginAuthorization
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/configs/"+configID+"/plugins/"+server+"/authorize", nil, &started))
	authorize, err := url.Parse(started.AuthorizeUrl)
	s.Require().NoError(err)
	return authorize
}

// callback is the browser arriving back from the authorization server, without following
// the redirect to the dashboard.
func (s *MCPLoginSuite) callback(state string) int {
	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := browser.Get(s.server.URL + plugins.CallbackPath + "?state=" + url.QueryEscape(state) + "&code=the-code")
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}
