//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PluginsSuite covers the hosted MCP catalog and the logins an agent holds against it.
//
// Starting a login discovers the provider's OAuth endpoints over the network, so what is
// asserted about authorizing is everything the router decides before it goes out.
type PluginsSuite struct {
	RouterSuite
}

func TestPluginsSuite(t *testing.T) {
	runSuite(t, new(PluginsSuite))
}

// SetupSuite lets the API reach the token servers the tests stand up on loopback, which
// the router's own client refuses.
func (s *PluginsSuite) SetupSuite() {
	s.pluginHTTP = &http.Client{}
	s.RouterSuite.SetupSuite()
}

func (s *PluginsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *PluginsSuite) TestTheCatalogIsTheOneBuiltIn() {
	// A built-in catalog rather than the customer's own rows, so a fresh app has all of it.
	listed := s.catalog("")

	s.Require().NotEmpty(listed)
	s.Contains(named(listed), "slack")
}

func (s *PluginsSuite) TestTheCatalogIsFilteredByWhatItIsFor() {
	scheduling := s.catalog("Scheduling")

	s.Require().NotEmpty(scheduling)
	for _, plugin := range scheduling {
		s.Equal("Scheduling", plugin.Category)
	}
	s.NotContains(named(scheduling), "slack")
}

func (s *PluginsSuite) TestEveryCatalogPluginHasALogoAnythingCanDraw() {
	// Whatever renders the card, a chat client or a browser, has no credential of ours to
	// put on an <img>, so the logo has to be served to a caller that sends none.
	for _, plugin := range s.catalog("") {
		s.Require().NotEmpty(plugin.LogoUrl, plugin.Id)

		status, body := s.unauthenticatedClient.call(http.MethodGet, plugins.LogoPath(plugin.Id), nil)

		s.Equal(http.StatusOK, status, plugin.Id)
		s.Contains(string(body), "<svg", plugin.Id)
	}
}

func (s *PluginsSuite) TestAPluginNobodyHasHasNoLogo() {
	status, _ := s.unauthenticatedClient.call(http.MethodGet, plugins.LogoPath("carrier-pigeon"), nil)

	s.Equal(http.StatusNotFound, status)
}

func (s *PluginsSuite) TestAPluginOnAHostOfItsOwnSaysSoInTheCatalog() {
	shopify := s.catalog("shopify")

	s.Require().Len(shopify, 1)
	s.Require().NotNil(shopify[0].InstanceRequired)
	s.True(*shopify[0].InstanceRequired, "a shop has no single global host")
	s.Require().NotNil(shopify[0].InstanceHint)
	s.NotEmpty(*shopify[0].InstanceHint)
}

func (s *PluginsSuite) TestAPluginThatNeedsTheAppsOwnClientSaysWhereItRedirects() {
	calendar := s.catalog("google calendar")

	s.Require().Len(calendar, 1)
	s.Require().NotNil(calendar[0].ClientRequired)
	s.True(*calendar[0].ClientRequired)
	s.Require().NotNil(calendar[0].RedirectUri)
	s.True(strings.HasSuffix(*calendar[0].RedirectUri, plugins.CallbackPath), *calendar[0].RedirectUri)
	s.Nil(s.catalog("linear")[0].RedirectUri, "Linear registers a client on the fly")
}

func (s *PluginsSuite) TestSlackListsHowToCreateItsClient() {
	slack := s.catalog("slack")

	s.Require().Len(slack, 1)
	s.Require().NotNil(slack[0].SetupUrl)
	s.Equal("https://api.slack.com/apps", *slack[0].SetupUrl)
	s.Require().NotNil(slack[0].SetupSteps)
	s.Equal("Create a Slack app", (*slack[0].SetupSteps)[0].Title)
	s.Nil(s.catalog("linear")[0].SetupSteps, "Linear needs no setup")
}

func (s *PluginsSuite) TestAnAgentHoldsNoLoginsUntilOneIsMade() {
	agent := s.data.createAgentConfig()

	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+agent.Id+"/plugins", nil, &connections))

	s.Empty(connections, "the rest of the catalog is implied absent")
}

func (s *PluginsSuite) TestAPluginTheAgentNamesThatNobodyConnectedIsLeftToRemindAbout() {
	var agent AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{
			Name:         "on-call-" + s.utils.uuid(),
			AgentPlugins: pointerTo([]PluginEntry{{Name: "sentry"}}),
			UserPlugins:  pointerTo([]PluginEntry{{Name: "google_calendar"}}),
		}, &agent))

	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+agent.Id+"/plugins", nil, &connections))

	s.Require().Len(connections, 2)
	s.Equal("sentry", connections[0].PluginId)
	s.Equal(PluginConnectionStatusNotConnected, connections[0].Status)
	s.Nil(connections[0].User)
	s.Equal("google_calendar", connections[1].PluginId)
	s.Require().NotNil(connections[1].User)
	s.True(*connections[1].User, "each user connects their own calendar, so the app has nothing to finish for it")
	s.Require().NotNil(connections[1].ClientRequired)
	s.True(*connections[1].ClientRequired, "Google registers no client on the fly")
	s.Nil(connections[1].Client)
	s.Equal([]PluginEntry{{Name: "google_calendar"}}, *agent.UserPlugins)
}

func (s *PluginsSuite) TestAnAgentsClientSecretIsSealedAndNeverReturned() {
	agent := s.data.createAgentConfig()

	var set PluginClient
	status := s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme.apps.googleusercontent.com", ClientSecret: pointerTo("acme-secret"), User: pointerTo(true)}, &set)

	s.Require().Equal(http.StatusOK, status)
	s.Equal(PluginClient{ClientId: "acme.apps.googleusercontent.com", HasSecret: true}, set)
	stored, err := s.store.PluginClient(s.T().Context(), s.customerID(), agent.Id, "google_calendar")
	s.Require().NoError(err)
	s.NotContains(string(stored.SecretSealed), "acme-secret")
	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+agent.Id+"/plugins", nil, &connections))
	s.Require().Len(connections, 1, "user names the plugin for each end user to connect")
	s.True(*connections[0].User)
	s.Equal(&set, connections[0].Client)
}

func (s *PluginsSuite) TestAnEndUsersLoginIsFinishedWithTheAgentsClient() {
	agent := s.data.createAgentConfig()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme-client", ClientSecret: pointerTo("acme-secret"), User: pointerTo(true)}, nil))
	tokens := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("acme-client", r.FormValue("client_id"))
		s.Equal("acme-secret", r.FormValue("client_secret"), "Google authenticates the client at the token endpoint")
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "alices-token", "expires_in": 3600})
	}))
	defer tokens.Close()
	state := s.utils.uuid()
	s.Require().NoError(s.store.UpsertPluginConnection(s.T().Context(), &store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: agent.Id, PluginID: "google_calendar", UserID: "alice",
		OAuthState: state, CodeVerifier: "verifier", ClientID: "acme-client", TokenEndpoint: tokens.URL,
	}))

	status, _ := s.unauthenticatedClient.call(http.MethodGet,
		plugins.CallbackPath+"?state="+state+"&code=the-code", nil)

	s.Equal(http.StatusOK, status)
	alices, err := s.store.UserPluginConnection(s.T().Context(), s.customerID(), agent.Id, "alice", "google_calendar")
	s.Require().NoError(err)
	s.Equal("alices-token", alices.AccessToken)
}

func (s *PluginsSuite) TestAPluginEachEndUserConnectsIsNotConnectedByTheApp() {
	var agent AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{
			Name:        "assistant-" + s.utils.uuid(),
			UserPlugins: pointerTo([]PluginEntry{{Name: "linear"}}),
		}, &agent))

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+agent.Id+"/plugins/linear/authorize", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "connected by each end user")
}

func (s *PluginsSuite) TestRemovingAPluginEachEndUserConnectsDropsItsClient() {
	agent := s.data.createAgentConfig()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)}, nil))

	status, _ := s.serverClient.call(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar", nil)

	s.Equal(http.StatusNoContent, status)
	_, err := s.store.PluginClient(s.T().Context(), s.customerID(), agent.Id, "google_calendar")
	s.ErrorIs(err, store.ErrUnknownPluginClient)
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+agent.Id, nil, &stored))
	s.Nil(stored.UserPlugins)
}

func (s *PluginsSuite) TestASyncNamingAPluginNobodyCanConnectYetIsStoredWithAWarning() {
	name := "assistant-" + s.utils.uuid()
	sync := map[string]any{"name": name, "hash": "v1", "mode": "text", "user_plugins": []string{"linear", "google_calendar"}}

	var synced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", sync, &synced))

	s.Require().Len(synced.Warnings, 1, "linear registers a client on the fly")
	s.Contains(synced.Warnings[0], "Google Calendar needs an OAuth client")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+synced.Config.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme-client"}, nil))
	sync["hash"] = "v2"
	var resynced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", sync, &resynced))
	s.False(resynced.Unchanged)
	s.Empty(resynced.Warnings)
}

func (s *PluginsSuite) TestAClientIsOnlyForAPluginWithAnOAuthLogin() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodPut,
		"/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon/client", SetPluginClientRequest{ClientId: "x"})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownPlugin.Message)
}

func (s *PluginsSuite) TestAnEndUsersDeviceMayNotSetAClient() {
	agent := s.data.createAgentConfig()

	status, _ := s.client.call(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "x"})

	s.Equal(http.StatusForbidden, status)
}

func (s *PluginsSuite) TestAnEndUsersLoginIsTheirsAloneAndSendsThemBackToTheConversation() {
	agent := s.data.createAgentConfig()
	tokens := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("the-code", r.FormValue("code"))
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "alices-token", "expires_in": 3600})
	}))
	defer tokens.Close()
	ctx := s.T().Context()
	state := s.utils.uuid()
	// The app's own pending login of the same plugin on the same agent is a different row.
	s.Require().NoError(s.store.UpsertPluginConnection(ctx, &store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: agent.Id, PluginID: "google_calendar",
		OAuthState: s.utils.uuid(), TokenEndpoint: tokens.URL,
	}))
	s.Require().NoError(s.store.UpsertPluginConnection(ctx, &store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: agent.Id, PluginID: "google_calendar", UserID: "alice",
		OAuthState: state, CodeVerifier: "verifier", ClientID: "client", TokenEndpoint: tokens.URL,
	}))

	status, page := s.unauthenticatedClient.call(http.MethodGet,
		plugins.CallbackPath+"?state="+state+"&code=the-code", nil)

	s.Equal(http.StatusOK, status)
	s.Contains(string(page), "Google Calendar is connected")
	alices, err := s.store.UserPluginConnection(ctx, s.customerID(), agent.Id, "alice", "google_calendar")
	s.Require().NoError(err)
	s.Equal(store.PluginConnected, alices.Status)
	s.Equal("alices-token", alices.AccessToken)
	_, err = s.store.UserPluginConnection(ctx, s.customerID(), agent.Id, "bob", "google_calendar")
	s.Error(err, "nobody else holds Alice's login")
	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+agent.Id+"/plugins", nil, &connections))
	s.Require().Len(connections, 1, "only the app's own login is the agent's")
	s.Equal(PluginConnectionStatusPending, connections[0].Status)
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+agent.Id, nil, &stored))
	s.Nil(stored.AgentPlugins, "a user's login does not hand the plugin to every session")
}

func (s *PluginsSuite) TestTheLoginsOfAnAgentThatIsNotThereAreNotFound() {
	status, failure := s.serverClient.failure(http.MethodGet,
		"/v1/agents/configs/"+s.utils.uuid()+"/plugins", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)
}

func (s *PluginsSuite) TestLoggingIntoSomethingThatIsNotAPluginIsRefused() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon/authorize", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownPlugin.Message)
}

func (s *PluginsSuite) TestAPluginOnAHostOfItsOwnCannotBeLoggedIntoWithoutOne() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+agent.Id+"/plugins/shopify/authorize", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "instance")
}

func (s *PluginsSuite) TestLoggingInOnAnAgentThatIsNotThereIsNotFound() {
	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+s.utils.uuid()+"/plugins/slack/authorize", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)
}

func (s *PluginsSuite) TestDisconnectingALoginNobodyMadeIsNotFound() {
	agent := s.data.createAgentConfig()

	status, _ := s.serverClient.call(http.MethodDelete,
		"/v1/agents/configs/"+agent.Id+"/plugins/slack", nil)

	s.Equal(http.StatusNotFound, status)
}

func (s *PluginsSuite) TestDisconnectingSomethingThatIsNotAPluginIsRefused() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodDelete,
		"/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownPlugin.Message)
}

func (s *PluginsSuite) TestAnotherAppsAgentHoldsNoLoginsItCanSee() {
	agent := s.data.createAgentConfig()

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/configs/"+agent.Id+"/plugins", nil, nil)
	})
}

func (s *PluginsSuite) TestTheProviderArrivesAtTheCallbackWithoutACredential() {
	// The browser comes from the identity provider holding none of ours, and the state is
	// the secret. A 401 here would be a login that can never finish.
	status, body := s.unauthenticatedClient.call(http.MethodGet, plugins.CallbackPath, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "a code and a state are required")
}

func (s *PluginsSuite) TestALoginThatNobodyStartedIsNotFound() {
	status, body := s.unauthenticatedClient.call(http.MethodGet,
		plugins.CallbackPath+"?state="+s.utils.uuid()+"&code="+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(string(body), "no such login")
}

func (s *PluginsSuite) TestAnEndUsersDeviceMayNotStartALogin() {
	// Refused before the login is started, so nothing reaches the provider on behalf of
	// somebody who may not connect one.
	agent := s.data.createAgentConfig()

	status, _ := s.client.call(http.MethodPost,
		"/v1/agents/configs/"+agent.Id+"/plugins/slack/authorize", nil)

	s.Equal(http.StatusForbidden, status)
}

// catalog is the plugins matching a filter, or all of them for an empty one.
func (s *PluginsSuite) catalog(query string) []Plugin {
	path := "/v1/agents/plugins"
	if query != "" {
		path += "?q=" + url.QueryEscape(query)
	}
	var listed []Plugin
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, path, nil, &listed))
	return listed
}

func named(listed []Plugin) []string {
	ids := make([]string, 0, len(listed))
	for _, plugin := range listed {
		ids = append(ids, plugin.Id)
	}
	return ids
}
