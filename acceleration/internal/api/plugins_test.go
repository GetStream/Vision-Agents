//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PluginsSuite covers the hosted MCP catalog and the logins an agent holds against it.
//
// Starting a login discovers the provider's OAuth endpoints over the network, so what is
// asserted about authorizing is everything the router decides before it goes out.
type PluginsSuite struct {
	RouterSuite

	// logged is what the router logged, for the tests of the deprecation it warns of.
	logged *lockedLog
}

func TestPluginsSuite(t *testing.T) {
	runSuite(t, new(PluginsSuite))
}

// SetupSuite lets the API reach the token servers the tests stand up on loopback, which
// the router's own client refuses.
func (s *PluginsSuite) SetupSuite() {
	s.pluginHTTP = &http.Client{}
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
	// The built-ins, as a router start seeds them, for the tests of a config binding nessy.
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
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
			Name:    "on-call-" + s.utils.uuid(),
			Plugins: pointerTo([]PluginEntry{{Name: "sentry"}, {Name: "google_calendar", User: pointerTo(true)}}),
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
	s.Equal([]PluginEntry{{Name: "sentry"}, {Name: "google_calendar"}}, *agent.Plugins)
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
			Name:    "assistant-" + s.utils.uuid(),
			Plugins: pointerTo([]PluginEntry{{Name: "linear", User: pointerTo(true)}}),
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
	s.Nil(stored.Plugins)
}

func (s *PluginsSuite) TestASyncNamingAPluginNobodyCanConnectYetIsStoredWithAWarning() {
	name := "assistant-" + s.utils.uuid()
	sync := map[string]any{"name": name, "hash": "v1", "mode": "text", "plugins": []any{
		map[string]any{"name": "linear", "user": true}, map[string]any{"name": "google_calendar", "user": true},
	}}

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
	s.Nil(stored.Plugins, "a user's login does not hand the plugin to every session")
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
func (s *PluginsSuite) TestCreatingAConfigWithPluginsWarnsOfTheDeprecationAndStoresThem() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "assistant-" + s.utils.uuid(), Plugins: pointerTo([]PluginEntry{{Name: "linear", User: pointerTo(true)}})}, &created))

	lines := deprecations(s.logged, plugins.PathConfigSave, created.Id)
	s.Require().Len(lines, 1)
	s.Contains(lines[0], "customer="+s.customerID()+" config="+created.Id+" plugin=[linear] unbound=1 plugin_events=0")
	s.Equal([]string{"linear"}, s.storedPlugins(created.Id), "stored as before")
}

func (s *PluginsSuite) TestReplacingAConfigWithPluginsWarnsOfTheDeprecationAndStoresThem() {
	agent := s.data.createAgentConfig()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id,
		AgentConfigRequest{Name: agent.Name, Plugins: pointerTo([]PluginEntry{{Name: "linear", User: pointerTo(true)}})}, nil))

	s.Len(deprecations(s.logged, plugins.PathConfigSave, agent.Id), 1)
	s.Equal([]string{"linear"}, s.storedPlugins(agent.Id))
}

func (s *PluginsSuite) TestPatchingAConfigThatHasPluginsWarnsOfTheDeprecationAndStoresThem() {
	agent := s.data.createAgentConfig()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+agent.Id,
		AgentConfigPatch{Plugins: pointerTo([]PluginEntry{{Name: "linear", User: pointerTo(true)}})}, nil))

	s.Len(deprecations(s.logged, plugins.PathConfigSave, agent.Id), 1)
	s.Equal([]string{"linear"}, s.storedPlugins(agent.Id))
}

func (s *PluginsSuite) TestSyncingAnAgentWithPluginsWarnsOfTheDeprecationAndStoresThem() {
	var synced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", map[string]any{
		"name": "assistant-" + s.utils.uuid(), "hash": "v1", "mode": "text",
		"plugins": []any{map[string]any{"name": "linear", "user": true}},
	}, &synced))

	s.Len(deprecations(s.logged, plugins.PathConfigSave, synced.Config.Id), 1)
	s.Equal([]string{"linear"}, s.storedPlugins(synced.Config.Id))
}

func (s *PluginsSuite) TestSavingAConfigWithoutPluginsWarnsOfNothing() {
	agent := s.data.createAgentConfig()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id,
		AgentConfigRequest{Name: agent.Name, Instructions: pointerTo("be kind")}, nil))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+agent.Id,
		AgentConfigPatch{Instructions: pointerTo("be brief")}, nil))
	var synced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "assistant-" + s.utils.uuid(), "hash": "v1", "mode": "text"}, &synced))

	s.Empty(deprecations(s.logged, plugins.PathConfigSave, agent.Id))
	s.Empty(deprecations(s.logged, plugins.PathConfigSave, synced.Config.Id))
	s.Empty(s.storedPlugins(agent.Id))
}

func (s *PluginsSuite) TestAFinishedEndUserLoginWarnsOfTheDeprecationAndIsStored() {
	agent := s.data.createAgentConfig()
	tokens := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "alices-token", "expires_in": 3600})
	}))
	defer tokens.Close()
	state := s.utils.uuid()
	s.Require().NoError(s.store.UpsertPluginConnection(s.T().Context(), &store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: agent.Id, PluginID: "google_calendar", UserID: "alice",
		OAuthState: state, CodeVerifier: "verifier", ClientID: "client", TokenEndpoint: tokens.URL,
	}))

	status, _ := s.unauthenticatedClient.call(http.MethodGet, plugins.CallbackPath+"?state="+state+"&code=the-code", nil)

	s.Equal(http.StatusOK, status)
	lines := deprecations(s.logged, plugins.PathCallback, agent.Id)
	s.Require().Len(lines, 1)
	s.Contains(lines[0], "customer="+s.customerID()+" config="+agent.Id+" plugin=google_calendar via=plugins")
	s.NotContains(s.logged.String(), "alices-token")
	alices, err := s.store.UserPluginConnection(s.T().Context(), s.customerID(), agent.Id, "alice", "google_calendar")
	s.Require().NoError(err)
	s.Equal(store.PluginConnected, alices.Status)
}

func (s *PluginsSuite) TestEveryOtherUseOfThePluginAPIWarnsOfTheDeprecation() {
	agent := s.data.createAgentConfig()
	clientURL := "/v1/agents/configs/" + agent.Id + "/plugins/google_calendar"

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, clientURL+"/client",
		SetPluginClientRequest{ClientId: "acme-client", ClientSecret: pointerTo("acme-secret"), User: pointerTo(true)}, nil))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+agent.Id+"/plugins", nil, nil))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, clientURL+"/client", nil, nil))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, clientURL, nil, nil))

	prefix := "customer=" + s.customerID() + " config=" + agent.Id
	set := deprecations(s.logged, plugins.PathClientSet, agent.Id)
	s.Require().Len(set, 1)
	s.Contains(set[0], prefix+" plugin=google_calendar via=plugins")
	listed := deprecations(s.logged, plugins.PathListConfig, agent.Id)
	s.Require().Len(listed, 1)
	s.Contains(listed[0], prefix)
	deleted := deprecations(s.logged, plugins.PathClientDelete, agent.Id)
	s.Require().Len(deleted, 1)
	s.Contains(deleted[0], prefix+" plugin=google_calendar via=plugins")
	disconnected := deprecations(s.logged, plugins.PathDisconnect, agent.Id)
	s.Require().Len(disconnected, 1)
	s.Contains(disconnected[0], prefix+" plugin=google_calendar via=plugins")
	s.NotContains(s.logged.String(), "acme-secret")
}

func (s *PluginsSuite) TestARefusedUseOfThePluginAPIWarnsOfNothing() {
	agent := s.data.createAgentConfig()

	status, _ := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon/client",
		SetPluginClientRequest{ClientId: "x"})
	s.Equal(http.StatusNotFound, status)
	status, _ = s.serverClient.failure(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon", nil)
	s.Equal(http.StatusNotFound, status)

	status, _ = s.serverClient.failure(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client", nil)
	s.Equal(http.StatusNotFound, status, "a client never set")
	status, _ = s.serverClient.failure(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon/client", nil)
	s.Equal(http.StatusNotFound, status, "an unknown plugin")
	status, _ = s.serverClient.failure(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/slack", nil)
	s.Equal(http.StatusNotFound, status, "a login nobody made")

	s.Empty(deprecations(s.logged, plugins.PathClientSet, agent.Id))
	s.Empty(deprecations(s.logged, plugins.PathClientDelete, agent.Id))
	s.Empty(deprecations(s.logged, plugins.PathDisconnect, agent.Id))
}

// nessyConfig stores a config binding nessy, which nessy.yaml marks sensitive, in an app Stream
// marks internal, the only kind that sees it, with the plugins given.
func (s *PluginsSuite) nessyConfig(entries ...map[string]any) AgentConfig {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	var created AgentConfig
	status, raw := s.serverClient.call(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena-" + s.utils.uuid(), "connectors": []map[string]any{sessionNessy("customers")}, "plugins": entries,
	})
	s.Require().Equal(http.StatusCreated, status, string(raw))
	s.Require().NoError(json.Unmarshal(raw, &created))
	return created
}

// A client set with user names the plugin, an entry that may send data out, so beside nessy it
// is refused as an update naming it would be, and stores neither the client nor the entry.
func (s *PluginsSuite) TestAClientNamingAPluginBesideNessyIsRefused() {
	agent := s.nessyConfig()

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy", whose tools read private data, `+
		`and plugin "google_calendar" can send data out`)
	_, err := s.store.PluginClient(s.T().Context(), s.customerID(), agent.Id, "google_calendar")
	s.ErrorIs(err, store.ErrUnknownPluginClient)
	s.Nil(s.storedPlugins(agent.Id))
}

// A client that names no plugin, without user or for one named read-only, adds nothing that
// sends data out. The base, cc60bf611, answered 200 to both (ai1641/author-1.md, «Control probe»).
func (s *PluginsSuite) TestAClientBesideNessyThatNamesNoNewPluginIsStored() {
	agent := s.nessyConfig(map[string]any{"name": "linear", "readonly": true, "user": false})

	for _, plugin := range []string{"google_calendar", "linear"} {
		status := s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/"+plugin+"/client",
			SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(plugin == "linear")}, nil)

		s.Equal(http.StatusOK, status, plugin)
	}
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), agent.Id)
	s.Require().NoError(err)
	s.Equal([]store.PluginEntry{{Name: "linear", Readonly: true}}, config.Plugins)
}

// The app's login names the plugin on the config (store.AddConfigPlugin), so beside nessy it
// is refused before the code is exchanged: no token is asked for or stored, and the login failed.
func (s *PluginsSuite) TestAnAppLoginNamingAPluginBesideNessyIsRefused() {
	agent := s.nessyConfig()
	exchanged := false
	tokens := s.tokenServer(&exchanged)
	state := s.pendingLogin(agent.Id, "linear", "", tokens.URL)

	status, failure := s.callback(state)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `and plugin "linear" can send data out`)
	s.False(exchanged)
	s.Nil(s.storedPlugins(agent.Id))
	conn := s.appLogin(agent.Id, "linear")
	s.Equal(store.PluginFailed, conn.Status)
	s.Empty(conn.AccessToken)
}

// Logins that name no new plugin are finished as before: the app's of a plugin named
// read-only, the app's of a server the config names by URL, and an end user's. The base,
// cc60bf611, answered 302, 302 and 200 (ai1641/author-1.md, «Control probe»).
func (s *PluginsSuite) TestLoginsBesideNessyThatNameNoNewPluginAreFinished() {
	agent := s.nessyConfig(map[string]any{"name": "linear", "readonly": true, "user": false})
	exchanged := false
	tokens := s.tokenServer(&exchanged)

	for _, login := range []struct {
		plugin, user string
		status       int
	}{{"linear", "", http.StatusFound}, {"tablejourney", "", http.StatusFound}, {"google_calendar", "alice", http.StatusOK}} {
		status, _ := s.callback(s.pendingLogin(agent.Id, login.plugin, login.user, tokens.URL))

		s.Equal(login.status, status, login.plugin)
	}
	s.True(exchanged)
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), agent.Id)
	s.Require().NoError(err)
	s.Equal([]store.PluginEntry{{Name: "linear", Readonly: true}}, config.Plugins)
	conn := s.appLogin(agent.Id, "linear")
	s.Equal(store.PluginConnected, conn.Status)
}

// TestAnAppLoginWithoutASensitiveConnectorNamesThePlugin is the control: the base,
// cc60bf611, answered 302 and named linear (ai1641/author-1.md, «Control probe»).
func (s *PluginsSuite) TestAnAppLoginWithoutASensitiveConnectorNamesThePlugin() {
	agent := s.data.createAgentConfig()
	exchanged := false
	tokens := s.tokenServer(&exchanged)

	status, _ := s.callback(s.pendingLogin(agent.Id, "linear", "", tokens.URL))

	s.Equal(http.StatusFound, status)
	s.Equal([]string{"linear"}, s.storedPlugins(agent.Id))
}

// The config an app login names is read before the code is exchanged, so a login on a config
// that is gone asks for no token. The base, cc60bf611, answered the same 500 after exchanging
// the code (ai1641/author-1.md, «Control probe»).
func (s *PluginsSuite) TestAnAppLoginOnAConfigThatIsGoneExchangesNoCode() {
	exchanged := false
	tokens := s.tokenServer(&exchanged)

	status, _ := s.callback(s.pendingLogin(s.utils.uuid(), "linear", "", tokens.URL))

	s.Equal(http.StatusInternalServerError, status)
	s.False(exchanged)
}

// tokenServer answers a code with a token, and records that it was asked.
func (s *PluginsSuite) tokenServer(exchanged *bool) *httptest.Server {
	tokens := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		*exchanged = true
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "a-token", "expires_in": 3600})
	}))
	s.T().Cleanup(tokens.Close)
	return tokens
}

// pendingLogin stores a login started for the plugin, the app's when user is empty, and
// returns its state.
func (s *PluginsSuite) pendingLogin(configID, pluginID, user, tokenEndpoint string) string {
	state := s.utils.uuid()
	s.Require().NoError(s.store.UpsertPluginConnection(s.T().Context(), &store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: configID, PluginID: pluginID, UserID: user,
		Status: store.PluginPending, OAuthState: state, CodeVerifier: "verifier", ClientID: "client",
		TokenEndpoint: tokenEndpoint,
	}))
	return state
}

// appLogin is the app's own login of the plugin on the config.
func (s *PluginsSuite) appLogin(configID, pluginID string) store.PluginConnection {
	conns, err := s.store.PluginConnections(s.T().Context(), s.customerID(), configID)
	s.Require().NoError(err)
	at := slices.IndexFunc(conns, func(conn store.PluginConnection) bool { return conn.PluginID == pluginID })
	s.Require().GreaterOrEqual(at, 0, pluginID)
	return conns[at]
}

// callback is the browser arriving back from the provider, without following the redirect
// to the dashboard.
func (s *PluginsSuite) callback(state string) (int, string) {
	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := browser.Get(s.server.URL + plugins.CallbackPath + "?state=" + url.QueryEscape(state) + "&code=the-code")
	s.Require().NoError(err)
	defer response.Body.Close()
	var answered ErrorResponse
	_ = json.NewDecoder(response.Body).Decode(&answered)
	return response.StatusCode, answered.Error.Message
}

// storedPlugins are the names of the plugins the config is stored with.
func (s *PluginsSuite) storedPlugins(configID string) []string {
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), configID)
	s.Require().NoError(err)
	if len(config.Plugins) == 0 {
		return nil
	}
	return store.PluginNames(config.Plugins)
}

// deprecations are the lines the router logged for a use of the plugin system on path by the
// config.
func deprecations(logged *lockedLog, path, configID string) []string {
	var found []string
	for _, line := range strings.Split(logged.String(), "\n") {
		if strings.Contains(line, `level=WARN msg="`+plugins.DeprecatedUse+`" path=`+path+" ") &&
			strings.Contains(line+" ", " config="+configID+" ") {
			found = append(found, line)
		}
	}
	return found
}

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
