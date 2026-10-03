//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
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

func (s *PluginsSuite) TestAPluginOnAHostOfItsOwnSaysSoInTheCatalog() {
	shopify := s.catalog("shopify")

	s.Require().Len(shopify, 1)
	s.Require().NotNil(shopify[0].InstanceRequired)
	s.True(*shopify[0].InstanceRequired, "a shop has no single global host")
	s.Require().NotNil(shopify[0].InstanceHint)
	s.NotEmpty(*shopify[0].InstanceHint)
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
			Name:        "on-call-" + s.utils.uuid(),
			Plugins:     pointerTo([]string{"sentry"}),
			UserPlugins: pointerTo([]string{"google_calendar"}),
		}, &agent))

	var connections []PluginConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/configs/"+agent.Id+"/plugins", nil, &connections))

	s.Require().Len(connections, 1, "each user connects their own calendar, so the app has nothing to finish for it")
	s.Equal("sentry", connections[0].PluginId)
	s.Equal(PluginConnectionStatusNotConnected, connections[0].Status)
	s.Equal([]string{"google_calendar"}, *agent.UserPlugins)
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
	s.Contains(failure, unknownConfig)
}

func (s *PluginsSuite) TestLoggingIntoSomethingThatIsNotAPluginIsRefused() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/configs/"+agent.Id+"/plugins/carrier-pigeon/authorize", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, unknownPlugin)
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
	s.Contains(failure, unknownConfig)
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

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, unknownPlugin)
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
