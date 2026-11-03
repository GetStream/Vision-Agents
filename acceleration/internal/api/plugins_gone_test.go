//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
)

// PluginsGoneSuite pins that the plugin operations are no longer served or declared:
// connectors replaced them, and a client that still calls one is told the route is not there.
type PluginsGoneSuite struct {
	RouterSuite
}

func TestPluginsGoneSuite(t *testing.T) {
	runSuite(t, new(PluginsGoneSuite))
}

func (s *PluginsGoneSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// pluginRoutes are the routes the plugin operations were served on, the three served by hand
// last.
var pluginRoutes = []struct{ method, path string }{
	{http.MethodGet, "/v1/agents/plugins"},
	{http.MethodGet, "/v1/agents/configs/{id}/plugins"},
	{http.MethodPost, "/v1/agents/configs/{id}/plugins/linear/authorize"},
	{http.MethodDelete, "/v1/agents/configs/{id}/plugins/linear"},
	{http.MethodPut, "/v1/agents/configs/{id}/plugins/linear/client"},
	{http.MethodDelete, "/v1/agents/configs/{id}/plugins/linear/client"},
	{http.MethodGet, "/v1/agents/plugins/callback?state=s1&code=c1"},
	{http.MethodGet, "/v1/agents/plugins/linear/logo"},
	{http.MethodPost, "/v1/agents/plugins/events/t1"},
}

func (s *PluginsGoneSuite) TestNoPluginRouteIsServed() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support"}, &created))

	for _, route := range pluginRoutes {
		path := strings.ReplaceAll(route.path, "{id}", created.Id)
		status, _ := s.serverClient.call(route.method, path, map[string]any{})
		s.Equal(http.StatusNotFound, status, route.method+" "+path)
	}
}

func (s *PluginsGoneSuite) TestTheSpecDeclaresNoPluginOperation() {
	rendered, err := Spec()
	s.Require().NoError(err)

	for _, operation := range []string{"listPlugins", "listConfigPlugins", "authorizePlugin", "disconnectPlugin",
		"setPluginClient", "deletePluginClient", "pluginOAuthCallback", "getPluginLogo", "receivePluginEvent"} {
		s.NotContains(string(rendered), "operationId: "+operation+"\n")
	}
	s.NotContains(string(rendered), "/plugins")
}
