//go:build integration

package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// appLogSuite is a router whose log a test reads back.
type appLogSuite struct {
	RouterSuite

	logged *lockedLog
}

func (s *appLogSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

// LogAppSuite reads what the router logs for a request and finds the app and public key on
// each line of an authenticated one.
type LogAppSuite struct {
	appLogSuite
}

func TestLogAppSuite(t *testing.T) {
	runSuite(t, new(LogAppSuite))
}

func (s *LogAppSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *LogAppSuite) TestTheAccessLineOfABackendNamesTheAppAndItsPublicKey() {
	line := s.servedAs(s.serverClient)

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
	s.Equal(1, strings.Count(line, "app_id="))
	s.Equal(1, strings.Count(line, "api_key="))
}

func (s *LogAppSuite) TestTheAccessLineOfAnEndUserNamesTheAppAndItsPublicKey() {
	line := s.servedAs(s.client)

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

func (s *LogAppSuite) TestNoLineNamesTheKeysSecret() {
	s.servedAs(s.serverClient)
	s.servedAs(s.client)

	s.NotContains(s.logged.String(), s.app.secret)
}

func (s *LogAppSuite) TestTheAccessLineStillNamesTheCustomer() {
	line := s.servedAs(s.serverClient)

	s.Contains(line, " customer="+s.customerID()+" ")
}

func (s *LogAppSuite) TestTheAccessLineOfARequestNobodyAuthenticatedNamesNeither() {
	line := s.servedAs(s.unauthenticatedClient)

	s.Contains(line, "status=401")
	s.NotContains(line, "app_id")
	s.NotContains(line, "api_key")
}

func (s *LogAppSuite) TestTheAccessLineOfAHookNamesNeitherEvenWhenTheCallerSendsAnApp() {
	hostile := *s.serverClient
	hostile.header = s.serverClient.header.Clone()
	hostile.header.Set(auth.CustomerHeader, s.customerID())
	id := s.utils.uuid()
	hostile.header.Set(RequestIDHeader, id)

	hostile.call(http.MethodPost, "/v1/phone/hooks/stream", map[string]any{})

	line := s.lineOf(id)
	s.Contains(line, "path=/v1/phone/hooks/stream")
	s.NotContains(line, "app_id")
	s.NotContains(line, "api_key")
}

func (s *LogAppSuite) TestALineAHandlerLogsWithTheRequestsContextNamesTheAppAndItsPublicKey() {
	handler := s.router.withCustomer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		s.router.logger.InfoContext(r.Context(), "probe from a handler")
		s.router.logger.Info("probe without the context")
	}))
	node := httptest.NewServer(handler)
	s.T().Cleanup(node.Close)
	request, err := http.NewRequest(http.MethodGet, node.URL+"/v1/agents/sessions", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()

	response, err := node.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), "probe without the context") },
		5*time.Second, 20*time.Millisecond)
	var with, without string
	for line := range strings.Lines(s.logged.String()) {
		switch {
		case strings.Contains(line, "probe from a handler"):
			with = line
		case strings.Contains(line, "probe without the context"):
			without = line
		}
	}
	s.Contains(with, " app_id="+s.customerID())
	s.Contains(with, " api_key="+s.app.key)
	s.NotContains(without, "app_id")
}

// servedAs is the access line of a GET by client.
func (s *appLogSuite) servedAs(client *testClient) string {
	id := s.utils.uuid()
	named := *client
	named.header = client.header.Clone()
	named.header.Set(RequestIDHeader, id)
	named.call(http.MethodGet, "/v1/agents/sessions", nil)
	return s.lineOf(id)
}

// lineOf is the access line of the request that carried id.
func (s *appLogSuite) lineOf(id string) string {
	var found string
	s.Require().Eventually(func() bool {
		for line := range strings.Lines(s.logged.String()) {
			if strings.Contains(line, `msg="served a request"`) && strings.Contains(line, " request_id="+id) {
				found = line
				return true
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond, "the access line of request %s", id)
	return found
}

// LogAppProxiedSuite is the same behind the proxy, which names the key the caller used.
type LogAppProxiedSuite struct {
	appLogSuite
}

func TestLogAppProxiedSuite(t *testing.T) {
	proxied := new(LogAppProxiedSuite)
	proxied.proxied = true
	runSuite(t, proxied)
}

func (s *LogAppProxiedSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *LogAppProxiedSuite) TestTheAccessLineNamesTheKeyTheProxySaidTheCallerUsed() {
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)
	header.Set(auth.APIKeyIDHeader, s.app.key)

	line := s.servedAs(&testClient{suite: &s.RouterSuite, header: header})

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

func (s *LogAppProxiedSuite) TestTheAccessLineOfACallerTheProxyNamedNoKeyForNamesNoKey() {
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)

	line := s.servedAs(&testClient{suite: &s.RouterSuite, header: header})

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.NotContains(line, "api_key")
}
