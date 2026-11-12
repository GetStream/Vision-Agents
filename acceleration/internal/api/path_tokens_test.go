//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

// PathTokensSuite reads what the router's log says of routes whose secret is in the path.
type PathTokensSuite struct {
	RouterSuite
	logged *lockedLog
}

func TestPathTokens(t *testing.T) {
	runSuite(t, new(PathTokensSuite))
}

func (s *PathTokensSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *PathTokensSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *PathTokensSuite) requireLogged(want string) {
	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), want) },
		5*time.Second, 20*time.Millisecond, want)
}

func (s *PathTokensSuite) TestATokenRouteIsLoggedWithItsTokenTakenOut() {
	for _, route := range []struct{ method, prefix string }{
		{http.MethodGet, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/answer/"},
		{http.MethodPost, "/v1/phone/status/"},
		{http.MethodPost, "/v1/phone/machine/"},
		{http.MethodPost, "/v1/connectors/mcp-events/"},
		{http.MethodGet, "/v1/agents/channels/hooks/"},
		{http.MethodPost, "/v1/agents/channels/hooks/"},
	} {
		token := s.utils.uuid()
		s.unauthenticatedClient.do(route.method, route.prefix+token, nil, nil)

		s.requireLogged("path=" + route.prefix + "{token}")
		s.NotContains(s.logged.String(), token)
	}
}

func (s *PathTokensSuite) TestATokenSentWithAMethodTheRouteDoesNotServeIsStillTakenOut() {
	token := s.utils.uuid()

	s.unauthenticatedClient.do(http.MethodGet, "/v1/phone/status/"+token, nil, nil)

	s.requireLogged("path=/v1/phone/status/{token}")
	s.NotContains(s.logged.String(), token)
}

func (s *PathTokensSuite) TestAnEncodedTokenIsTakenOut() {
	token := s.utils.uuid()

	s.unauthenticatedClient.do(http.MethodPost, "/v1/phone/status/"+token+"%2Fx", nil, nil)

	s.requireLogged("path=/v1/phone/status/{token}")
	s.NotContains(s.logged.String(), token)
	s.NotContains(s.logged.String(), "%2Fx")
}

func (s *PathTokensSuite) TestARouteWithoutATokenIsLoggedAsItWasSent() {
	path := "/v1/agents/sessions/" + s.utils.uuid() + "/events"

	s.unauthenticatedClient.do(http.MethodGet, path, nil, nil)

	s.requireLogged("path=" + path)
}
