package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// ProxyKindSuite runs a proxy-mode router and sends it requests shaped the way the hosted
// gateway forwards them: the app and the user named in headers, the kind declared or not.
// No Stream keys are configured, so an operation that gets past the server-side check
// answers 400 for want of them, and one stopped at the check answers 403.
type ProxyKindSuite struct {
	suite.Suite
}

func TestProxyKindSuite(t *testing.T) {
	suite.Run(t, new(ProxyKindSuite))
}

func (s *ProxyKindSuite) router(declares bool) *httptest.Server {
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth:     auth.NewProxy(auth.ProxyOptions{DeclaresKind: declares}),
		AuthMode: auth.Proxy,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	running := httptest.NewServer(server.Handler())
	s.T().Cleanup(running.Close)
	return running
}

// chatToken asks for a server-only operation as the gateway forwards a guest's token.
func (s *ProxyKindSuite) chatToken(router *httptest.Server, declared string) int {
	body, err := json.Marshal(map[string]string{"agent_id": "support-agent", "user_id": "someone-else"})
	s.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, router.URL+"/v1/agents/chat-token", bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(auth.AppHeader, "42")
	request.Header.Set(auth.OrganizationHeader, "7")
	request.Header.Set(auth.UserHeader, "guest-visitor")
	if declared != "" {
		request.Header.Set(auth.AuthTypeHeader, declared)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *ProxyKindSuite) TestAnUndeclaredKindIsClientSideWhenTheProxyDeclaresKinds() {
	router := s.router(true)

	s.Equal(http.StatusForbidden, s.chatToken(router, ""), "a guest's token must not reach a server-only operation")
	s.Equal(http.StatusForbidden, s.chatToken(router, auth.AuthTypeJWT))
	s.Equal(http.StatusBadRequest, s.chatToken(router, auth.AuthTypeServer), "a declared backend gets past the check")
}

func (s *ProxyKindSuite) TestAProxyThatDeclaresNothingKeepsTreatingEveryCallerAsABackend() {
	// The default is what deployments behind a proxy that never declares rely on.
	router := s.router(false)

	s.Equal(http.StatusBadRequest, s.chatToken(router, ""))
	s.Equal(http.StatusForbidden, s.chatToken(router, auth.AuthTypeJWT))
}

// idleInspector is a router with nothing to route, which is all a server needs to start.
type idleInspector struct{}

func (idleInspector) Modality() routing.Modality                    { return routing.LLM }
func (idleInspector) Config() routing.ModalityConfig                { return routing.ModalityConfig{} }
func (idleInspector) Providers(context.Context) []routing.Candidate { return nil }
func (idleInspector) Resolve(context.Context, string, []string) ([]routing.Candidate, error) {
	return nil, nil
}
