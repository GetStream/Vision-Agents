//go:build integration

package api

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ForwardedTokensSuite runs a router behind the hosted gateway with no Stream secret of any
// customer's app: the gateway forwards a server token for the app it vouched for, and a
// webhook it checked the signature of names the app it came from.
type ForwardedTokensSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	sealer   *auth.Sealer
	app      int64
	customer string
}

func TestForwardedTokensSuite(t *testing.T) {
	suite.Run(t, new(ForwardedTokensSuite))
}

func (s *ForwardedTokensSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "forwarded_tokens"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.sealer, err = auth.NewSealer("forwarded tokens suite")
	s.Require().NoError(err)
}

func (s *ForwardedTokensSuite) SetupTest() {
	s.app = int64(uuid.New().ID()) + 1_000_000
	s.customer = strconv.FormatInt(s.app, 10)
}

// router is a proxy-mode router acting in customers' apps with forwarded tokens.
func (s *ForwardedTokensSuite) router(declares bool) *httptest.Server {
	forwarded, err := streamapp.NewForwarded(streamapp.ForwardedOptions{Store: s.store, Sealer: s.sealer})
	s.Require().NoError(err)
	stored, err := streamapp.NewStored(streamapp.StoredOptions{
		Store: s.store, Forwarded: forwarded,
		Deployment: streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", App: 1, Strict: true}),
	})
	s.Require().NoError(err)
	server, err := NewServer(Options{
		Routers:           map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth:              auth.NewProxy(auth.ProxyOptions{DeclaresKind: declares}),
		AuthMode:          auth.Proxy,
		ProxyDeclaresKind: declares,
		Stream:            streamapp.NewClients(stored, streamapp.ClientsOptions{}),
		Logger:            slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	running := httptest.NewServer(server.Handler())
	s.T().Cleanup(running.Close)
	return running
}

func (s *ForwardedTokensSuite) serverToken() string {
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"exp": time.Now().Add(24 * time.Hour).Unix(),
	}).SignedString([]byte("the-apps-own-secret"))
	s.Require().NoError(err)
	return token
}

// send makes a request as the gateway forwards one, with the server token it minted.
func (s *ForwardedTokensSuite) send(router *httptest.Server, method, path, body string, headers map[string]string) int {
	request, err := http.NewRequest(method, router.URL+path, bytes.NewReader([]byte(body)))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *ForwardedTokensSuite) forwardedBy(token string) map[string]string {
	return map[string]string{
		auth.AppHeader: s.customer, auth.AuthTypeHeader: auth.AuthTypeServer,
		auth.ServerTokenHeader: token, auth.ServerTokenKeyHeader: "forwarded-key",
	}
}

func (s *ForwardedTokensSuite) TestTheServerTokenTheGatewayForwardsIsKeptForTheCallersApp() {
	token := s.serverToken()

	s.send(s.router(true), http.MethodGet, "/v1/agents/sessions", "", s.forwardedBy(token))

	held, err := s.store.StreamAppToken(s.ctx, s.app)
	s.Require().NoError(err)
	s.Equal("forwarded-key", held.APIKey)
	s.Equal(s.customer, held.CustomerID)
	s.NotContains(string(held.Sealed), token, "the token is sealed")
}

func (s *ForwardedTokensSuite) TestAProxyThatDeclaresNoKindsCannotHandTheRouterAToken() {
	s.send(s.router(false), http.MethodGet, "/v1/agents/sessions", "", s.forwardedBy(s.serverToken()))

	_, err := s.store.StreamAppToken(s.ctx, s.app)
	s.ErrorIs(err, store.ErrNoStreamAppToken)
}

func (s *ForwardedTokensSuite) hook() string {
	return fmt.Sprintf(`{"type": "message.new", "channel_id": %q, "channel_type": "agent", "created_at": %q,
  "message": {"id": %q, "text": "hello", "user": {"id": "sam"}}}`,
		"chat-"+uuid.NewString(), time.Now().UTC().Format(time.RFC3339Nano), "message-"+uuid.NewString())
}

func (s *ForwardedTokensSuite) TestAHookTheGatewayVouchedForIsTakenWithoutTheAppsSecret() {
	headers := s.forwardedBy(s.serverToken())
	headers[auth.HookAppHeader] = s.customer

	status := s.send(s.router(true), http.MethodPost, "/v1/chat/hooks/stream/"+s.customer, s.hook(), headers)

	s.Equal(http.StatusOK, status)
	_, err := s.store.StreamAppToken(s.ctx, s.app)
	s.NoError(err, "the token forwarded with the hook is kept for its work")
}

func (s *ForwardedTokensSuite) TestAHookNobodyVouchedForIsRefused() {
	status := s.send(s.router(true), http.MethodPost, "/v1/chat/hooks/stream/"+s.customer, s.hook(),
		map[string]string{signatureHeader: "not-a-signature"})

	s.Equal(http.StatusUnauthorized, status)
}

func (s *ForwardedTokensSuite) TestAHookVouchedForAnotherAppIsRefused() {
	headers := s.forwardedBy(s.serverToken())
	headers[auth.HookAppHeader] = strconv.FormatInt(s.app+1, 10)

	status := s.send(s.router(true), http.MethodPost, "/v1/chat/hooks/stream/"+s.customer, s.hook(), headers)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *ForwardedTokensSuite) TestAVouchIsNotBelievedFromAProxyThatDeclaresNoKinds() {
	headers := s.forwardedBy(s.serverToken())
	headers[auth.HookAppHeader] = s.customer

	status := s.send(s.router(false), http.MethodPost, "/v1/chat/hooks/stream/"+s.customer, s.hook(), headers)

	s.Equal(http.StatusUnauthorized, status)
}
