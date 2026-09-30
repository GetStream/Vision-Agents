//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	suiteKey    = "vak_live_suite00000000000000000000"
	suiteSecret = "vas_live_suite"
)

// testClient calls the router with one set of credentials.
type testClient struct {
	require *require.Assertions
	server  *httptest.Server
	header  http.Header
}

// do sends body as JSON and decodes the answer into into, when there is one to decode.
func (c testClient) do(method, path string, body, into any) int {
	payload := bytes.NewReader(nil)
	if body != nil {
		encoded, err := json.Marshal(body)
		c.require.NoError(err)
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequest(method, c.server.URL+path, payload)
	c.require.NoError(err)
	request.Header = c.header.Clone()
	request.Header.Set("Content-Type", "application/json")

	response, err := c.server.Client().Do(request)
	c.require.NoError(err)
	defer response.Body.Close()
	if into != nil && response.StatusCode < http.StatusBadRequest {
		c.require.NoError(json.NewDecoder(response.Body).Decode(into))
	}
	return response.StatusCode
}

// RouterSuite runs the router against Postgres with real API key auth, for suites to embed.
//
// Every test is a different app, so nothing an earlier test or run left in Postgres is
// listed to the next one.
type RouterSuite struct {
	suite.Suite

	store  *store.Store
	server *httptest.Server
	app    string

	// anonymous sends no credentials, client is the end user alice, and backend is the
	// app's own server.
	anonymous testClient
	client    testClient
	backend   testClient
}

func (s *RouterSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	ctx := context.Background()
	logger := slog.New(slog.DiscardHandler)

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(ctx))
	s.store = pgStore

	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Hello."}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: routableConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:    reasoner,
		Store:  pgStore,
		Logger: logger,
		Edge: func(session.Spec, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { sessions.Shutdown() })

	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		if presented != suiteKey {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{OrganizationID: "org-suite", AppID: s.app, Secret: suiteSecret}, nil
	})
	s.Require().NoError(err)

	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.LLM: reasoner},
		Sessions: sessions,
		Store:    pgStore,
		Auth:     authenticator,
		Logger:   logger,
	})
	s.Require().NoError(err)
	s.server = httptest.NewServer(server.Handler())
}

func (s *RouterSuite) TearDownSuite() {
	if s.server != nil {
		s.server.Close()
	}
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *RouterSuite) SetupTest() {
	s.app = "suite-" + uuid.NewString()
	s.anonymous = testClient{require: s.Require(), server: s.server, header: http.Header{}}
	s.client = s.user("alice")
	s.backend = s.caller(jwt.MapClaims{"server": true}, auth.AuthTypeServer)
}

// user is a client signed in as an end user of the app.
func (s *RouterSuite) user(id string) testClient {
	return s.caller(jwt.MapClaims{"user_id": id}, auth.AuthTypeJWT)
}

// caller is a client holding a token signed with the app's secret.
func (s *RouterSuite) caller(claims jwt.MapClaims, authType string) testClient {
	claims["exp"] = time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(suiteSecret))
	s.Require().NoError(err)

	header := http.Header{}
	header.Set(auth.APIKeyHeader, suiteKey)
	header.Set("Authorization", "Bearer "+token)
	header.Set(auth.AuthTypeHeader, authType)
	return testClient{require: s.Require(), server: s.server, header: header}
}

// textSession asks for a conversation in writing, which needs no call.
func textSession(id *string) CreateSessionRequest {
	target, text := "en-low-latency", true
	return CreateSessionRequest{Id: id, Text: &text, Llm: &target}
}

// createSession opens a session and closes it once the test is over.
func (s *RouterSuite) createSession(as testClient, request CreateSessionRequest) Session {
	var created Session
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/sessions", request, &created))
	s.T().Cleanup(func() { as.do(http.MethodDelete, "/v1/agents/sessions/"+created.Id, nil, nil) })
	return created
}

// listSessions reads one page of sessions.
func (s *RouterSuite) listSessions(as testClient, query string) SessionPage {
	var page SessionPage
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/sessions"+query, nil, &page))
	return page
}

// ids is the ids of a list of sessions, in order.
func ids(sessions []Session) []string {
	listed := make([]string, 0, len(sessions))
	for _, one := range sessions {
		listed = append(listed, one.Id)
	}
	return listed
}
