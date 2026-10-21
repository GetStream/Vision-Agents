package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

// AgentSuite checks the SDK router against a fake router over HTTP.
type AgentSuite struct {
	suite.Suite
	server  *httptest.Server
	mu      sync.Mutex
	methods []string
	logs    int // the status the logs endpoint answers with
}

func TestAgentSuite(t *testing.T) { suite.Run(t, new(AgentSuite)) }

func (s *AgentSuite) SetupTest() {
	s.methods = nil
	s.logs = http.StatusOK
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.methods = append(s.methods, r.Method)
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/health":
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		case "/v1/phone/numbers":
			_, _ = w.Write([]byte(`[]`))
		case "/v1/agents/logs":
			w.WriteHeader(s.logs)
			_, _ = w.Write([]byte(`{"items":[],"has_more":false}`))
		default:
			http.NotFound(w, r)
		}
	}))
}

func (s *AgentSuite) TearDownTest() { s.server.Close() }

func (s *AgentSuite) router() *sdkRouter {
	r, err := newSDKRouter(s.server.URL, "customer-1", newLogger("text", io.Discard))
	s.Require().NoError(err)
	return r
}

func (s *AgentSuite) TestTheChecksOnlyRead() {
	r := s.router()
	ctx := context.Background()

	s.Require().NoError(r.Healthy(ctx))
	_, err := r.Numbers(ctx)
	s.Require().NoError(err)
	s.Require().NoError(r.LogsAvailable(ctx))

	s.Equal([]string{http.MethodGet, http.MethodGet, http.MethodGet}, s.methods)
}

func (s *AgentSuite) TestARouterWithoutPostgresKeepsNoLogs() {
	s.logs = http.StatusServiceUnavailable

	err := s.router().LogsAvailable(context.Background())

	s.ErrorIs(err, errNoLogStore)
}

// runCall runs against the SDK router, with fakes for everything else.
func (s *AgentSuite) runCall(r *sdkRouter, address string) error {
	cfg := Config{Direction: person.Outbound, Router: address, AgentNumber: agentNumber, PersonNumber: personNumber}
	return runCall(context.Background(), cfg, deps{
		router: r, person: &fakePerson{steps: &steps{}}, locator: fakeLocator{}, logger: newLogger("text", io.Discard),
	})
}

func (s *AgentSuite) TestNothingIsWrittenToTheRouterWhenAPreconditionFails() {
	err := s.runCall(s.router(), s.server.URL)

	s.ErrorContains(err, "is not registered")
	s.NotEmpty(s.methods)
	for _, method := range s.methods {
		s.Equal(http.MethodGet, method)
	}
}

func (s *AgentSuite) TestAnUnreachableRouterSaysToStartItFirst() {
	s.server.Close()

	err := s.runCall(s.router(), s.server.URL)

	s.ErrorContains(err, "no router answers at "+s.server.URL+"; start it first")
}
