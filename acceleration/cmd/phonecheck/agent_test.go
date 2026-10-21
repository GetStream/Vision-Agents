package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"
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
	r, err := sdkRouterAt(s.server.URL, "customer-1", newLogger("text", io.Discard))
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
