package daytona

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"
)

// platform stands in for Daytona so the wire contract can be tested without a key.
type platform struct {
	server *httptest.Server

	mu sync.Mutex
	// created is every sandbox this platform was asked for.
	created []string
	// deleted is every sandbox it was asked to release.
	deleted []string
	// ran is the body of every code-run, in order.
	ran []map[string]any
	// auth is the last Authorization header seen.
	auth string

	// runStatus and runBody are what a code-run answers with.
	deleteStatus int
	runStatus    int
	runBody      string
	// sequence names the next sandbox created.
	sequence int
}

func newPlatform() *platform {
	stub := &platform{runStatus: http.StatusOK, runBody: `{"result":"12.63\n","exitCode":0}`}
	stub.server = httptest.NewServer(http.HandlerFunc(stub.serve))
	return stub
}

func (p *platform) serve(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.auth = r.Header.Get("Authorization")

	switch {
	case strings.HasPrefix(r.URL.Path, "/api/socket.io"):
		w.WriteHeader(404)
		return
	case r.Method == http.MethodPost && r.URL.Path == "/api/sandbox":
		p.sequence++
		id := "sandbox-" + string(rune('0'+p.sequence))
		p.created = append(p.created, id)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"organizationId":"org","name":"test","user":"root","env":{},"public":false,"networkBlockAll":false,"target":"us","cpu":2,"gpu":0,"memory":4,"disk":10,"id":"` + id + `","state":"started","toolboxProxyUrl":"` + p.server.URL + `/toolbox","labels":{"daytona.io/code-toolbox-language":"python"}}`))

	case r.Method == http.MethodDelete:
		p.deleted = append(p.deleted, r.URL.Path)
		if p.deleteStatus != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(p.deleteStatus)
			_, _ = w.Write([]byte(`{"message":"retry deletion"}`))
			return
		}
		w.WriteHeader(http.StatusNoContent)

	default:
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		body["path"] = r.URL.Path
		body["authorization"] = r.Header.Get("Authorization")
		p.ran = append(p.ran, body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(p.runStatus)
		_, _ = w.Write([]byte(p.runBody))
	}
}

func (p *platform) executed() []map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]map[string]any(nil), p.ran...)
}

func (p *platform) sandboxes() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.created...)
}

func (p *platform) released() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.deleted...)
}

type DaytonaSuite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	box      *Sandbox
}

func TestDaytonaSuite(t *testing.T) {
	suite.Run(t, new(DaytonaSuite))
}

func (s *DaytonaSuite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)

	box, err := New(Options{
		APIKey:   "key-1",
		APIURL:   s.platform.server.URL + "/api",
		ProxyURL: s.platform.server.URL + "/toolbox",
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.box = box
	s.T().Cleanup(func() { _ = box.Close() })
}

func (s *DaytonaSuite) TestAKeyIsRequired() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})

	s.ErrorContains(err, apiKeyEnvVar)
}

func (s *DaytonaSuite) TestCodeRunsInASandboxAndItsOutputComesBack() {
	result, err := s.box.Run(s.ctx, "print(84.20 * 0.15)")

	s.Require().NoError(err)
	s.Equal("12.63\n", result.Output)
	s.Zero(result.ExitCode)

	s.Require().Len(s.platform.executed(), 1)
	ran := s.platform.executed()[0]
	s.Equal("print(84.20 * 0.15)", ran["code"])
	s.Equal("python", ran["language"])
	s.Equal("/toolbox/sandbox-1/process/code-run", ran["path"])
	s.Equal("Bearer key-1", ran["authorization"])
}

func (s *DaytonaSuite) TestOneSandboxServesEveryPieceOfCode() {
	// Creating one is the slow part, and a conversation that delegates twice should not
	// wait for it twice.
	_, err := s.box.Run(s.ctx, "print(1)")
	s.Require().NoError(err)
	_, err = s.box.Run(s.ctx, "print(2)")
	s.Require().NoError(err)

	s.Len(s.platform.sandboxes(), 1)
	s.Len(s.platform.executed(), 2)
}

func (s *DaytonaSuite) TestNothingIsCreatedUntilThereIsCodeToRun() {
	s.Empty(s.platform.sandboxes(), "a session that never delegates never pays for one")
}

func (s *DaytonaSuite) TestCodeThatFailedIsAResultRatherThanAnError() {
	// The code ran. That it did not work is something the model can read and act on.
	s.platform.mu.Lock()
	s.platform.runBody = `{"result":"NameError: total","exitCode":1}`
	s.platform.mu.Unlock()

	result, err := s.box.Run(s.ctx, "print(total)")

	s.Require().NoError(err)
	s.Equal(1, result.ExitCode)
	s.Equal("NameError: total", result.Output)
}

func (s *DaytonaSuite) TestARefusedRunIsReportedWithWhatDaytonaSaid() {
	s.platform.mu.Lock()
	s.platform.runStatus = http.StatusBadGateway
	s.platform.runBody = `{"message":"the sandbox is gone"}`
	s.platform.mu.Unlock()

	_, err := s.box.Run(s.ctx, "print(1)")

	s.ErrorContains(err, "the sandbox is gone")
}

func (s *DaytonaSuite) TestClosingReleasesTheSandbox() {
	_, err := s.box.Run(s.ctx, "print(1)")
	s.Require().NoError(err)

	s.Require().NoError(s.box.Close())

	s.Equal([]string{"/api/sandbox/sandbox-1"}, s.platform.released())
}

func (s *DaytonaSuite) TestClosingTwiceIsSafe() {
	s.NoError(s.box.Close())
	s.NoError(s.box.Close())

	s.Empty(s.platform.released(), "there was never a sandbox to release")
}

func (s *DaytonaSuite) TestCodeIsRefusedOnceTheSandboxIsClosed() {
	s.Require().NoError(s.box.Close())

	_, err := s.box.Run(s.ctx, "print(1)")

	s.ErrorContains(err, "closed")
}

func (s *DaytonaSuite) TestThereHasToBeSomethingToRun() {
	_, err := s.box.Run(s.ctx, "   ")

	s.ErrorContains(err, "no code")
	s.Empty(s.platform.sandboxes())
}

func (s *DaytonaSuite) TestConcurrentRunsCreateOnlyOneSandbox() {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); _, err := s.box.Run(s.ctx, "print(1)"); s.NoError(err) }()
	}
	wg.Wait()
	s.Len(s.platform.sandboxes(), 1)
	s.Len(s.platform.executed(), 8)
	s.NoError(s.box.Close())
}
func (s *DaytonaSuite) TestFailedDeletionRetainsIdentity() {
	_, err := s.box.Run(s.ctx, "print(1)")
	s.Require().NoError(err)
	s.platform.mu.Lock()
	s.platform.deleteStatus = 503
	s.platform.mu.Unlock()
	s.Error(s.box.Close())
	s.Require().NotNil(s.box.box)
	_, err = s.box.Run(s.ctx, "print(2)")
	s.ErrorContains(err, "closed")
	s.platform.mu.Lock()
	s.platform.deleteStatus = 0
	s.platform.mu.Unlock()
	s.NoError(s.box.Close())
	s.Nil(s.box.box)
	s.Len(s.platform.sandboxes(), 1)
}
