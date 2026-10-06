package session

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// withApps has sessions act in the apps given, keyed by customer, and in the deployment's
// own app for anybody else.
func (s *SessionSuite) withApps(apps map[string]streamapp.Identity) {
	s.apps = streamapp.NewClients(streamapp.NewStatic(apps, streamapp.NewDeployment(
		streamapp.DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret"})), streamapp.ClientsOptions{})
}

func (s *SessionSuite) TestASessionPinsTheAppItWasCreatedIn() {
	s.withApps(map[string]streamapp.Identity{
		"acme": {StreamApp: 77, APIKey: "acme-key", Secret: streamapp.NewSecret("acme-secret")},
	})
	s.manages()

	created := s.joins(Spec{CustomerID: "acme"})

	s.Equal(int64(77), created.Spec().StreamApp, "the session keeps the app it was created in")
}

func (s *SessionSuite) TestAVoiceEdgeIsBuiltWithTheCallersKey() {
	// The customer's clients created the call in their own app, so the agent has to join
	// it there, with that app's key.
	s.withApps(map[string]streamapp.Identity{
		"acme": {StreamApp: 77, APIKey: "acme-key", Secret: streamapp.NewSecret("acme-secret")},
	})
	s.manages()

	s.joins(Spec{CustomerID: "acme"})

	s.Require().Len(s.edgeApps, 1)
	s.Equal("acme-key", s.edgeApps[0].APIKey)
	s.Equal("acme", s.edgeApps[0].CustomerID)
}

func (s *SessionSuite) TestADeploymentSessionCarriesNoPin() {
	s.withApps(nil)
	s.manages()

	created := s.joins(Spec{CustomerID: "globex"})

	s.Zero(created.Spec().StreamApp)
	s.Require().Len(s.edgeApps, 1)
	s.Equal("deploy-key", s.edgeApps[0].APIKey)
}

// asked is what a guardrail webhook was sent.
type asked struct {
	mu      sync.Mutex
	headers http.Header
	body    []byte
}

func (s *SessionSuite) webhookReceiver() (*httptest.Server, *asked) {
	seen := &asked{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen.mu.Lock()
		seen.headers, seen.body = r.Header.Clone(), body
		seen.mu.Unlock()
		_ = json.NewEncoder(w).Encode(guardrail.Answer{Allow: true})
	}))
	s.T().Cleanup(server.Close)
	return server, seen
}

func (s *SessionSuite) screened(customer string, stream streamapp.Identity, url string) {
	screening, err := s.manager.guardrail(s.ctx, Spec{
		CustomerID: customer, CallID: "call-1",
		Guardrail: "---\ntype: webhook\nurl: " + url + "\n---\nAllow everything.",
	}, stream)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = screening.Close() })
	_, err = screening.Check(s.ctx, "turn-1", "hello")
	s.Require().NoError(err)
}

func (s *SessionSuite) TestAGuardrailWebhookIsSignedWithTheAppsSecret() {
	// A customer acting in its own app checks the signature with its own secret, and is
	// told which of its keys that was.
	s.manages()
	receiver, seen := s.webhookReceiver()

	s.screened("77", streamapp.Identity{
		CustomerID: "77", StreamApp: 77, APIKey: "own-key", Secret: streamapp.NewSecret("own-secret"),
	}, receiver.URL)

	seen.mu.Lock()
	defer seen.mu.Unlock()
	s.Equal(guardrail.Sign("own-secret", seen.headers.Get(guardrail.TimestampHeader), seen.body),
		seen.headers.Get(guardrail.SignatureHeader))
	s.Equal("own-key", seen.headers.Get(guardrail.APIKeyHeader))
}

func (s *SessionSuite) TestAFallbackAppsGuardrailIsSignedAsToday() {
	// A customer acting in the deployment's shared app is signed for as it always was, and
	// no key is named: the key is not theirs to check with.
	s.manages()
	receiver, seen := s.webhookReceiver()

	s.screened("acme", streamapp.Identity{
		CustomerID: "acme", APIKey: "deploy-key", Secret: streamapp.NewSecret("deploy-secret"),
	}, receiver.URL)

	seen.mu.Lock()
	defer seen.mu.Unlock()
	s.Equal(guardrail.Sign("deploy-secret", seen.headers.Get(guardrail.TimestampHeader), seen.body),
		seen.headers.Get(guardrail.SignatureHeader))
	s.Empty(seen.headers.Get(guardrail.APIKeyHeader))
}
