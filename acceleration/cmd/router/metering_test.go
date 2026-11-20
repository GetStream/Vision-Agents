package main

import (
	"context"
	"crypto/rand"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// RefusalAlertsSuite runs the router's metering alerts as its start does, against the
// process's Sentry hub, with a client that keeps each event and drops it.
type RefusalAlertsSuite struct {
	suite.Suite

	mu   sync.Mutex
	sent []*sentry.Event
}

func TestRefusalAlertsSuite(t *testing.T) {
	suite.Run(t, new(RefusalAlertsSuite))
}

func (s *RefusalAlertsSuite) SetupTest() {
	client, err := sentry.NewClient(sentry.ClientOptions{BeforeSend: func(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.sent = append(s.sent, event)
		return nil
	}})
	s.Require().NoError(err)
	hub := sentry.CurrentHub()
	previous := hub.Client()
	hub.BindClient(client)
	s.T().Cleanup(func() { hub.BindClient(previous) })
}

func (s *RefusalAlertsSuite) fingerprints() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var fingerprints []string
	for _, event := range s.sent {
		fingerprints = append(fingerprints, event.Fingerprint...)
	}
	return fingerprints
}

// Both loops of a router set one gauge by the alert's name, so a name in both would make its
// series flip between the two loops' answers.
func (s *RefusalAlertsSuite) TestTheUsageAndMeteringLoopsShareNoAlertName() {
	seen := map[string]bool{}
	for _, condition := range slices.Concat(routing.Alerts(usageAlertLimit), meteringConditions()) {
		s.False(seen[condition.Name], condition.Name)
		seen[condition.Name] = true
	}
}

func (s *RefusalAlertsSuite) TestAMem0RefusalAndARefusedKeyAlert() {
	quota := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	s.T().Cleanup(quota.Close)
	store, err := mem0.New(mem0.Options{APIKey: "test-key", BaseURL: quota.URL, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	keyVar := "REFUSAL_ALERTS_" + rand.Text() + "_API_KEY"
	key := "refusal-alerts-" + rand.Text()
	s.T().Setenv(keyVar, key)
	pool := keypool.FromEnv("", keyVar)
	stop := startMeteringAlerts(context.Background(), "router/test", "test", slog.New(slog.NewTextHandler(io.Discard, nil)))

	_, err = store.Recall(context.Background(), memory.Query{Scope: memory.Scope{AppID: "router", UserID: "acme", AgentID: "support", RunID: "session-1"}, Text: "anything"})
	s.Require().ErrorContains(err, "429")
	pool.QuotaExceeded(key)
	stop()

	s.Contains(s.fingerprints(), "usage_mem0_refused:search/429")
	s.Contains(s.fingerprints(), "usage_key_refused:"+keyVar+"/quota_exceeded")
}
