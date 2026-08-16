//go:build integration

package ttsrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// sentence takes a couple of seconds to say, so the recorded duration and cost are
// unambiguously non-zero.
const sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

type TTSRouterIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	live       *live.Client
	customerID string
	// broken and working are provider names unique to each test. Health is keyed by
	// provider, so unique names keep one test's health out of another's ranking and let
	// config order, rather than accumulated history, decide the candidate order.
	broken  string
	working string
}

func TestTTSRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TTSRouterIntegrationSuite))
}

func (s *TTSRouterIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	if os.Getenv("ELEVENLABS_API_KEY") == "" {
		s.T().Skip("ELEVENLABS_API_KEY not set")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient
}

func (s *TTSRouterIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

func (s *TTSRouterIntegrationSuite) SetupTest() {
	// Unique names per test keep rows, counters and health from colliding.
	unique := time.Now().Format("150405.000000000")
	s.customerID = "customer-" + unique
	s.broken = "unreachable-" + unique
	s.working = "elevenlabs-" + unique
}

// config declares a provider that cannot be reached ahead of a working one, so failover
// can be observed against a real connection attempt.
func (s *TTSRouterIntegrationSuite) config() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: s.broken, Model: elevenlabs.DefaultModel,
				Languages: []string{"en"}, Realtime: true,
				Price: routing.Price{PerMillionChars: 50},
			},
			{
				Provider: s.working, Model: elevenlabs.DefaultModel,
				Languages: []string{"en"}, Realtime: true,
				Price: routing.Price{PerMillionChars: 50},
			},
		},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}

func (s *TTSRouterIntegrationSuite) registry() *Registry {
	registry := DefaultRegistry()
	registry.Register(s.broken, func(spec routing.Spec) (tts.TTS, error) {
		// Port 1 is reserved and nothing listens on it, so the dial fails outright rather
		// than depending on how a particular vendor rejects a bad request.
		return elevenlabs.New(elevenlabs.Options{
			APIKey:           "not-a-real-key",
			Model:            spec.Model,
			BaseURL:          "ws://127.0.0.1:1",
			HandshakeTimeout: 2 * time.Second,
		})
	})
	registry.Register(s.working, func(spec routing.Spec) (tts.TTS, error) {
		return elevenlabs.New(elevenlabs.Options{Model: spec.Model})
	})
	return registry
}

func (s *TTSRouterIntegrationSuite) newRouter() *Router {
	router, err := New(Options{
		Config:   s.config(),
		Registry: s.registry(),
		Store:    s.store,
		Live:     s.live,
	})
	s.Require().NoError(err)
	return router
}

// requests reads back the rows recorded for the current customer.
func (s *TTSRouterIntegrationSuite) requests() []store.Request {
	var rows []store.Request
	err := s.store.DB().NewSelect().
		Model(&rows).
		Where("customer_id = ?", s.customerID).
		Order("id ASC").
		Scan(s.ctx)
	s.Require().NoError(err)
	return rows
}

func (s *TTSRouterIntegrationSuite) TestFailoverSkipsTheBrokenProviderAndRecordsWhy() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Target: "en-low-latency"})
	s.Require().NoError(err, "the second candidate should serve the request")
	s.Equal(s.working, session.Provider(), "the request should be served by the reachable provider")
	s.Require().NoError(session.Close())

	// Closing the router drains the stat writer, so the rows are all in by now.
	router.Close()

	rows := s.requests()
	s.Require().NotEmpty(rows)

	first := rows[0]
	s.Equal(s.broken, first.Provider, "the failed candidate should be recorded")
	s.Equal("tts", first.Modality)
	s.False(first.Success)
	s.Equal("start_failed", first.ErrorCode)

	health, err := s.live.Health(s.ctx, "tts", s.broken, elevenlabs.DefaultModel)
	s.Require().NoError(err)
	s.EqualValues(1, health.Errors, "a failed start should count against the provider")
}

func (s *TTSRouterIntegrationSuite) TestCompletedSynthesesAreRecordedForTheCustomer() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{
		CustomerID: s.customerID,
		Target:     s.working + "/" + elevenlabs.DefaultModel,
	})
	s.Require().NoError(err)
	s.Equal(s.working, session.Provider(), "the session reports its routing identity")

	settled := make(chan tts.SynthesisComplete, 4)
	go func() {
		for event := range session.Events() {
			if complete, ok := event.(tts.SynthesisComplete); ok {
				settled <- complete
			}
		}
		close(settled)
	}()

	s.Require().NoError(session.Synthesize(tts.Request{ID: "u1", Text: sentence, Final: true}))

	var complete tts.SynthesisComplete
	select {
	case complete = <-settled:
	case <-time.After(60 * time.Second):
		s.FailNow("timed out waiting for a completed synthesis")
	}
	s.Positive(complete.AudioDurationMs)
	s.EqualValues(len(sentence), complete.Characters)

	s.Require().NoError(session.Close())
	router.Close()

	rows := s.requests()
	s.Require().NotEmpty(rows, "a completed synthesis should be recorded")

	var successes []store.Request
	for _, row := range rows {
		if row.Success {
			successes = append(successes, row)
		}
	}
	s.Require().Len(successes, 1, "one utterance is one row")

	recorded := successes[0]
	s.Equal("tts", recorded.Modality)
	s.Equal(s.working, recorded.Provider, "stats are keyed by the routing identity")
	s.Equal(elevenlabs.DefaultModel, recorded.Model)
	s.EqualValues(len(sentence), recorded.Characters, "text is what a voice is billed by")
	s.Positive(recorded.AudioMs, "the row should carry the speech it produced")
	s.Positive(recorded.CostMicros, "characters at a configured rate have a cost")
	s.Require().NotNil(recorded.LatencyMs)
	s.Positive(*recorded.LatencyMs, "latency is the wait for the first audio")

	usage, err := s.live.Usage(s.ctx, "tts", s.customerID)
	s.Require().NoError(err)
	s.EqualValues(1, usage.Requests)
	s.Positive(usage.Characters)
	s.Positive(usage.CostMicros)

	// The same rows aggregate into the customer's hourly stats.
	from := time.Now().UTC().Add(-2 * time.Hour)
	to := time.Now().UTC().Add(2 * time.Hour)
	_, err = s.store.Rollup(s.ctx, store.Hourly, from, to)
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(s.ctx, "tts", s.customerID, store.Hourly, from, to, nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(buckets)
	s.Positive(buckets[0].CharactersTotal)
	s.Positive(buckets[0].CostMicrosTotal)
	s.Positive(buckets[0].RequestCount)
}

func (s *TTSRouterIntegrationSuite) TestOneSessionSaysManyThings() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{
		CustomerID: s.customerID,
		Target:     s.working + "/" + elevenlabs.DefaultModel,
	})
	s.Require().NoError(err)

	settled := make(chan tts.SynthesisComplete, 8)
	go func() {
		for event := range session.Events() {
			if complete, ok := event.(tts.SynthesisComplete); ok {
				settled <- complete
			}
		}
		close(settled)
	}()

	for _, id := range []string{"u1", "u2", "u3"} {
		s.Require().NoError(session.Synthesize(tts.Request{ID: id, Text: sentence, Final: true}))
	}

	for range 3 {
		select {
		case <-settled:
		case <-time.After(60 * time.Second):
			s.FailNow("timed out waiting for every utterance to settle")
		}
	}

	s.Require().NoError(session.Close())
	router.Close()

	var successes int
	for _, row := range s.requests() {
		if row.Success {
			successes++
		}
	}
	s.Equal(3, successes, "each utterance is its own unit of billable work")
}
