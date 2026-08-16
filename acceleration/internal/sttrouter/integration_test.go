//go:build integration

package sttrouter

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
)

type STTRouterIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	live       *live.Client
	audio      stt.PcmData
	customerID string
	// broken and working are provider names unique to each test. Health is keyed by
	// provider, so unique names keep one test's health out of another's ranking and let
	// config order, rather than accumulated history, decide the candidate order.
	broken  string
	working string
}

func TestSTTRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(STTRouterIntegrationSuite))
}

func (s *STTRouterIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		s.T().Skip("DEEPGRAM_API_KEY not set")
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to decode the audio fixture")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient

	audio, err := testaudio.Load16kMono("mia.mp3")
	s.Require().NoError(err)
	s.audio = audio
}

func (s *STTRouterIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

func (s *STTRouterIntegrationSuite) SetupTest() {
	// Unique names per test keep rows, counters and health from colliding.
	unique := time.Now().Format("150405.000000000")
	s.customerID = "customer-" + unique
	s.broken = "unauthorised-" + unique
	s.working = "deepgram-" + unique
}

// config declares a provider whose credentials are deliberately wrong, ahead of a working
// one, so failover can be observed against a real upstream rejection.
func (s *STTRouterIntegrationSuite) config() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: s.broken, Model: "flux-general-en",
				Languages: []string{"en"}, Realtime: true,
				Price: routing.Price{PerAudioHour: 0.36},
			},
			{
				Provider: s.working, Model: "flux-general-en",
				Languages: []string{"en"}, Realtime: true,
				Price: routing.Price{PerAudioHour: 0.36},
			},
		},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}

func (s *STTRouterIntegrationSuite) registry() *Registry {
	registry := DefaultRegistry()
	registry.Register(s.broken, func(spec routing.Spec) (stt.STT, error) {
		return deepgram.New(deepgram.Options{APIKey: "not-a-real-key", Model: spec.Model})
	})
	registry.Register(s.working, func(spec routing.Spec) (stt.STT, error) {
		return deepgram.New(deepgram.Options{Model: spec.Model})
	})
	return registry
}

func (s *STTRouterIntegrationSuite) newRouter() *Router {
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
func (s *STTRouterIntegrationSuite) requests() []store.Request {
	var rows []store.Request
	err := s.store.DB().NewSelect().
		Model(&rows).
		Where("customer_id = ?", s.customerID).
		Order("id ASC").
		Scan(s.ctx)
	s.Require().NoError(err)
	return rows
}

func (s *STTRouterIntegrationSuite) TestFailoverSkipsTheBrokenProviderAndRecordsWhy() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Target: "en-low-latency"})
	s.Require().NoError(err, "the second candidate should serve the request")
	s.Require().NoError(session.Close())

	// Closing the router drains the stat writer, so the rows are all in by now.
	router.Close()

	rows := s.requests()
	s.Require().NotEmpty(rows)

	first := rows[0]
	s.Equal(s.broken, first.Provider, "the failed candidate should be recorded")
	s.Equal("stt", first.Modality)
	s.False(first.Success)
	s.Equal("start_failed", first.ErrorCode)

	health, err := s.live.Health(s.ctx, "stt", s.broken, "flux-general-en")
	s.Require().NoError(err)
	s.EqualValues(1, health.Errors, "a failed start should count against the provider")
}

func (s *STTRouterIntegrationSuite) TestFailoverIsReflectedInTheNextRoutingDecision() {
	router := s.newRouter()
	defer router.Close()

	// Enough failures to push the broken provider past the error-rate threshold.
	for range 3 {
		_, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Target: s.broken + "/flux-general-en"})
		s.Require().Error(err)
	}
	router.Recorder().Close()

	candidates, err := router.Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(candidates)
	s.Equal(s.working+"/flux-general-en", candidates[0].Config.Name(),
		"a provider that keeps failing should stop being the first choice")
}

func (s *STTRouterIntegrationSuite) TestCompletedTurnsAreRecordedForTheCustomer() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Target: s.working + "/flux-general-en"})
	s.Require().NoError(err)
	s.Equal(s.working, session.Provider(), "the session reports its routing identity")

	finals := make(chan stt.Transcript, 8)
	go func() {
		for event := range session.Events() {
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				finals <- transcript
			}
		}
		close(finals)
	}()

	speaker := stt.Participant{ID: "test-user", UserID: "test-user"}
	for _, chunk := range testaudio.Chunks(s.audio, 80) {
		s.Require().NoError(session.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}
	for _, chunk := range testaudio.Chunks(testaudio.Silence(2000), 80) {
		s.Require().NoError(session.ProcessAudio(chunk, speaker))
		time.Sleep(10 * time.Millisecond)
	}

	var transcript stt.Transcript
	select {
	case transcript = <-finals:
	case <-time.After(30 * time.Second):
		s.FailNow("timed out waiting for a final transcript")
	}
	s.Contains(strings.ToLower(transcript.Text), "forgotten treasures")

	s.Require().NoError(session.Close())
	router.Close()

	rows := s.requests()
	s.Require().NotEmpty(rows, "a completed turn should be recorded")

	var successes []store.Request
	for _, row := range rows {
		if row.Success {
			successes = append(successes, row)
		}
	}
	s.Require().NotEmpty(successes)

	recorded := successes[0]
	s.Equal(s.working, recorded.Provider, "stats are keyed by the routing identity")
	s.Equal("flux-general-en", recorded.Model)
	s.Positive(recorded.AudioMs, "the row should carry the billable audio")
	s.Positive(recorded.CostMicros, "billable audio at a configured rate has a cost")
	s.Require().NotNil(recorded.LatencyMs)
	s.Positive(*recorded.LatencyMs)

	usage, err := s.live.Usage(s.ctx, "stt", s.customerID)
	s.Require().NoError(err)
	s.Positive(usage.Requests, "live counters should follow the same requests")
	s.Positive(usage.AudioMs)
	s.Positive(usage.CostMicros)

	// The same rows aggregate into the customer's hourly stats.
	from := time.Now().UTC().Add(-2 * time.Hour)
	to := time.Now().UTC().Add(2 * time.Hour)
	_, err = s.store.Rollup(s.ctx, store.Hourly, from, to)
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(s.ctx, "stt", s.customerID, store.Hourly, from, to, nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(buckets)
	s.Positive(buckets[0].AudioMsTotal)
	s.Positive(buckets[0].RequestCount)
	s.Positive(buckets[0].CostMicrosTotal)
}
