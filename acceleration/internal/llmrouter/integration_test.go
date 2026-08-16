//go:build integration

package llmrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/deepseek"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// flashModel is the model the test routes to. It is on Baseten's shared Model APIs, so no
// deployment is needed.
const flashModel = "DeepSeek-V4-Flash-0731"

type LLMRouterIntegrationSuite struct {
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

func TestLLMRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(LLMRouterIntegrationSuite))
}

func (s *LLMRouterIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
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

func (s *LLMRouterIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

func (s *LLMRouterIntegrationSuite) SetupTest() {
	// Unique names per test keep rows, counters and health from colliding.
	unique := time.Now().Format("150405.000000000")
	s.customerID = "customer-" + unique
	s.broken = "unreachable-" + unique
	s.working = "deepseek-" + unique
}

// config declares a provider that cannot be reached ahead of a working one, so failover can
// be observed against a real request.
func (s *LLMRouterIntegrationSuite) config() routing.ModalityConfig {
	price := routing.Price{
		PerMillionInputTokens:       0.13,
		PerMillionCachedInputTokens: 0.028,
		PerMillionOutputTokens:      0.26,
	}
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: s.broken, Model: flashModel,
				Languages: []string{"en"}, Realtime: true, Price: price,
			},
			{
				Provider: s.working, Model: flashModel,
				Languages: []string{"en"}, Realtime: true, Price: price,
			},
		},
		Aliases: map[string]routing.Alias{
			"llm-fast": {RequireRealtime: true},
		},
	}
}

func (s *LLMRouterIntegrationSuite) registry() *Registry {
	registry := NewRegistry()
	registry.Register(s.broken, func(spec routing.Spec) (llm.LLM, error) {
		// No API key at all, so the provider cannot even be built and routing has to move
		// on. This is the same shape of failure an undeployed Gemma produces.
		return nil, os.ErrNotExist
	})
	registry.Register(s.working, func(spec routing.Spec) (llm.LLM, error) {
		return deepseek.New(deepseek.Options{Model: spec.Model, Logger: spec.Logger})
	})
	return registry
}

func (s *LLMRouterIntegrationSuite) newRouter() *Router {
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
func (s *LLMRouterIntegrationSuite) requests() []store.Request {
	var rows []store.Request
	err := s.store.DB().NewSelect().
		Model(&rows).
		Where("customer_id = ?", s.customerID).
		Order("id ASC").
		Scan(s.ctx)
	s.Require().NoError(err)
	return rows
}

// ask sends one turn and waits for it to settle.
func (s *LLMRouterIntegrationSuite) ask(session *Session, request llm.Request) llm.CompletionComplete {
	settled := make(chan llm.CompletionComplete, 4)
	go func() {
		for event := range session.Events() {
			if complete, ok := event.(llm.CompletionComplete); ok {
				settled <- complete
			}
		}
		close(settled)
	}()

	s.Require().NoError(session.Respond(request))

	select {
	case complete := <-settled:
		return complete
	case <-time.After(90 * time.Second):
		s.FailNow("timed out waiting for a completed turn")
		return llm.CompletionComplete{}
	}
}

func (s *LLMRouterIntegrationSuite) TestFailoverSkipsTheBrokenProviderAndRecordsWhy() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Target: "llm-fast"})
	s.Require().NoError(err, "the second candidate should serve the request")
	s.Equal(s.working, session.Provider())
	s.Require().NoError(session.Close())

	// Closing the router drains the stat writer, so the rows are all in by now.
	router.Close()

	rows := s.requests()
	s.Require().NotEmpty(rows)

	first := rows[0]
	s.Equal(s.broken, first.Provider, "the failed candidate should be recorded")
	s.Equal("llm", first.Modality)
	s.False(first.Success)

	health, err := s.live.Health(s.ctx, "llm", s.broken, flashModel)
	s.Require().NoError(err)
	s.EqualValues(1, health.Errors, "a failed start should count against the provider")
}

func (s *LLMRouterIntegrationSuite) TestCompletedTurnsAreRecordedWithTheirTokens() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{
		CustomerID: s.customerID,
		Target:     s.working + "/" + flashModel,
	})
	s.Require().NoError(err)

	complete := s.ask(session, llm.Request{
		ID:           "c1",
		Instructions: "Answer with a single word.",
		Messages:     []llm.Message{{Role: llm.User, Content: "What is the capital of France?"}},
		MaxTokens:    32,
	})
	s.Positive(complete.InputTokens)
	s.Positive(complete.OutputTokens)

	s.Require().NoError(session.Close())
	router.Close()

	var successes []store.Request
	for _, row := range s.requests() {
		if row.Success {
			successes = append(successes, row)
		}
	}
	s.Require().Len(successes, 1, "one completion is one row")

	recorded := successes[0]
	s.Equal("llm", recorded.Modality)
	s.Equal(s.working, recorded.Provider, "stats are keyed by the routing identity")
	s.Equal(flashModel, recorded.Model)
	s.Positive(recorded.InputTokens, "tokens are what a model is billed by")
	s.Positive(recorded.OutputTokens)
	s.Positive(recorded.CostMicros, "tokens at a configured rate have a cost")
	s.Zero(recorded.AudioMs, "an LLM bills no audio")
	s.Zero(recorded.Characters, "nor characters")
	s.Require().NotNil(recorded.LatencyMs)
	s.Positive(*recorded.LatencyMs, "latency is the wait for the first token")

	usage, err := s.live.Usage(s.ctx, "llm", s.customerID)
	s.Require().NoError(err)
	s.EqualValues(1, usage.Requests)
	s.Positive(usage.InputTokens)
	s.Positive(usage.OutputTokens)
	s.Positive(usage.CostMicros)

	// The same rows aggregate into the customer's hourly stats.
	from := time.Now().UTC().Add(-2 * time.Hour)
	to := time.Now().UTC().Add(2 * time.Hour)
	_, err = s.store.Rollup(s.ctx, store.Hourly, from, to)
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(s.ctx, "llm", s.customerID, store.Hourly, from, to, nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(buckets)
	s.Positive(buckets[0].InputTokensTotal)
	s.Positive(buckets[0].OutputTokensTotal)
	s.Positive(buckets[0].CostMicrosTotal)
	s.Positive(buckets[0].RequestCount)
	s.Require().NotNil(buckets[0].Uptime)
	s.EqualValues(1, *buckets[0].Uptime, "every turn succeeded")
}

func (s *LLMRouterIntegrationSuite) TestOneSessionAnswersManyTurns() {
	router := s.newRouter()

	session, err := router.Start(s.ctx, Request{
		CustomerID: s.customerID,
		Target:     s.working + "/" + flashModel,
	})
	s.Require().NoError(err)

	settled := make(chan llm.CompletionComplete, 8)
	go func() {
		for event := range session.Events() {
			if complete, ok := event.(llm.CompletionComplete); ok {
				settled <- complete
			}
		}
		close(settled)
	}()

	for _, id := range []string{"c1", "c2", "c3"} {
		s.Require().NoError(session.Respond(llm.Request{
			ID:           id,
			Instructions: "Answer with a single word.",
			Messages:     []llm.Message{{Role: llm.User, Content: "Name a colour."}},
			MaxTokens:    16,
		}))
	}

	for range 3 {
		select {
		case <-settled:
		case <-time.After(90 * time.Second):
			s.FailNow("timed out waiting for every turn to settle")
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
	s.Equal(3, successes, "each turn is its own unit of billable work")
}
