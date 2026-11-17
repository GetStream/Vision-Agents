//go:build integration

package routing

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// HealthSuite routes over a live client whose snapshot refreshes only when nothing asks it
// to, so every health read Redis serves is one the router's walk made.
type HealthSuite struct {
	suite.Suite
	ctx  context.Context
	live *live.Client
}

func TestHealthSuite(t *testing.T) {
	suite.Run(t, new(HealthSuite))
}

func (s *HealthSuite) SetupTest() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	s.ctx = context.Background()
	var err error
	s.live, err = redisready.Retry(address, func() (*live.Client, error) {
		return live.New(live.Options{Address: address, Refresh: -1})
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.live.Close)
}

// healthReads is how many round trips for health of kind every live client of the process
// has made.
func (s *HealthSuite) healthReads(kind string) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	for _, family := range families {
		if family.GetName() != "router_health_reads_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "kind" && label.GetValue() == kind {
					return metric.GetCounter().GetValue()
				}
			}
		}
	}
	return 0
}

// router routes "any" over n providers of the test's own, as llm-smart routes over 21.
func (s *HealthSuite) router(n int) (*Router[*stubProvider], []ProviderConfig) {
	prefix := "health-" + uuid.NewString()[:8]
	config := ModalityConfig{Aliases: map[string]Alias{"any": {Languages: []string{"en"}}}}
	registry := NewRegistry[*stubProvider]()
	for i := range n {
		provider := ProviderConfig{Provider: fmt.Sprintf("%s-%02d", prefix, i), Model: "m", Languages: []string{"en"}}
		config.Providers = append(config.Providers, provider)
		registry.Register(provider.Provider, func(spec Spec) (*stubProvider, error) {
			return &stubProvider{provider: provider.Provider, model: spec.Model}, nil
		})
	}
	router, err := New(Options[*stubProvider]{
		Modality: LLM,
		Config:   config,
		Registry: registry,
		Live:     s.live,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router, config.Providers
}

func (s *HealthSuite) TestAColdRouterReadsAWalkInOneRoundTripAndAWarmOneNotAtAll() {
	router, _ := s.router(21)

	misses := s.healthReads("miss")
	candidates, err := router.Resolve(s.ctx, "any", nil)
	s.Require().NoError(err)
	s.Len(candidates, 21)
	s.Equal(misses+1, s.healthReads("miss"), "the 21 candidates are read in one pipelined round trip")

	for range 5 {
		_, err = router.Resolve(s.ctx, "any", nil)
		s.Require().NoError(err)
	}
	s.Equal(misses+1, s.healthReads("miss"), "a warm snapshot routes without Redis")
}

func (s *HealthSuite) TestTheFirstRefreshReadsEveryModelTheRouterCanBuild() {
	router, _ := s.router(21)

	s.Require().NoError(s.live.Refresh(s.ctx))
	misses := s.healthReads("miss")
	_, err := router.Resolve(s.ctx, "any", nil)
	s.Require().NoError(err)
	s.Equal(misses, s.healthReads("miss"))
}

func (s *HealthSuite) TestAFatalFailureRecordedHereTakesTheModelOutOfTheRankingAtOnce() {
	router, providers := s.router(3)
	candidates, err := router.Resolve(s.ctx, "any", nil)
	s.Require().NoError(err)
	s.Equal(providers[0].Name(), candidates[0].Config.Name(), "unmeasured models keep config order")

	refreshes, misses := s.healthReads("refresh"), s.healthReads("miss")
	router.recorder.Record(providers[0], Stat{
		Owner: Owner{CustomerID: "health-" + uuid.NewString()}, StartedAt: time.Now().UTC(),
		ErrorCode: "account_refused", Fatal: true,
	})

	s.Require().Eventually(func() bool {
		candidates, err := router.Resolve(s.ctx, "any", nil)
		return err == nil && candidates[2].Config.Name() == providers[0].Name() && !candidates[2].Health.Available
	}, 2*time.Second, 10*time.Millisecond, "ranked last as soon as the writer counts it")
	s.Equal(refreshes, s.healthReads("refresh"), "without a refresh")
	s.Equal(misses, s.healthReads("miss"))
}
