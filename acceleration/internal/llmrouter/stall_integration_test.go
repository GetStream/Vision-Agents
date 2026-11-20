//go:build integration

package llmrouter

import (
	"context"
	"maps"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// stallHedge is how long a flow decision may say nothing before these tests ask it again.
const stallHedge = 50 * time.Millisecond

// stallWindow is how long health remembers a stall: the shortest window Redis can expire.
const stallWindow = 2 * time.Second

// StallSuite races flow decisions in real time against provider health kept in Redis, and
// reads back what that health then makes of the provider that stalled.
type StallSuite struct {
	suite.Suite
	ctx  context.Context
	live *live.Client
	// stalling and steady are provider names unique to each test, since health is kept per
	// provider and outlives the test.
	stalling, steady string
}

func TestStallSuite(t *testing.T) {
	suite.Run(t, new(StallSuite))
}

func (s *StallSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.ctx = context.Background()
	client, err := redisready.Retry(address, func() (*live.Client, error) { return live.New(live.Options{Address: address, Window: stallWindow}) })
	s.Require().NoError(err)
	s.live = client
}

func (s *StallSuite) SetupTest() {
	unique := uuid.NewString()
	s.stalling, s.steady = "stalling-"+unique, "steady-"+unique
}

// router is a target preferring the stalling provider, with the steady one behind it, whose
// flow decisions are hedged after stallHedge.
func (s *StallSuite) router() (*rig, *Router) {
	r := &rig{headers: map[string]time.Duration{}, failing: map[string]error{}, scripts: map[string][]*llmtest.Script{}}
	registry := NewRegistry()
	var config routing.ModalityConfig
	var members []string
	for _, name := range []string{s.stalling, s.steady} {
		config.Providers = append(config.Providers, routing.ProviderConfig{Provider: name, Model: "model", Languages: []string{"en"}})
		members = append(members, name+"/model")
		registry.Register(name, func(spec routing.Spec) (Provider, error) { return &pace{rig: r, name: name, model: spec.Model}, nil })
	}
	config.Aliases = map[string]routing.Alias{"flow": {Prefer: members[0], Only: members}}
	router, err := New(Options{Config: config, Registry: registry, Live: s.live, FlowHedge: stallHedge})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return r, router
}

// start opens a session on the target, as a call opens its flow controller.
func (s *StallSuite) start(r *rig, router *Router) {
	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "flow"})
	s.Require().NoError(err)
	r.session = session
	s.T().Cleanup(func() { s.Require().NoError(session.Close()) })
}

// decide asks the session for a flow decision, which returns once one attempt has said
// something.
func (s *StallSuite) decide(r *rig) <-chan *llm.Stream {
	params := r.params()
	params.Purpose = flowPurpose
	answered := make(chan *llm.Stream, 1)
	go func() {
		stream, err := r.session.Create(s.ctx, params)
		s.NoError(err)
		answered <- stream
	}()
	return answered
}

// asked waits for a provider to have been asked n requests.
func (s *StallSuite) asked(r *rig, name string, n int) {
	s.Require().Eventually(func() bool { return r.requests(name) == n }, 3*time.Second, time.Millisecond, name+" was never asked")
}

// answer has a provider answer the latest request it was asked, and reads it out.
func (s *StallSuite) answer(r *rig, name string, answered <-chan *llm.Stream) llm.Response {
	r.mu.Lock()
	script := r.scripts[name][len(r.scripts[name])-1]
	r.mu.Unlock()
	script.OutputText(`{"disposition":"respond"}`)
	stream := <-answered
	script.Done()
	return drain(stream)
}

// available waits for the provider's health to settle on available or not.
func (s *StallSuite) available(name string, want bool) {
	s.Require().Eventually(func() bool {
		health, err := s.live.Health(s.ctx, string(routing.LLM), name, "model")
		s.Require().NoError(err)
		return health.Available == want
	}, 2*stallWindow, 10*time.Millisecond, "%s never became available=%t", name, want)
}

// outcomes is what router_requests_total has counted for the provider, by outcome, read as
// /metrics serves it. The provider is the test's own, so the counts are the test's alone.
func (s *StallSuite) outcomes(provider string) map[string]float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	counted := map[string]float64{}
	for _, family := range families {
		if family.GetName() != "router_requests_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["provider"] == provider {
				counted[labels["outcome"]] += metric.GetCounter().GetValue()
			}
		}
	}
	return counted
}

// counted waits for router_requests_total to have counted want for the provider, and fails
// with what it counted instead.
func (s *StallSuite) counted(provider string, want map[string]float64, why string) {
	for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		if maps.Equal(s.outcomes(provider), want) {
			return
		}
	}
	s.Require().Equal(want, s.outcomes(provider), why)
}

// stall has the stalling provider say nothing until the steady one answers a decision.
func (s *StallSuite) stall(r *rig) {
	answered := s.decide(r)
	s.asked(r, s.steady, r.requests(s.steady)+1)
	s.Equal(s.steady, s.answer(r, s.steady, answered).Provider)
}

// stalls has the stalling provider lose as many decisions in silence as health needs to judge
// it, all in one session, which keeps asking the provider it started on.
func (s *StallSuite) stalls(r *rig) {
	for range live.DefaultMinRequests {
		s.stall(r)
	}
}

func (s *StallSuite) TestOneStallOnFreshHistoryLeavesTheProviderAvailable() {
	r, router := s.router()
	s.start(r, router)
	s.stall(r)

	// Errors rather than requests: the counters are written together but not atomically.
	var health live.Health
	s.Require().Eventually(func() bool {
		var err error
		health, err = s.live.Health(s.ctx, string(routing.LLM), s.stalling, "model")
		s.Require().NoError(err)
		return health.Errors == 1
	}, 2*stallWindow, 10*time.Millisecond, "the stall was never counted")
	s.True(health.Available, "one stall is not enough to take a provider out for the whole shard")

	s.start(r, router)
	s.Equal(s.stalling, r.session.Provider(), "the next session still starts on the preferred provider")
}

func (s *StallSuite) TestAProviderThatSaysNothingUntilItLosesIsRoutedAround() {
	r, router := s.router()
	s.start(r, router)
	s.stalls(r)
	s.available(s.stalling, false)

	s.start(r, router)
	s.Equal(s.steady, r.session.Provider(), "the next session starts on the provider that answers")
	before := r.requests(s.steady)
	answered := s.decide(r)
	s.asked(r, s.steady, before+1)
	// Nothing tells a hedge that found no candidate from one not yet due, so this waits out
	// the hedge delay. hedgeCandidates leaving out an unavailable provider is pinned without
	// a clock in TestAHedgeGoesToADifferentModelBeforeADifferentProvider.
	time.Sleep(4 * stallHedge)
	s.Equal(live.DefaultMinRequests, r.requests(s.stalling), "a provider found unavailable is not asked as a hedge either")
	s.Equal(s.steady, s.answer(r, s.steady, answered).Provider)
}

func (s *StallSuite) TestAStalledProviderIsAskedFirstAgainOnceItsWindowPasses() {
	r, router := s.router()
	s.start(r, router)
	s.stalls(r)
	s.available(s.stalling, false)

	s.available(s.stalling, true)
	s.Require().NoError(s.live.Refresh(s.ctx), "the router routes by its snapshot, which hears of the forgotten stall at its next refresh")
	s.start(r, router)
	s.Equal(s.stalling, r.session.Provider(), "the preferred provider is tried again once its stall is forgotten")
}

func (s *StallSuite) TestAProviderThatSaidSomethingBeforeItLostIsNotPenalised() {
	r, router := s.router()
	s.start(r, router)
	answered := s.decide(r)
	s.asked(r, s.stalling, 1)
	r.script(s.stalling).ReasoningText("Thinking")
	s.asked(r, s.steady, 1)
	s.Equal(s.steady, s.answer(r, s.steady, answered).Provider)

	s.Require().Eventually(func() bool { return r.called()[s.stalling] != "" }, 3*time.Second, time.Millisecond)
	s.Equal(routing.ErrorCancelled, r.called()[s.stalling],
		"a provider that had begun answering lost a race, which is no fault of its own")
}

func (s *StallSuite) TestAProviderWhoseCallerHungUpIsNotPenalised() {
	r, router := s.router()
	s.start(r, router)
	ctx, hangUp := context.WithCancel(s.ctx)
	params := r.params()
	params.Purpose = flowPurpose
	done := make(chan error, 1)
	go func() {
		_, err := r.session.Create(ctx, params)
		done <- err
	}()
	s.asked(r, s.steady, 1)
	hangUp()
	s.ErrorIs(<-done, context.Canceled)

	s.Require().Eventually(func() bool { return len(r.called()) == 2 }, 3*time.Second, time.Millisecond)
	s.Equal(map[string]string{s.stalling: routing.ErrorCancelled, s.steady: routing.ErrorCancelled}, r.called(),
		"a decision nobody waits on any more says nothing about either provider")
}

func (s *StallSuite) TestAProviderThatLosesInSilenceIsCountedAsAHedgeLostNotATimeout() {
	r, router := s.router()
	s.start(r, router)
	s.stall(r)

	s.counted(s.stalling, map[string]float64{"hedge_lost": 1}, "a hedge outran it, which is what a hedge is for")
	s.counted(s.steady, map[string]float64{"ok": 1}, "the hedge served the decision")
	s.Equal(routing.ErrorTimeout, r.called()[s.stalling], "it is still reported as a stall")
	var health live.Health
	s.Require().Eventually(func() bool {
		var err error
		health, err = s.live.Health(s.ctx, string(routing.LLM), s.stalling, "model")
		s.Require().NoError(err)
		return health.Errors == 1
	}, 2*stallWindow, 10*time.Millisecond, "and still counted against its provider's health")
}

func (s *StallSuite) TestAProviderStillBeingAskedWhenItLosesIsCountedAsAHedgeLost() {
	r, router := s.router()
	// Its headers never come, so the request is still being made when the hedge answers.
	r.slow(s.stalling, time.Hour)
	s.start(r, router)
	answered := s.decide(r)
	s.asked(r, s.steady, 1)
	s.Equal(s.steady, s.answer(r, s.steady, answered).Provider)

	s.counted(s.stalling, map[string]float64{"hedge_lost": 1}, "a request cut off unanswered lost its hedge as a stream does")
	s.Equal(routing.ErrorTimeout, r.called()[s.stalling])
}

func (s *StallSuite) TestAProviderThatRunsPastADeadlineBeforeItsHedgeAnswersIsCountedAsATimeout() {
	r, router := s.router()
	s.start(r, router)
	answered := s.decide(r)
	s.asked(r, s.stalling, 1)
	s.asked(r, s.steady, 1)
	// It is read out, and so recorded, only once the hedge has won, as a loser is.
	r.script(s.stalling).TimeOut()
	s.Equal(s.steady, s.answer(r, s.steady, answered).Provider)

	s.counted(s.stalling, map[string]float64{"timeout": 1}, "its provider ran past a deadline before it lost")
	s.Equal(routing.ErrorTimeout, r.called()[s.stalling])
}
