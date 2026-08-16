package llmrouter

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubLLM stands in for a real provider so a session can be driven without credentials.
type stubLLM struct {
	emitter *llm.Emitter
	asked   []llm.Request
	// interrupts counts barge-ins, so a session can be checked to forward them.
	interrupts int
	// abandoned is every completion id an interrupt named.
	abandoned []string
	closed    bool
	reasoning bool
}

func newStubLLM() *stubLLM {
	return &stubLLM{emitter: llm.NewEmitter(64)}
}

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Respond(request llm.Request) error {
	s.asked = append(s.asked, request)
	return nil
}

func (s *stubLLM) Interrupt(completionIDs ...string) error {
	s.interrupts++
	s.abandoned = append(s.abandoned, completionIDs...)
	return nil
}

func (s *stubLLM) Events() <-chan llm.Event { return s.emitter.Events() }

func (s *stubLLM) Close() error {
	s.closed = true
	s.emitter.Close()
	return nil
}

func (s *stubLLM) Provider() string { return "stub" }
func (s *stubLLM) Model() string    { return "stub-model" }
func (s *stubLLM) Reasoning() bool  { return s.reasoning }

type LLMRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestLLMRouterSuite(t *testing.T) {
	suite.Run(t, new(LLMRouterSuite))
}

func (s *LLMRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// newRouter routes over the built-in LLM config, which is what a deployment gets when it
// sets no config file.
func (s *LLMRouterSuite) newRouter() *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := New(Options{Config: config[routing.LLM], Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// newSession returns a session over a stub provider, so event handling is the only thing
// under test.
func (s *LLMRouterSuite) newSession() (*Session, *stubLLM) {
	return s.sessionFor(routing.ProviderConfig{
		Provider: "stub",
		Model:    "stub-model",
		Price: routing.Price{
			PerMillionInputTokens:  1,
			PerMillionOutputTokens: 2,
		},
	})
}

func (s *LLMRouterSuite) sessionFor(config routing.ProviderConfig) (*Session, *stubLLM) {
	provider := newStubLLM()
	recorder := routing.NewRecorder(routing.LLM, nil, nil, slog.Default())
	session := newSession(provider, config, routing.Owner{CustomerID: "acme"}, recorder)

	s.T().Cleanup(func() {
		_ = session.Close()
		recorder.Close()
	})
	return session, provider
}

// drain reads the session's forwarded events until the channel closes.
func (s *LLMRouterSuite) drain(session *Session) []llm.Event {
	var events []llm.Event
	for {
		select {
		case event, open := <-session.Events():
			if !open {
				return events
			}
			events = append(events, event)
		case <-time.After(5 * time.Second):
			s.FailNow("timed out draining the session")
			return events
		}
	}
}

func (s *LLMRouterSuite) TestRouterServesTheLLMModality() {
	s.Equal(routing.LLM, s.newRouter().Modality())
}

func (s *LLMRouterSuite) TestEveryShortcutResolvesToAProvider() {
	router := s.newRouter()

	for alias := range router.Config().Aliases {
		candidates, err := router.Resolve(s.ctx, alias, nil)
		s.Require().NoErrorf(err, "alias %s", alias)
		s.NotEmptyf(candidates, "alias %s", alias)
	}
}

func (s *LLMRouterSuite) TestLLMFastResolvesToALowLatencyModel() {
	router := s.newRouter()

	candidates, err := router.Resolve(s.ctx, "llm-fast", nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(candidates)

	for _, candidate := range candidates {
		s.Equal(routing.LowLatency, candidate.Config.Tier,
			"llm-fast must never pick a model chosen for quality over speed")
	}
}

func (s *LLMRouterSuite) TestLowLatencyAndQualityShortcutsPickDifferentModels() {
	router := s.newRouter()

	fast, err := router.Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)
	best, err := router.Resolve(s.ctx, "en-high-accuracy", nil)
	s.Require().NoError(err)

	s.Require().NotEmpty(fast)
	s.Require().NotEmpty(best)
	s.NotEqual(fast[0].Config.Name(), best[0].Config.Name(),
		"asking for speed and asking for quality should not land on the same model")
}

func (s *LLMRouterSuite) TestTheCheapestFastModelIsPreferred() {
	// Ranking is stable within a tier, so config order decides. Flash is first because it
	// is both the cheapest and the quickest of the low-latency models.
	router := s.newRouter()

	candidates, err := router.Resolve(s.ctx, "llm-fast", nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(candidates)

	s.Equal("deepseek/DeepSeek-V4-Flash-0731", candidates[0].Config.Name())
}

func (s *LLMRouterSuite) TestAnUnknownTargetIsRejected() {
	_, err := s.newRouter().Resolve(s.ctx, "no-such-model", nil)

	s.Error(err)
}

func (s *LLMRouterSuite) TestSessionForwardsRequestsToTheProvider() {
	session, provider := s.newSession()

	request := llm.Request{ID: "c1", Messages: []llm.Message{{Role: llm.User, Content: "hi"}}}
	s.Require().NoError(session.Respond(request))

	s.Require().Len(provider.asked, 1)
	s.Equal("c1", provider.asked[0].ID)
}

func (s *LLMRouterSuite) TestSessionForwardsInterruptions() {
	session, provider := s.newSession()

	s.Require().NoError(session.Interrupt())

	s.Equal(1, provider.interrupts)
	s.Empty(provider.abandoned, "an unnamed interrupt abandons everything in flight")
}

func (s *LLMRouterSuite) TestSessionForwardsWhichCompletionToAbandon() {
	// A caller running several completions at once has to be able to abandon only the one
	// whose premise has gone stale.
	session, provider := s.newSession()

	s.Require().NoError(session.Interrupt("task-2"))

	s.Equal([]string{"task-2"}, provider.abandoned)
}

func (s *LLMRouterSuite) TestSessionForwardsEveryProviderEvent() {
	session, provider := s.newSession()

	provider.emitter.Send(llm.CompletionStarted{CompletionID: "c1", At: time.Now()})
	provider.emitter.Send(llm.TextDelta{CompletionID: "c1", Text: "Hi"})
	provider.emitter.Send(llm.CompletionComplete{CompletionID: "c1", Text: "Hi"})
	s.Require().NoError(session.Close())

	events := s.drain(session)

	s.Require().Len(events, 3, "the session observes events without swallowing them")
	s.IsType(llm.CompletionStarted{}, events[0])
	s.IsType(llm.TextDelta{}, events[1])
	s.IsType(llm.CompletionComplete{}, events[2])
}

func (s *LLMRouterSuite) TestSessionIdentityComesFromTheRoutingConfig() {
	// A provider registered under a different name still aggregates under the config's
	// name, so stats and health stay coherent.
	session, _ := s.sessionFor(routing.ProviderConfig{
		Provider: "deepseek",
		Model:    "DeepSeek-V4-Flash-0731",
	})

	s.Equal("deepseek", session.Provider())
	s.Equal("DeepSeek-V4-Flash-0731", session.Model())
}

func (s *LLMRouterSuite) TestSessionExposesThePriceItWillBeBilledAt() {
	session, _ := s.newSession()

	s.EqualValues(3_000_000, session.Price().CostMicros(routing.Usage{
		InputTokens:  1_000_000,
		OutputTokens: 1_000_000,
	}), "a million tokens each way at $1 in and $2 out is $3")
}

func (s *LLMRouterSuite) TestSessionReportsWhetherTheModelThinks() {
	session, provider := s.newSession()
	s.False(session.Reasoning())

	provider.reasoning = true
	s.True(session.Reasoning())
}

func (s *LLMRouterSuite) TestSessionExposesTheUnderlyingProvider() {
	session, provider := s.newSession()

	s.Same(provider, session.LLM(),
		"provider-specific features stay reachable through the session")
}

func (s *LLMRouterSuite) TestClosingTheSessionClosesTheProvider() {
	session, provider := s.newSession()

	s.Require().NoError(session.Close())

	s.True(provider.closed)
	s.Empty(s.drain(session), "the event channel closes so a consumer's range loop ends")
}

func (s *LLMRouterSuite) TestClosingTwiceIsSafe() {
	session, _ := s.newSession()

	s.NoError(session.Close())
	s.NoError(session.Close())
}

func (s *LLMRouterSuite) TestErrorCodeExplainsTheFailure() {
	s.Equal("provider_fatal", errorCode(llm.Error{Fatal: true, Context: "stream"}))
	s.Equal("stream", errorCode(llm.Error{Context: "stream"}))
	s.Equal("provider_error", errorCode(llm.Error{}))
}

func (s *LLMRouterSuite) TestACompletionSettlesOnceEvenIfItIsReportedTwice() {
	// A duplicate completion must not bill twice, so the second one finds nothing in
	// flight and is stamped with now rather than with the original start.
	session, _ := s.newSession()
	started := time.Now().Add(-time.Minute).UTC()

	session.observe(llm.CompletionStarted{CompletionID: "c1", At: started})
	first := session.settle("c1")
	second := session.settle("c1")

	s.Equal(started, first.startedAt)
	s.True(second.startedAt.After(first.startedAt))
}

func (s *LLMRouterSuite) TestAFailureNamingACompletionIsSettledByThatCompletion() {
	// One turn is one row: the error marks the completion, and the completion writes it.
	session, _ := s.newSession()

	session.observe(llm.CompletionStarted{CompletionID: "c1", At: time.Now()})
	s.True(session.fail("c1", "stream"), "the completion was still in flight")

	settled := session.settle("c1")
	s.Equal("stream", settled.errorCode)
}

func (s *LLMRouterSuite) TestOnlyTheFirstFailureExplainsACompletion() {
	session, _ := s.newSession()

	session.observe(llm.CompletionStarted{CompletionID: "c1", At: time.Now()})
	session.fail("c1", "stream")
	session.fail("c1", "provider_fatal")

	s.Equal("stream", session.settle("c1").errorCode)
}

func (s *LLMRouterSuite) TestAFailureForAnUnknownCompletionIsSessionLevel() {
	session, _ := s.newSession()

	s.False(session.fail("never-started", "stream"),
		"a failure the session cannot attribute becomes its own row")
}

func (s *LLMRouterSuite) TestStartFailsWhenNoCandidateCanBeBuilt() {
	// Nothing is registered, so every candidate for the shortcut fails and the caller is
	// told rather than handed a session that cannot answer.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := New(Options{Config: config[routing.LLM], Registry: NewRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	_, err = router.Start(s.ctx, Request{CustomerID: "acme", Target: "llm-fast"})

	s.Error(err)
	s.ErrorContains(err, "llm-fast")
}

func (s *LLMRouterSuite) TestStartFailsOverToTheNextCandidate() {
	// The first candidate cannot be built, so routing moves on rather than giving up.
	// This is what keeps a shortcut working while Gemma is undeployed.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	registry := NewRegistry()
	registry.Register("deepseek", func(routing.Spec) (llm.LLM, error) {
		return nil, errors.New("no credentials")
	})
	registry.Register("openai", func(routing.Spec) (llm.LLM, error) {
		return newStubLLM(), nil
	})

	router, err := New(Options{Config: config[routing.LLM], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "llm-fast"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { session.Close() })

	s.Equal("openai", session.Provider(), "the candidate that could be built served the turn")
}
