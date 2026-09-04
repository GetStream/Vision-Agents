package llmrouter

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubLLM stands in for a real provider so a session can be driven without credentials.
type stubLLM struct {
	asked  []llm.ResponseParams
	closed bool
	// scripts are the responses handed out, newest last, so a test can drive one after
	// the session has returned it.
	scripts []*llmtest.Script
	// capabilities is what this stub claims to accept.
	capabilities llm.Capabilities
}

func newStubLLM() *stubLLM { return &stubLLM{} }

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	s.asked = append(s.asked, params)

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   s.Provider(),
		Model:      s.Model(),
	})
	s.scripts = append(s.scripts, script)
	return script.Stream(), nil
}

// script is the response handed out for the nth request, so a test can write it.
func (s *stubLLM) script(n int) *llmtest.Script { return s.scripts[n] }

func (s *stubLLM) Close() error {
	s.closed = true
	for _, script := range s.scripts {
		script.Done()
	}
	return nil
}

func (s *stubLLM) Provider() string               { return "stub" }
func (s *stubLLM) Model() string                  { return "stub-model" }
func (s *stubLLM) Capabilities() llm.Capabilities { return s.capabilities }

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

	_, err := session.Create(s.ctx, llm.ResponseParams{
		ID:    "c1",
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
	})
	s.Require().NoError(err)

	s.Require().Len(provider.asked, 1)
	s.Equal("c1", provider.asked[0].ID)
}

func (s *LLMRouterSuite) TestClosingAResponseAbandonsOnlyThatOne() {
	// A caller running several responses at once has to be able to abandon only the one
	// whose premise has gone stale.
	session, provider := s.newSession()

	first, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	second, err := session.Create(s.ctx, llm.ResponseParams{ID: "c2", Input: prompt()})
	s.Require().NoError(err)

	s.Require().NoError(first.Close())
	provider.script(1).OutputText("still going")
	provider.script(1).Done()

	s.Equal(llm.StatusCancelled, drain(first).Status)
	s.Equal("still going", drain(second).OutputText,
		"abandoning one response leaves the other alone")
}

// prompt is the smallest input a request can carry.
func prompt() []llm.Message {
	return []llm.Message{{Role: llm.User, Content: "hi"}}
}

// drain reads a stream to the end and returns what it settled as.
func drain(stream *llm.Stream) llm.Response {
	for stream.Next() {
	}
	return stream.Response()
}

func (s *LLMRouterSuite) TestSessionForwardsEveryProviderEvent() {
	session, provider := s.newSession()

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)

	provider.script(0).OutputText("Hi")
	provider.script(0).Done()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}

	s.Require().Len(events, 3, "the session observes events without swallowing them")
	s.IsType(llm.ResponseCreated{}, events[0])
	s.IsType(llm.OutputTextDelta{}, events[1])
	s.IsType(llm.ResponseCompleted{}, events[2])
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

func (s *LLMRouterSuite) TestSessionReportsWhatTheModelAccepts() {
	session, provider := s.newSession()
	s.Empty(session.Capabilities().ReasoningEfforts)

	provider.capabilities = llm.Capabilities{ReasoningEfforts: []string{"low"}}
	s.Equal([]string{"low"}, session.Capabilities().ReasoningEfforts)
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
}

func (s *LLMRouterSuite) TestClosingTwiceIsSafe() {
	session, _ := s.newSession()

	s.NoError(session.Close())
	s.NoError(session.Close())
}

func (s *LLMRouterSuite) TestErrorCodeExplainsTheFailure() {
	s.Equal("provider_error", errorCode(llm.Response{Status: llm.StatusFailed}))
	s.Empty(errorCode(llm.Response{Status: llm.StatusCompleted}))
	s.Empty(errorCode(llm.Response{Status: llm.StatusCancelled}),
		"a response the caller abandoned is not a provider failure")
}

func (s *LLMRouterSuite) TestAnAbandonedResponseIsStillBilled() {
	// What it generated before being cut off was generated all the same.
	session, provider := s.newSession()

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 4})

	s.Require().True(stream.Next())
	s.Require().True(stream.Next())
	s.Require().NoError(stream.Close())

	response := drain(stream)
	s.Equal(llm.StatusCancelled, response.Status)
	s.EqualValues(4, response.Usage.OutputTokens)
}

func (s *LLMRouterSuite) TestAFailedResponseStillSettles() {
	// One turn is one row: the failure is carried on the response rather than written as
	// a row of its own.
	session, provider := s.newSession()

	stream, err := session.Create(s.ctx, llm.ResponseParams{ID: "c1", Input: prompt()})
	s.Require().NoError(err)
	provider.script(0).Fail(errors.New("the model is down"), "stream")

	s.Equal(llm.StatusFailed, drain(stream).Status)
	s.ErrorContains(stream.Err(), "the model is down")
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
	registry.Register("deepseek", func(routing.Spec) (Provider, error) {
		return nil, errors.New("no credentials")
	})
	registry.Register("openai", func(routing.Spec) (Provider, error) {
		return Started[llm.LLM](newStubLLM(), nil)
	})

	router, err := New(Options{Config: config[routing.LLM], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "llm-fast"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { session.Close() })

	s.Equal("openai", session.Provider(), "the candidate that could be built served the turn")
}
