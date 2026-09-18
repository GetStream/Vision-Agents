package llmclassifierrouter

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifier"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubClassifier stands in for a real provider so routing can be driven without credentials.
type stubClassifier struct {
	model    string
	answered llmclassifier.Result
	err      error

	asked  []llmclassifier.Request
	closed bool
}

func (s *stubClassifier) Classify(
	_ context.Context, request llmclassifier.Request,
) (llmclassifier.Result, error) {
	s.asked = append(s.asked, request)
	if s.err != nil {
		return llmclassifier.Result{}, s.err
	}
	return s.answered, nil
}

func (s *stubClassifier) Start(context.Context) error { return nil }
func (s *stubClassifier) Close() error                { s.closed = true; return nil }
func (s *stubClassifier) Provider() string            { return "stub" }
func (s *stubClassifier) Model() string               { return s.model }

type ClassifierRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestClassifierRouterSuite(t *testing.T) {
	suite.Run(t, new(ClassifierRouterSuite))
}

func (s *ClassifierRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// config is two providers at the same tier, which is what failover needs something to fall
// back to. The alias is named as the default route is, since that is what a caller who
// names nothing gets.
func (s *ClassifierRouterSuite) config() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "quick", Model: "judge", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
				Price: routing.Price{PerMillionInputTokens: 0.042},
			},
			{
				Provider: "spare", Model: "judge", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
			},
		},
		Aliases: map[string]routing.Alias{
			"classify-fast": {RequireRealtime: true, Tier: routing.LowLatency},
		},
	}
}

func (s *ClassifierRouterSuite) newRouter(
	factories map[string]routing.Factory[llmclassifier.Provider],
) *Router {
	registry := NewRegistry()
	for name, factory := range factories {
		registry.Register(name, factory)
	}

	router, err := New(Options{
		Config:   s.config(),
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// allowed is an answer to one noul, which is the shape a guardrail asks for.
func allowed(probability float64) llmclassifier.Result {
	return llmclassifier.Result{
		Model: "judge-1.0",
		Answers: map[string]llmclassifier.Answer{
			"violates": {Type: llmclassifier.TypeNoul, Yes: probability},
		},
		Usage: llmclassifier.Usage{InputTokens: 312},
	}
}

func (s *ClassifierRouterSuite) TestATargetResolvesAndAnswers() {
	provider := &stubClassifier{model: "judge", answered: allowed(0.92)}
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "classify-fast"})
	s.Require().NoError(err)

	answered, err := session.Classify(s.ctx, llmclassifier.Request{
		State: "how do I make a pizza",
		Questions: map[string]llmclassifier.Question{
			"violates": llmclassifier.Noul("Is this off topic?", "", ""),
		},
	})
	s.Require().NoError(err)

	s.Equal("quick", session.Provider())
	s.Equal("judge", session.Model())
	s.InDelta(0.92, answered.Answers["violates"].Yes, 0.001)
	s.Require().Len(provider.asked, 1)
	s.Equal("how do I make a pizza", provider.asked[0].State)
}

func (s *ClassifierRouterSuite) TestNamingNoTargetTakesTheFastRoute() {
	// A judgement on the live path of a conversation is wanted quickly above all else, so
	// a caller who says nothing gets the low-latency tier rather than a refusal.
	provider := &stubClassifier{model: "judge", answered: allowed(0.1)}
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme"})
	s.Require().NoError(err)

	s.Equal("quick", session.Provider())
}

func (s *ClassifierRouterSuite) TestAProviderWithoutAKeyDropsToTheNextCandidate() {
	// A provider is built when the session opens, so a deployment holding a key for one
	// of two still judges rather than refusing every turn it was meant to screen.
	spare := &stubClassifier{model: "judge"}
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) {
			return nil, errors.New("QUICK_API_KEY is required")
		},
		"spare": func(routing.Spec) (llmclassifier.Provider, error) { return spare, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "classify-fast"})
	s.Require().NoError(err)

	s.Equal("spare", session.Provider())
}

func (s *ClassifierRouterSuite) TestNoProviderAtAllSaysWhatEachOneComplainedAbout() {
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) {
			return nil, errors.New("QUICK_API_KEY is required")
		},
		"spare": func(routing.Spec) (llmclassifier.Provider, error) {
			return nil, errors.New("SPARE_API_KEY is required")
		},
	})

	_, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "classify-fast"})

	s.Require().Error(err)
	s.ErrorContains(err, "QUICK_API_KEY")
	s.ErrorContains(err, "SPARE_API_KEY")
}

func (s *ClassifierRouterSuite) TestAJudgementThatFailedIsReportedRatherThanRetriedElsewhere() {
	// Failover is start-time only, as it is for a model: the caller is waiting on a reply
	// this judgement is holding, and a second provider's latency on top of the first one's
	// failure is a longer silence than letting the turn's own policy decide.
	provider := &stubClassifier{model: "judge", err: errors.New("rate limited")}
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "classify-fast"})
	s.Require().NoError(err)

	_, err = session.Classify(s.ctx, llmclassifier.Request{
		State:     "anything",
		Questions: map[string]llmclassifier.Question{"violates": llmclassifier.Noul("Off topic?", "", "")},
	})

	s.ErrorContains(err, "rate limited")
	s.Equal("quick", session.Provider(), "the session stays on the provider it chose")
}

func (s *ClassifierRouterSuite) TestClosingASessionClosesTheProvider() {
	provider := &stubClassifier{model: "judge"}
	router := s.newRouter(map[string]routing.Factory[llmclassifier.Provider]{
		"quick": func(routing.Spec) (llmclassifier.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "classify-fast"})
	s.Require().NoError(err)

	s.Require().NoError(session.Close())
	s.Require().NoError(session.Close())
	s.True(provider.closed)
}

func (s *ClassifierRouterSuite) TestTheDefaultRegistryHasEveryProviderTheConfigDeclares() {
	// A provider declared in router.yaml with no factory behind it is a candidate that
	// can never be chosen, and the router only says so once a turn is already waiting.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	section, ok := config[routing.LLMClassifier]
	s.Require().True(ok, "llm_classifier is a routed modality")

	registry := DefaultRegistry()
	for _, provider := range section.Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is declared but nothing can build it", provider.Name())
	}
}

func (s *ClassifierRouterSuite) TestTheDefaultConfigOffersTheRouteAGuardrailAsksFor() {
	// The default route is what a guardrail that names no target resolves, so a config
	// without it would refuse every guardrail rather than one that asked for something.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	section := config[routing.LLMClassifier]
	s.Contains(section.Aliases, "classify-fast")
}
