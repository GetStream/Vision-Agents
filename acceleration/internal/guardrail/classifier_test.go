package guardrail

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifier"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifierrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubClassifier answers with a fixed probability, so what the threshold does can be tested
// without a model having an opinion about it.
type stubClassifier struct {
	probability float64
	err         error

	asked []llmclassifier.Request
}

func (s *stubClassifier) Classify(
	_ context.Context, request llmclassifier.Request,
) (llmclassifier.Result, error) {
	s.asked = append(s.asked, request)
	if s.err != nil {
		return llmclassifier.Result{}, s.err
	}
	return llmclassifier.Result{
		Model: "stub-1.0",
		Answers: map[string]llmclassifier.Answer{
			violates: {Type: llmclassifier.TypeNoul, Yes: s.probability},
		},
		Usage: llmclassifier.Usage{InputTokens: 120},
	}, nil
}

func (s *stubClassifier) Start(context.Context) error { return nil }
func (s *stubClassifier) Close() error                { return nil }
func (s *stubClassifier) Provider() string            { return "stub" }
func (s *stubClassifier) Model() string               { return "judge" }

type ClassifierSuite struct {
	suite.Suite
	ctx context.Context
}

func TestClassifierSuite(t *testing.T) {
	suite.Run(t, new(ClassifierSuite))
}

func (s *ClassifierSuite) SetupTest() {
	s.ctx = context.Background()
}

// router routes every check to one stub.
func (s *ClassifierSuite) router(stub *stubClassifier) *llmclassifierrouter.Router {
	registry := llmclassifierrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmclassifier.Provider, error) {
		return stub, nil
	})

	router, err := llmclassifierrouter.New(llmclassifierrouter.Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{{
				Provider: "stub", Model: "judge", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
			}},
			Aliases: map[string]routing.Alias{
				"classify-fast": {RequireRealtime: true, Tier: routing.LowLatency},
			},
		},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// guardrail builds a classifier guardrail over the stub, at the given threshold.
func (s *ClassifierSuite) guardrail(stub *stubClassifier, threshold float64) Guardrail {
	screening, err := New(s.ctx, Policy{
		Kind:      KindClassifier,
		Mode:      ModeParallel,
		Threshold: threshold,
		Refusal:   "I can only help with questions about Stream.",
		Text:      "Only answer questions about Stream's SDKs.",
	}, Deps{
		Owner:      routing.Owner{CustomerID: "acme"},
		Classifier: s.router(stub),
		Logger:     slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = screening.Close() })
	return screening
}

func (s *ClassifierSuite) TestTheThresholdIsWhatDecides() {
	// The same answer either side of the same threshold, because the number in the file is
	// the whole of the policy's strictness and nothing else should be deciding.
	tests := []struct {
		name        string
		probability float64
		threshold   float64
		allowed     bool
	}{
		{name: "well under the threshold", probability: 0.1, threshold: 0.6, allowed: true},
		{name: "just under the threshold", probability: 0.59, threshold: 0.6, allowed: true},
		{name: "exactly at the threshold", probability: 0.6, threshold: 0.6, allowed: false},
		{name: "well over the threshold", probability: 0.97, threshold: 0.6, allowed: false},
		{name: "a strict policy lets through what a lax one blocks", probability: 0.5, threshold: 0.9, allowed: true},
		{name: "a lax policy blocks what a strict one lets through", probability: 0.5, threshold: 0.2, allowed: false},
	}

	for _, test := range tests {
		s.Run(test.name, func() {
			stub := &stubClassifier{probability: test.probability}
			verdict, err := s.guardrail(stub, test.threshold).
				Check(s.ctx, "turn-1", "how do I make a pizza")
			s.Require().NoError(err)

			s.Equal(test.allowed, verdict.Allowed)
			s.InDelta(test.probability, verdict.Probability, 0.001)
		})
	}
}

func (s *ClassifierSuite) TestThePolicyAndTheMessageAreAskedAboutAsTwoThings() {
	// The policy travels as state rather than pasted into the question, which is what lets
	// the question point at each of them by name. Pasting the policy into the question
	// would let a message that looked like policy prose rewrite what is being asked.
	stub := &stubClassifier{probability: 0.1}

	_, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "how do I install stream-chat-react")
	s.Require().NoError(err)

	s.Require().Len(stub.asked, 1)
	state, ok := stub.asked[0].State.(map[string]string)
	s.Require().True(ok)
	s.Equal("Only answer questions about Stream's SDKs.", state["policy"])
	s.Equal("how do I install stream-chat-react", state["message"])

	asked := stub.asked[0].Questions[violates]
	s.Equal(llmclassifier.TypeNoul, asked.Type)
	s.Contains(asked.Instructions, "`policy`")
	s.Contains(asked.Instructions, "`message`")
}

func (s *ClassifierSuite) TestAReasonIsReportedButTheRefusalIsThePolicysOwn() {
	// What the caller hears is the line the policy wrote. The reason is for the log: one
	// that quoted the policy back would be a map of how to get around it.
	stub := &stubClassifier{probability: 0.95}
	screening := s.guardrail(stub, 0.6)

	verdict, err := screening.Check(s.ctx, "turn-1", "why is the sky blue")
	s.Require().NoError(err)

	s.False(verdict.Allowed)
	s.NotEmpty(verdict.Reason)
	s.Empty(verdict.Refusal, "only a webhook has something better to say than the policy")
	s.Equal("I can only help with questions about Stream.", screening.Policy().Refusal)
}

func (s *ClassifierSuite) TestAClassifierThatFailedIsReportedRatherThanReadAsAVerdict() {
	// A failure read as either verdict invents a decision nobody made. What to do about it
	// is the agent's to decide, and it can only decide if it is told.
	stub := &stubClassifier{err: errors.New("rate limited")}

	_, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "anything")

	s.ErrorContains(err, "rate limited")
}

func (s *ClassifierSuite) TestAPolicyWithNothingToRouteItIsRefusedWhenBuilt() {
	// Refused here rather than at the first turn: a guardrail that starts and then cannot
	// reach its classifier allows every turn, which is an unguarded agent that looks
	// guarded.
	_, err := New(s.ctx, Policy{
		Kind: KindClassifier, Mode: ModeParallel, Threshold: 0.6,
		Refusal: "No.", Text: "Only Stream questions.",
	}, Deps{Owner: routing.Owner{CustomerID: "acme"}, Logger: slog.New(slog.DiscardHandler)})

	s.ErrorContains(err, "llm_classifier")
}
