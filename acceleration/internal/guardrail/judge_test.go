package guardrail

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubModel answers every request with whatever the test wrote down, so what the judge
// makes of a reply can be tested without a model deciding what the reply is.
type stubModel struct {
	says string
	err  error

	asked []llm.ResponseParams
}

func (m *stubModel) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	m.asked = append(m.asked, params)
	if m.err != nil {
		return nil, m.err
	}

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID, Provider: m.Provider(), Model: m.Model(),
	})
	script.OutputText(m.says)
	script.Usage(llm.Usage{InputTokens: 220, OutputTokens: 18})
	script.Done()
	return script.Stream(), nil
}

func (m *stubModel) Start(context.Context) error    { return nil }
func (m *stubModel) Close() error                   { return nil }
func (m *stubModel) Provider() string               { return "stub" }
func (m *stubModel) Model() string                  { return "judge" }
func (m *stubModel) Capabilities() llm.Capabilities { return llm.Capabilities{} }

type JudgeSuite struct {
	suite.Suite
	ctx context.Context
}

func TestJudgeSuite(t *testing.T) {
	suite.Run(t, new(JudgeSuite))
}

func (s *JudgeSuite) SetupTest() {
	s.ctx = context.Background()
}

// router routes every judgement to one stub.
func (s *JudgeSuite) router(stub *stubModel) *llmrouter.Router {
	registry := llmrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		return stub, nil
	})

	router, err := llmrouter.New(llmrouter.Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{{
				Provider: "stub", Model: "judge", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
			}},
			Aliases: map[string]routing.Alias{
				"llm-fast": {RequireRealtime: true, Tier: routing.LowLatency},
			},
		},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// guardrail builds an llm-judge guardrail over the stub, at the given threshold.
func (s *JudgeSuite) guardrail(stub *stubModel, threshold float64) Guardrail {
	screening, err := New(s.ctx, Policy{
		Kind:      KindLLM,
		Mode:      ModeParallel,
		Threshold: threshold,
		Refusal:   "I can only help with questions about Stream.",
		Text:      "Only answer questions about Stream's SDKs.",
	}, Deps{
		Owner:  routing.Owner{CustomerID: "acme"},
		LLM:    s.router(stub),
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = screening.Close() })
	return screening
}

func (s *JudgeSuite) TestTheThresholdMeansTheSameThingItMeansToAClassifier() {
	// A judge is asked for a probability rather than a verdict so that a policy can be
	// moved between llm and lcm without the number in the file changing what
	// it means. These are the classifier's own cases, answered by a model instead.
	tests := []struct {
		name      string
		says      string
		threshold float64
		allowed   bool
	}{
		{name: "plainly permitted", says: `{"violates":0.02,"reason":"an SDK question"}`, threshold: 0.6, allowed: true},
		{name: "just under the threshold", says: `{"violates":0.59,"reason":"unclear"}`, threshold: 0.6, allowed: true},
		{name: "exactly at the threshold", says: `{"violates":0.6,"reason":"unclear"}`, threshold: 0.6, allowed: false},
		{name: "plainly forbidden", says: `{"violates":0.98,"reason":"about cooking"}`, threshold: 0.6, allowed: false},
		{name: "a strict policy lets through what a lax one blocks", says: `{"violates":0.5,"reason":"unclear"}`, threshold: 0.9, allowed: true},
	}

	for _, test := range tests {
		s.Run(test.name, func() {
			stub := &stubModel{says: test.says}
			verdict, err := s.guardrail(stub, test.threshold).
				Check(s.ctx, "turn-1", "how do I make a pizza")
			s.Require().NoError(err)

			s.Equal(test.allowed, verdict.Allowed)
		})
	}
}

func (s *JudgeSuite) TestTheJudgeIsToldToScreenRatherThanToAnswer() {
	// The policy and the request are labelled and the model is told which job it has. A
	// judge that answered the request instead would report every turn as permitted, since
	// it would be judging its own answer.
	stub := &stubModel{says: `{"violates":0.05,"reason":"an SDK question"}`}

	_, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "how do I install stream-chat-react")
	s.Require().NoError(err)

	s.Require().Len(stub.asked, 1)
	asked := stub.asked[0]
	s.Contains(asked.Instructions, "You do not answer them")
	s.Require().Len(asked.Input, 1)
	s.Contains(asked.Input[0].Content, "Only answer questions about Stream's SDKs.")
	s.Contains(asked.Input[0].Content, "how do I install stream-chat-react")
	s.Equal(llm.FormatJSONObject, asked.Text.Format,
		"a judge asked for JSON in prose is a judge whose answer will not parse")
	s.Positive(asked.MaxOutputTokens, "a number and a sentence needs no essay")
}

func (s *JudgeSuite) TestTheJudgesOwnWordsAreTheReasonWhenItGivesThem() {
	stub := &stubModel{says: `{"violates":0.93,"reason":"it asks for a recipe"}`}

	verdict, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "how do I make a pizza")
	s.Require().NoError(err)

	s.False(verdict.Allowed)
	s.Equal("it asks for a recipe", verdict.Reason)
	s.Empty(verdict.Refusal, "what the caller hears is the policy's own line")
}

func (s *JudgeSuite) TestARefusalWithNoReasonStillSaysSomethingToTheLog() {
	stub := &stubModel{says: `{"violates":0.93}`}

	verdict, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "how do I make a pizza")
	s.Require().NoError(err)

	s.False(verdict.Allowed)
	s.NotEmpty(verdict.Reason)
}

func (s *JudgeSuite) TestAnAnswerThatIsNotAVerdictIsAFailureRatherThanAGuess() {
	// Reading prose as either verdict invents a decision nobody made. The agent decides
	// what to do about a check it could not get, and it can only decide if it is told.
	for _, said := range []string{
		"I cannot judge that.",
		"",
		`{"violates":`,
	} {
		s.Run(said, func() {
			stub := &stubModel{says: said}

			_, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "anything")

			s.ErrorContains(err, "not a verdict")
		})
	}
}

func (s *JudgeSuite) TestAModelThatFailedIsReportedRatherThanReadAsAVerdict() {
	stub := &stubModel{err: errors.New("overloaded")}

	_, err := s.guardrail(stub, 0.6).Check(s.ctx, "turn-1", "anything")

	s.ErrorContains(err, "overloaded")
}

func (s *JudgeSuite) TestAPolicyWithNoModelToJudgeItIsRefusedWhenBuilt() {
	// Refused here rather than at the first turn, for the same reason a classifier policy
	// is: a guardrail that starts and cannot reach its judge allows everything while
	// looking like it is screening.
	_, err := New(s.ctx, Policy{
		Kind: KindLLM, Mode: ModeParallel, Threshold: 0.6,
		Refusal: "No.", Text: "Only Stream questions.",
	}, Deps{Owner: routing.Owner{CustomerID: "acme"}, Logger: slog.New(slog.DiscardHandler)})

	s.ErrorContains(err, "llm")
}
