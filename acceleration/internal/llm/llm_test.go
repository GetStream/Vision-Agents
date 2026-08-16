package llm

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

var errUnauthorized = errors.New("unauthorized")

type LLMSuite struct {
	suite.Suite
}

func TestLLMSuite(t *testing.T) {
	suite.Run(t, new(LLMSuite))
}

func (s *LLMSuite) TestCompletionGeneratesAnIDWhenTheCallerHasNone() {
	first := NewCompletion("")
	second := NewCompletion("")

	s.NotEmpty(first.ID)
	s.NotEqual(first.ID, second.ID, "two completions must not share an id")
	s.Equal("mine", NewCompletion("mine").ID, "a caller's id is kept")
}

func (s *LLMSuite) TestCompletionAssemblesTheAnswerFromItsDeltas() {
	completion := NewCompletion("c1")
	completion.Delta("Hello")
	completion.Delta(", world")

	s.Equal("Hello, world", completion.Text())
	s.Equal("Hello, world", completion.Complete("openai", "gpt-4o-mini", false).Text)
}

func (s *LLMSuite) TestCompletionKeepsReasoningOutOfTheAnswer() {
	// Thinking must never be spoken as the reply, so it is tracked but not assembled.
	completion := NewCompletion("c1")
	completion.Reasoning("The user greeted me, so")
	completion.Delta("Hi there")

	s.Equal("Hi there", completion.Complete("deepseek", "DeepSeek-V4-Flash-0731", false).Text)
}

func (s *LLMSuite) TestCompletionNumbersDeltasInOrderAcrossBothKinds() {
	completion := NewCompletion("c1")

	thinking := completion.Reasoning("hmm")
	first := completion.Delta("Hi")
	second := completion.Delta(" there")

	s.Equal(0, thinking.Index)
	s.Equal(1, first.Index)
	s.Equal(2, second.Index, "reasoning and text share one sequence, so order is total")
	s.Equal("c1", second.CompletionID)
}

func (s *LLMSuite) TestCompletionMeasuresTimeToFirstToken() {
	completion := NewCompletion("c1")
	time.Sleep(20 * time.Millisecond)
	completion.Delta("Hi")
	time.Sleep(20 * time.Millisecond)
	completion.Delta(" there")

	complete := completion.Complete("openai", "gpt-4o-mini", false)

	s.GreaterOrEqual(complete.TimeToFirstTokenMs, 15.0, "the wait for the first token")
	s.Less(complete.TimeToFirstTokenMs, complete.CompletionTimeMs,
		"the first token arrives before the last one")
}

func (s *LLMSuite) TestReasoningCountsTowardsTimeToFirstToken() {
	// A reasoning model is working while it thinks, so the caller's wait ends there.
	completion := NewCompletion("c1")
	time.Sleep(20 * time.Millisecond)
	completion.Reasoning("thinking")
	time.Sleep(20 * time.Millisecond)
	completion.Delta("Hi")

	complete := completion.Complete("deepseek", "DeepSeek-V4-Flash-0731", false)

	s.Less(complete.TimeToFirstTokenMs, 35.0, "thinking is the first sign of life, not the answer")
}

func (s *LLMSuite) TestCompletionWithNoOutputReportsNoTimeToFirstToken() {
	complete := NewCompletion("c1").Complete("openai", "gpt-4o-mini", false)

	s.Zero(complete.TimeToFirstTokenMs, "nothing came back, so there was no first token")
	s.Empty(complete.Text)
}

func (s *LLMSuite) TestCompletionKeepsTheLastUsageItWasTold() {
	// Providers repeat a cumulative usage frame on every chunk, so the last one is right.
	completion := NewCompletion("c1")
	completion.Usage(15, 0, 43, 43)
	completion.Usage(15, 8, 64, 47)

	complete := completion.Complete("deepseek", "DeepSeek-V4-Flash-0731", false)

	s.EqualValues(15, complete.InputTokens)
	s.EqualValues(8, complete.CachedInputTokens)
	s.EqualValues(64, complete.OutputTokens)
	s.EqualValues(47, complete.ReasoningTokens)
}

func (s *LLMSuite) TestCompletionReportsBeingInterrupted() {
	completion := NewCompletion("c1")
	completion.Delta("this will be cut ")
	completion.Usage(10, 0, 4, 0)

	complete := completion.Complete("openai", "gpt-4o-mini", true)

	s.True(complete.Interrupted)
	s.Equal("this will be cut ", complete.Text, "the text that did arrive still counts")
	s.EqualValues(4, complete.OutputTokens, "the tokens already generated are still billed")
}

func (s *LLMSuite) TestCompletionReportsWhyTheModelStopped() {
	completion := NewCompletion("c1")
	completion.Delta("truncated")
	completion.Finish("length")

	s.Equal("length", completion.Complete("openai", "gpt-4o-mini", false).FinishReason)
}

func (s *LLMSuite) TestEmitterDeliversEventsInOrder() {
	emitter := NewEmitter(4)
	defer emitter.Close()

	emitter.Send(CompletionStarted{CompletionID: "c1"})
	emitter.Send(TextDelta{CompletionID: "c1", Text: "Hi"})

	started, ok := (<-emitter.Events()).(CompletionStarted)
	s.Require().True(ok)
	s.Equal("c1", started.CompletionID)

	delta, ok := (<-emitter.Events()).(TextDelta)
	s.Require().True(ok)
	s.Equal("Hi", delta.Text)
}

func (s *LLMSuite) TestSendAfterCloseDoesNotPanic() {
	emitter := NewEmitter(1)
	emitter.Close()

	s.NotPanics(func() { emitter.Send(Connected{Provider: "openai"}) })
}

func (s *LLMSuite) TestErrorUnwrapsToTheProviderFailure() {
	cause := Error{Provider: "deepseek", Err: errUnauthorized, Context: "request"}

	s.ErrorIs(cause, errUnauthorized)
	s.Equal("unauthorized", cause.Error())
}
