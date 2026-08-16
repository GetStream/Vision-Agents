//go:build integration

package deepseek

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

type DeepSeekIntegrationSuite struct {
	suite.Suite
}

func TestDeepSeekIntegrationSuite(t *testing.T) {
	suite.Run(t, new(DeepSeekIntegrationSuite))
}

func (s *DeepSeekIntegrationSuite) SetupSuite() {
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
	}
}

func (s *DeepSeekIntegrationSuite) start(options Options) *openaicompat.LLM {
	provider, err := New(options)
	s.Require().NoError(err)

	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// collect reads events until the completion settles. A provider error fails the test
// straight away, so a rejected request reports what went wrong instead of timing out.
func (s *DeepSeekIntegrationSuite) collect(provider *openaicompat.LLM) (
	llm.CompletionComplete, []llm.Event,
) {
	var events []llm.Event
	deadline := time.After(90 * time.Second)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the provider closed before the completion settled")
				return llm.CompletionComplete{}, events
			}
			events = append(events, event)
			if failure, failed := event.(llm.Error); failed {
				s.FailNowf("provider error", "%v", failure.Err)
			}
			if complete, done := event.(llm.CompletionComplete); done {
				return complete, events
			}
		case <-deadline:
			s.FailNow("timed out waiting for a completion")
			return llm.CompletionComplete{}, events
		}
	}
}

func (s *DeepSeekIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Messages: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxTokens: 32,
	}))

	complete, events := s.collect(provider)

	s.Contains(strings.ToLower(complete.Text), "paris")
	s.Equal("c1", complete.CompletionID)
	s.Positive(complete.InputTokens, "there is nothing to bill without a token count")
	s.Positive(complete.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)
	s.LessOrEqual(complete.TimeToFirstTokenMs, complete.CompletionTimeMs)
	s.False(complete.Interrupted)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.TextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *DeepSeekIntegrationSuite) TestThinkingIsOffByDefaultSoTheAnswerArrivesFirst() {
	// With reasoning on, a small token budget is spent entirely on thinking and the answer
	// never appears. That is exactly the failure the default is there to avoid.
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Messages:  []llm.Message{{Role: llm.User, Content: "Say hello in five words."}},
		MaxTokens: 64,
	}))

	complete, events := s.collect(provider)

	s.NotEmpty(complete.Text)
	s.Zero(complete.ReasoningTokens, "the chat template argument really does disable thinking")

	for _, event := range events {
		_, thinking := event.(llm.ReasoningDelta)
		s.False(thinking, "a non-thinking request must not stream reasoning")
	}
}

func (s *DeepSeekIntegrationSuite) TestThinkingStreamsReasoningWhenTurnedOn() {
	provider := s.start(Options{Thinking: true, ReasoningEffort: "low"})

	s.Require().NoError(provider.Respond(llm.Request{
		Messages:  []llm.Message{{Role: llm.User, Content: "Is 91 prime? Answer yes or no."}},
		MaxTokens: 2048,
	}))

	complete, events := s.collect(provider)

	var thinking strings.Builder
	for _, event := range events {
		if delta, ok := event.(llm.ReasoningDelta); ok {
			thinking.WriteString(delta.Text)
		}
	}

	s.NotEmpty(thinking.String(), "a reasoning model should show its working")
	s.Positive(complete.ReasoningTokens)
	s.LessOrEqual(complete.ReasoningTokens, complete.OutputTokens,
		"reasoning is part of the output, not an extra charge on top of it")
	s.NotContains(complete.Text, thinking.String(), "thinking is not part of the answer")
}

func (s *DeepSeekIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Instructions: "Answer with a single number and nothing else.",
		Messages: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxTokens: 32,
	}))

	complete, _ := s.collect(provider)

	s.Contains(complete.Text, "7", "the whole conversation travels with the request")
}

func (s *DeepSeekIntegrationSuite) TestInterruptStopsTheAnswerMidStream() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Messages:  []llm.Message{{Role: llm.User, Content: "Count slowly from 1 to 200."}},
		MaxTokens: 2048,
	}))

	// Wait for the model to start talking, then cut it off the way barge-in would.
	deadline := time.After(60 * time.Second)
	for talking := false; !talking; {
		select {
		case event := <-provider.Events():
			_, talking = event.(llm.TextDelta)
		case <-deadline:
			s.FailNow("the model never started answering")
		}
	}
	s.Require().NoError(provider.Interrupt())

	complete, _ := s.collect(provider)

	s.True(complete.Interrupted)
	s.NotEmpty(complete.Text, "what was already said still counts")
	s.NotContains(complete.Text, "200", "the answer was cut short")
}
