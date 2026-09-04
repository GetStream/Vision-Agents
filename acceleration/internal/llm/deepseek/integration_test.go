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
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
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
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected request reports what went wrong rather than timing out.
func (s *DeepSeekIntegrationSuite) ask(provider *openaicompat.LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	stream, err := provider.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	s.Require().NoError(stream.Err())
	return stream.Response(), events
}

func (s *DeepSeekIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	complete, events := s.ask(provider, llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxOutputTokens: 32,
	})

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal("c1", complete.ID)
	s.Positive(complete.Usage.InputTokens, "there is nothing to bill without a token count")
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)
	s.LessOrEqual(complete.TimeToFirstTokenMs, complete.DurationMs)
	s.Equal(llm.StatusCompleted, complete.Status)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *DeepSeekIntegrationSuite) TestThinkingIsOffByDefaultSoTheAnswerArrivesFirst() {
	// With reasoning on, a small token budget is spent entirely on thinking and the answer
	// never appears. That is exactly the failure the default is there to avoid.
	provider := s.start(Options{})

	complete, events := s.ask(provider, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Say hello in five words."}},
		MaxOutputTokens: 64,
	})

	s.NotEmpty(complete.OutputText)
	s.Zero(complete.Usage.OutputTokensDetails.ReasoningTokens, "the chat template argument really does disable thinking")

	for _, event := range events {
		_, thinking := event.(llm.ReasoningTextDelta)
		s.False(thinking, "a non-thinking request must not stream reasoning")
	}
}

func (s *DeepSeekIntegrationSuite) TestThinkingStreamsReasoningWhenTurnedOn() {
	provider := s.start(Options{Thinking: true, ReasoningEffort: "low"})

	complete, events := s.ask(provider, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Is 91 prime? Answer yes or no."}},
		MaxOutputTokens: 2048,
	})

	var thinking strings.Builder
	for _, event := range events {
		if delta, ok := event.(llm.ReasoningTextDelta); ok {
			thinking.WriteString(delta.Delta)
		}
	}

	s.NotEmpty(thinking.String(), "a reasoning model should show its working")
	s.Positive(complete.Usage.OutputTokensDetails.ReasoningTokens)
	s.LessOrEqual(complete.Usage.OutputTokensDetails.ReasoningTokens, complete.Usage.OutputTokens,
		"reasoning is part of the output, not an extra charge on top of it")
	s.NotContains(complete.OutputText, thinking.String(), "thinking is not part of the answer")
}

func (s *DeepSeekIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Answer with a single number and nothing else.",
		Input: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxOutputTokens: 32,
	})

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}

func (s *DeepSeekIntegrationSuite) TestClosingStopsTheAnswerMidStream() {
	provider := s.start(Options{})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	stream, err := provider.Create(ctx, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Count slowly from 1 to 200."}},
		MaxOutputTokens: 2048,
	})
	s.Require().NoError(err)

	// Wait for the model to start talking, then cut it off the way barge-in would. The
	// stream is drained to the end afterwards: what it generated before being closed was
	// generated all the same, and it is the last event that reports it.
	for stream.Next() {
		if _, talking := stream.Current().(llm.OutputTextDelta); talking {
			s.Require().NoError(stream.Close())
			break
		}
	}
	for stream.Next() {
	}
	s.Require().NoError(stream.Err())

	complete := stream.Response()
	s.Equal(llm.StatusCancelled, complete.Status)
	s.NotEmpty(complete.OutputText, "what was already said still counts")
	s.NotContains(complete.OutputText, "200", "the answer was cut short")
}
