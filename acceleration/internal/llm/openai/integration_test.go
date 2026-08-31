//go:build integration

package openai

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

type OpenAIIntegrationSuite struct {
	suite.Suite
}

func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIIntegrationSuite))
}

func (s *OpenAIIntegrationSuite) SetupSuite() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		s.T().Skip("OPENAI_API_KEY not set")
	}
}

func (s *OpenAIIntegrationSuite) start(options Options) *openaicompat.LLM {
	provider, err := New(options)
	s.Require().NoError(err)

	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// collect reads events until the completion settles, failing fast on a provider error.
func (s *OpenAIIntegrationSuite) collect(provider *openaicompat.LLM) (
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

func (s *OpenAIIntegrationSuite) TestAnswersAndReportsWhatItCost() {
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
	s.Positive(complete.InputTokens)
	s.Positive(complete.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)
	s.Equal("stop", complete.FinishReason)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.TextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *OpenAIIntegrationSuite) TestConversationHistoryIsHonoured() {
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

func (s *OpenAIIntegrationSuite) TestATruncatedAnswerSaysWhyItStopped() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Messages:  []llm.Message{{Role: llm.User, Content: "Write a long essay about the sea."}},
		MaxTokens: 16,
	}))

	complete, _ := s.collect(provider)

	s.Equal("length", complete.FinishReason)
	s.NotEmpty(complete.Text)
}
