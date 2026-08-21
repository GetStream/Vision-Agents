//go:build integration

package gemini

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

type GeminiIntegrationSuite struct {
	suite.Suite
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, new(GeminiIntegrationSuite))
}

func (s *GeminiIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " not set")
	}
}

func (s *GeminiIntegrationSuite) start(options Options) *openaicompat.LLM {
	provider, err := New(options)
	s.Require().NoError(err)

	s.Require().NoError(provider.Start(context.Background()))
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// collect reads events until the completion settles, failing fast on a provider error.
func (s *GeminiIntegrationSuite) collect(provider *openaicompat.LLM) (
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

func (s *GeminiIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Messages: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxTokens: 512,
	}))

	complete, events := s.collect(provider)

	s.Contains(strings.ToLower(complete.Text), "paris")
	s.Equal("c1", complete.CompletionID)
	s.Positive(complete.InputTokens)
	s.Positive(complete.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.TextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *GeminiIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Instructions: "Answer with a single number and nothing else.",
		Messages: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxTokens: 512,
	}))

	complete, _ := s.collect(provider)

	s.Contains(complete.Text, "7", "the whole conversation travels with the request")
}

func (s *GeminiIntegrationSuite) TestToolsAreReachableAlongsideTheReasoningSetting() {
	// OpenAI's own API refuses a request carrying both tools and a reasoning effort,
	// which is why the agent's tools cost that provider its thinking. Google accepts
	// both, and the agent is useless here if it turns out otherwise.
	provider := s.start(Options{})

	s.Require().NoError(provider.Respond(llm.Request{
		Instructions: "Use the tool to answer.",
		Messages:     []llm.Message{{Role: llm.User, Content: "What is the weather in Paris?"}},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Look up the weather somewhere",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
		MaxTokens: 512,
	}))

	complete, _ := s.collect(provider)

	s.Require().Len(complete.ToolCalls, 1)
	s.Equal("get_weather", complete.ToolCalls[0].Name)
	s.Contains(strings.ToLower(complete.ToolCalls[0].Arguments), "paris")
}
