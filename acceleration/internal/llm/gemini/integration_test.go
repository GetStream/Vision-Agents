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
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
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
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected request reports what went wrong rather than timing out.
func (s *GeminiIntegrationSuite) ask(provider *openaicompat.LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
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

func (s *GeminiIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	complete, events := s.ask(provider, llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxOutputTokens: 512,
	})

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal("c1", complete.ID)
	s.Positive(complete.Usage.InputTokens)
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *GeminiIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Answer with a single number and nothing else.",
		Input: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxOutputTokens: 512,
	})

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}

func (s *GeminiIntegrationSuite) TestToolsAreReachableAlongsideTheReasoningSetting() {
	// OpenAI's own API refuses a request carrying both tools and a reasoning effort,
	// which is why the agent's tools cost that provider its thinking. Google accepts
	// both, and the agent is useless here if it turns out otherwise.
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Use the tool to answer.",
		Input:        []llm.Message{{Role: llm.User, Content: "What is the weather in Paris?"}},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Look up the weather somewhere",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
		MaxOutputTokens: 512,
	})

	s.Require().Len(complete.ToolCalls, 1)
	s.Equal("get_weather", complete.ToolCalls[0].Name)
	s.Contains(strings.ToLower(complete.ToolCalls[0].Arguments), "paris")
}

func (s *GeminiIntegrationSuite) TestAToolResultIsAnsweredRatherThanRefused() {
	// Google signs every call it asks for and rejects the turn that carries the result
	// back unless the signature comes with it. That turn is the one that says the answer
	// out loud, so losing the signature is a caller asking a question, a tool running,
	// and nobody ever telling them what it found.
	provider := s.start(Options{})
	tools := []llm.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather somewhere",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		},
	}}
	asked := llm.Message{Role: llm.User, Content: "What is the weather in Paris?"}

	called, _ := s.ask(provider, llm.ResponseParams{
		Instructions:    "Use the tool to answer.",
		Input:           []llm.Message{asked},
		Tools:           tools,
		MaxOutputTokens: 512,
	})
	s.Require().Len(called.ToolCalls, 1)
	s.NotEmpty(called.ToolCalls[0].Signature, "Google signs its calls and wants them back signed")

	answered, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Tell the caller what the tool found.",
		Input: []llm.Message{
			asked,
			{Role: llm.Assistant, Content: called.OutputText, ToolCalls: called.ToolCalls},
			{
				Role:       llm.ToolResult,
				ToolCallID: called.ToolCalls[0].ID,
				Content:    "It is 20 degrees and sunny in Paris.",
			},
		},
		Tools:           tools,
		MaxOutputTokens: 512,
	})

	s.Contains(answered.OutputText, "20")
}
