//go:build integration

package anthropic

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

// AnthropicIntegrationSuite needs nothing of its own for the shared tests: Claude's thinking
// streams as empty blocks unless a display is asked for, so the suite sees a model that
// answers without reasoning.
type AnthropicIntegrationSuite struct {
	llmsuite.Suite
}

// Opus 5.5 is the one that takes a note mid-conversation as a system turn (systemTurnModels);
// the others take it on the user's side, so between them both ways are run.
func TestAnthropicIntegrationSuite(t *testing.T) {
	for _, model := range []string{"claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"} {
		t.Run(model, func(t *testing.T) {
			suite.Run(t, &AnthropicIntegrationSuite{Suite: llmsuite.Suite{
				New: func() llm.LLM {
					provider, err := New(Options{Model: model})
					require.NoError(t, err)
					return provider
				},
				Requires: []string{"ANTHROPIC_API_KEY"},
			}})
		})
	}
}

// TestTheNextRoundReadsThePrefixFromTheCache asks twice with the same long instructions and
// a conversation one round longer, and expects the second to be served mostly from cache.
func (s *AnthropicIntegrationSuite) TestTheNextRoundReadsThePrefixFromTheCache() {
	instructions := "Answer in one word. " + strings.Repeat("The webhook retries a failed delivery three times, a minute apart. ", 300)
	first := llm.ResponseParams{Instructions: instructions, Input: []llm.Message{{Role: llm.User, Content: "How many retries?"}}}
	answer, _ := s.Ask(first)
	s.Positive(answer.Usage.InputTokensDetails.CacheWriteTokens, "the first round writes the prefix")

	second := first
	second.Input = append(append([]llm.Message(nil), first.Input...),
		llm.Message{Role: llm.Assistant, Content: answer.OutputText},
		llm.Message{Role: llm.User, Content: "And how far apart?"})
	again, _ := s.Ask(second)

	s.Greater(again.Usage.InputTokensDetails.CachedTokens, again.Usage.InputTokens/2)
}

// TestThinkingIsReplayedThroughAToolRound sends a tool round's signed thinking back with its
// result. Claude refuses a turn whose thinking was moved or edited, so an answer is the proof
// the blocks went back as they came.
func (s *AnthropicIntegrationSuite) TestThinkingIsReplayedThroughAToolRound() {
	clock := llm.Tool{Name: "clock", Description: "Says the time in a city.", Parameters: map[string]any{
		"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"},
	}}
	question := llm.Message{Role: llm.User, Content: "What time is it in Paris? Think it over, then use the tool."}
	first := llm.ResponseParams{Instructions: "Use the tool to answer.", Input: []llm.Message{question}, Tools: []llm.Tool{clock}, Reasoning: llm.ReasoningParams{Effort: "high"}}
	called, _ := s.Ask(first)
	s.Require().NotEmpty(called.ToolCalls)
	if len(called.ReasoningBlocks) == 0 {
		s.T().Skip("the model called without thinking, so there is nothing to replay")
	}

	second := first
	second.Input = []llm.Message{question, {Role: llm.Assistant, Content: called.OutputText, ToolCalls: called.ToolCalls, ReasoningBlocks: called.ReasoningBlocks}}
	for _, call := range called.ToolCalls {
		second.Input = append(second.Input, llm.Message{Role: llm.ToolResult, ToolCallID: call.ID, Content: "14:05"})
	}
	answered, _ := s.Ask(second)

	s.Equal(llm.StatusCompleted, answered.Status)
	s.Contains(answered.OutputText, "14")
}
