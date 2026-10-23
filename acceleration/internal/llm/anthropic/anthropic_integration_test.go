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

func TestAnthropicIntegrationSuite(t *testing.T) {
	for _, model := range []string{"claude-sonnet-5-5", "claude-haiku-5-5"} {
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
