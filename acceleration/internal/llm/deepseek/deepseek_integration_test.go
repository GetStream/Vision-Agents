//go:build integration

package deepseek

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

type DeepSeekIntegrationSuite struct {
	llmsuite.Suite
}

func TestDeepSeekIntegrationSuite(t *testing.T) {
	suite.Run(t, &DeepSeekIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}

func (s *DeepSeekIntegrationSuite) TestThinkingIsOffByDefaultSoTheAnswerArrivesFirst() {
	// With reasoning on, a small token budget is spent entirely on thinking and the answer
	// never appears. That is exactly the failure the default is there to avoid.
	complete, events := s.Ask(llm.ResponseParams{
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
	provider, err := New(Options{Thinking: true, ReasoningEffort: "low"})
	s.Require().NoError(err)

	complete, events := s.AskOn(provider, llm.ResponseParams{
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
