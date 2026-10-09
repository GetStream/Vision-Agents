//go:build integration

package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

type OpenAIIntegrationSuite struct {
	llmsuite.Suite
}

func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &OpenAIIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}

func (s *OpenAIIntegrationSuite) TestEveryGPT6ModelAnswersAtItsDefaultEffort() {
	for _, model := range []string{"gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra"} {
		s.Run(model, func() {
			provider, err := New(Options{Model: model})
			s.Require().NoError(err)

			complete, _ := s.AskOn(provider, llm.ResponseParams{
				Instructions: "Answer with a single word and no punctuation.",
				Input:        []llm.Message{{Role: llm.User, Content: "What is the capital of France?"}},
			})

			s.Contains(strings.ToLower(complete.OutputText), "paris")
			s.Equal(llm.StatusCompleted, complete.Status)
		})
	}
}

// TestACallLeavesOutTheOptionalArgumentsItDoesNotNeed is AI-969 against the model that sent
// thread_ts:"" to Slack: a plain post names the channel and the message, and nothing else.
func (s *OpenAIIntegrationSuite) TestACallLeavesOutTheOptionalArgumentsItDoesNotNeed() {
	provider, err := New(Options{Model: "gpt-5.6-sol"})
	s.Require().NoError(err)

	called, _ := s.AskOn(provider, llm.ResponseParams{
		Input:      []llm.Message{{Role: llm.User, Content: "Post 'hello from the e2e' to Slack channel C0123456789."}},
		Tools:      []llm.Tool{sendMessage},
		ToolChoice: "required",
	})

	s.Require().Len(called.ToolCalls, 1)
	var arguments map[string]any
	s.Require().NoError(json.Unmarshal([]byte(called.ToolCalls[0].Arguments), &arguments))
	s.Contains(arguments, "channel_id")
	s.Contains(arguments, "message")
	s.NotContains(arguments, "thread_ts")
	s.NotContains(arguments, "draft_id")
}
