//go:build integration

package openai

import (
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
