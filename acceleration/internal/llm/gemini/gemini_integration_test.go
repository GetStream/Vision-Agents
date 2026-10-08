//go:build integration

package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

type GeminiIntegrationSuite struct {
	llmsuite.Suite
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}

func (s *GeminiIntegrationSuite) TestAToolCallComesBackSigned() {
	// Google signs every call it asks for and rejects the turn that carries the result
	// back unless the signature comes with it. The suite checks that turn is answered;
	// this is the signature it depends on.
	called, _ := s.Ask(llm.ResponseParams{
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
	})

	s.Require().Len(called.ToolCalls, 1)
	s.NotEmpty(called.ToolCalls[0].Signature, "Google signs its calls and wants them back signed")
}
