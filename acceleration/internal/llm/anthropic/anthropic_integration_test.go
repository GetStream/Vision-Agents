//go:build integration

package anthropic

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

// AnthropicIntegrationSuite needs nothing of its own: the compatibility endpoint does not
// return Claude's thinking, so the suite sees a model that answers without reasoning.
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
