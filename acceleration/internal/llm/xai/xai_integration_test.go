//go:build integration

package xai

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

// XAIIntegrationSuite needs nothing of its own: grok cannot be told not to think, and the
// suite already holds a model that streams its reasoning to showing it.
type XAIIntegrationSuite struct {
	llmsuite.Suite
}

func TestXAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &XAIIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
