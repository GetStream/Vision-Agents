//go:build integration

package gemma

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
)

// GemmaIntegrationSuite runs against our own deployment, so it needs its URL as well as
// the key: see deploy/gemma-4.
type GemmaIntegrationSuite struct {
	llmsuite.Suite
}

func TestGemmaIntegrationSuite(t *testing.T) {
	suite.Run(t, &GemmaIntegrationSuite{Suite: llmsuite.Suite{
		New: func() llm.LLM {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar, baseURLEnvVar},
	}})
}
