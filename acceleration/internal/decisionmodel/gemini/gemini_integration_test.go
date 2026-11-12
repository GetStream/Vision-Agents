//go:build integration

package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/decisionmodelsuite"
)

// GeminiIntegrationSuite asks Gemini Decision 1 on the Gemini API.
type GeminiIntegrationSuite struct {
	decisionmodelsuite.Suite
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{Suite: decisionmodelsuite.Suite{
		New: func() decisionmodel.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
