//go:build integration

package perplexity

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/decisionmodelsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
)

// PerplexityIntegrationSuite asks Decider V1.1 on Perplexity's own API.
type PerplexityIntegrationSuite struct {
	decisionmodelsuite.Suite
}

func TestPerplexityIntegrationSuite(t *testing.T) {
	suite.Run(t, &PerplexityIntegrationSuite{Suite: decisionmodelsuite.Suite{
		New: func() decisionmodel.Provider {
			provider, err := New(systemone.Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{Endpoint.APIKeyEnvVar},
	}})
}
