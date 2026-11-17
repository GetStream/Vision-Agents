//go:build integration

package openrouter

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/decisionmodelsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
)

// OpenRouterIntegrationSuite asks OpenRouter's default decision model, Jev.
type OpenRouterIntegrationSuite struct {
	decisionmodelsuite.Suite
}

func TestOpenRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, &OpenRouterIntegrationSuite{Suite: decisionmodelsuite.Suite{
		New: func() decisionmodel.Provider {
			provider, err := New(systemone.Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{Endpoint.APIKeyEnvVar},
	}})
}

// TestOpenRoutersCostIsReported is what the price list is checked against: OpenRouter says
// what every answer cost it.
func (s *OpenRouterIntegrationSuite) TestOpenRoutersCostIsReported() {
	s.Positive(s.Answered().VendorCostMicros())
}
