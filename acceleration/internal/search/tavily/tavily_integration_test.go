//go:build integration

package tavily

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/searchsuite"
)

// TavilyIntegrationSuite inherits what every search provider owes a caller from
// searchsuite. Tavily is asked for an answer alongside its sources.
type TavilyIntegrationSuite struct {
	searchsuite.Suite
}

func TestTavilyIntegrationSuite(t *testing.T) {
	suite.Run(t, &TavilyIntegrationSuite{Suite: searchsuite.Suite{
		New: func() search.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
		Answers:  true,
	}})
}
