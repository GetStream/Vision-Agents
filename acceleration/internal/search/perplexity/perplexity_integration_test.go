//go:build integration

package perplexity

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/searchsuite"
)

// PerplexityIntegrationSuite inherits what every search provider owes a caller from
// searchsuite. Perplexity answers in a sentence of its own, with its citations as the
// sources.
type PerplexityIntegrationSuite struct {
	searchsuite.Suite
}

func TestPerplexityIntegrationSuite(t *testing.T) {
	suite.Run(t, &PerplexityIntegrationSuite{Suite: searchsuite.Suite{
		New: func() search.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
		Answers:  true,
	}})
}
