//go:build integration

package perplexity

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/searchsuite"
)

// PerplexityIntegrationSuite inherits what every search provider owes a caller from
// searchsuite. The ranked index returns pages; Sonar answers in a sentence of its own,
// with the pages it searched as the sources.
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
	}})
}

func TestSonarIntegrationSuite(t *testing.T) {
	suite.Run(t, &PerplexityIntegrationSuite{Suite: searchsuite.Suite{
		New: func() search.Provider {
			provider, err := New(Options{Model: "sonar"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
		Answers:  true,
	}})
}

func TestASonarAnswerCountsItsTokensAndSearches(t *testing.T) {
	if os.Getenv(apiKeyEnvVar) == "" {
		t.Skip(apiKeyEnvVar + " is not set")
	}
	provider, err := New(Options{Model: "sonar"})
	require.NoError(t, err)

	found, err := provider.Search(context.Background(), search.Query{Text: "who won the most recent Tour de France"})
	require.NoError(t, err)

	used := found.Used()
	require.NotEmpty(t, found.Answer)
	require.NotEmpty(t, found.Documents)
	require.Positive(t, used[pricing.InputTokens]+used[pricing.CachedInputTokens])
	require.Positive(t, used[pricing.OutputTokens])
	require.Positive(t, used[pricing.Requests], "the answer ran a web search")
	require.Positive(t, found.VendorCostMicros())
}
