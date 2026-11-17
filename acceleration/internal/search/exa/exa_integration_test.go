//go:build integration

package exa

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/searchsuite"
)

// ExaIntegrationSuite inherits what every search provider owes a caller from searchsuite.
// Exa writes no answer of its own, which is what makes it the fast option, and it is the
// one provider that narrows by domain and reads a page it is handed.
type ExaIntegrationSuite struct {
	searchsuite.Suite
}

func TestExaIntegrationSuite(t *testing.T) {
	suite.Run(t, &ExaIntegrationSuite{Suite: searchsuite.Suite{
		New: func() search.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:        []string{apiKeyEnvVar},
		NarrowsByDomain: true,
	}})
}
