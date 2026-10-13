//go:build integration

package moondream

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/detectionsuite"
)

// MoondreamIntegrationSuite inherits what every detection provider owes from
// detectionsuite. Moondream is asked for the cat by name.
type MoondreamIntegrationSuite struct {
	detectionsuite.Suite
}

func TestMoondreamIntegrationSuite(t *testing.T) {
	suite.Run(t, &MoondreamIntegrationSuite{Suite: detectionsuite.Suite{
		New: func() detection.Provider {
			provider, err := New(Options{Model: "moondream3-preview"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{APIKeyEnvVar},
	}})
}
