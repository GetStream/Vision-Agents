//go:build integration

package google

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen/imagegensuite"
)

// GoogleIntegrationSuite draws with Gemini 3.1 Flash Image, the model the router serves on
// Google.
type GoogleIntegrationSuite struct {
	imagegensuite.Suite
}

func TestGoogleIntegrationSuite(t *testing.T) {
	suite.Run(t, &GoogleIntegrationSuite{Suite: imagegensuite.Suite{
		New: func() imagegen.Provider {
			provider, err := New(Options{Model: "gemini-3.1-flash-image"})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
