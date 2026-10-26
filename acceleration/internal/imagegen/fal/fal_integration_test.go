//go:build integration

package fal

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen/imagegensuite"
)

// FALIntegrationSuite draws with Qwen Image 3, the model the router serves on FAL.
type FALIntegrationSuite struct {
	imagegensuite.Suite
}

func TestFALIntegrationSuite(t *testing.T) {
	suite.Run(t, &FALIntegrationSuite{Suite: imagegensuite.Suite{
		New: func() imagegen.Provider {
			provider, err := New(Options{Model: model})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
