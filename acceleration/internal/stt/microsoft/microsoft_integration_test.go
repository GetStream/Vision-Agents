//go:build integration

package microsoft

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// MicrosoftIntegrationSuite inherits what every provider owes a call from sttsuite.
// AZURE_MAI_DEPLOYMENT_NAME is optional, for a deployment not named after the model.
type MicrosoftIntegrationSuite struct {
	sttsuite.Suite
}

func TestMicrosoftIntegrationSuite(t *testing.T) {
	suite.Run(t, &MicrosoftIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar, endpointEnvVar},
		// Committing the buffer makes the server transcribe what it is still holding.
		SettlesOnClose: true,
	}})
}
