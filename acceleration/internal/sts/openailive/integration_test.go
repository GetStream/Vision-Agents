//go:build integration

package openailive

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/stssuite"
)

// OpenAILiveIntegrationSuite inherits what every speech-to-speech provider owes a call.
type OpenAILiveIntegrationSuite struct {
	stssuite.Suite
}

func TestOpenAILiveIntegrationSuite(t *testing.T) {
	suite.Run(t, &OpenAILiveIntegrationSuite{Suite: stssuite.Suite{
		New: func(ask stssuite.Ask) sts.STS {
			provider, err := New(Options{Instructions: ask.Instructions, Tools: ask.Tools})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
