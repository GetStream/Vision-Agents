//go:build integration

package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/stssuite"
)

// GeminiIntegrationSuite inherits what every speech-to-speech provider owes a call.
type GeminiIntegrationSuite struct {
	stssuite.Suite
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{Suite: stssuite.Suite{
		New: func(ask stssuite.Ask) sts.STS {
			provider, err := New(Options{
				Instructions:     ask.Instructions,
				Tools:            ask.Tools,
				InputTranscript:  true,
				OutputTranscript: true,
			})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}
