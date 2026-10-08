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

// TestGeminiIntegrationSuite holds every model router.yaml routes here to the same call,
// since what one Live model accepts at setup another can refuse.
func TestGeminiIntegrationSuite(t *testing.T) {
	for _, model := range []string{"gemini-3.1-flash-live-preview", "gemini-3.8-live", "gemini-3.8-live-extended-thinking"} {
		t.Run(model, func(t *testing.T) {
			suite.Run(t, &GeminiIntegrationSuite{Suite: stssuite.Suite{
				New: func(ask stssuite.Ask) sts.STS {
					provider, err := New(Options{
						Model:            model,
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
		})
	}
}
