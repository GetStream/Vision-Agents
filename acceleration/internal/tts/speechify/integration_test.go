//go:build integration

package speechify

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type SpeechifyIntegrationSuite struct {
	ttssuite.Suite
}

func TestSpeechifyIntegrationSuite(t *testing.T) {
	suite.Run(t, &SpeechifyIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:           []string{"SPEECHIFY_API_KEY"},
		Interruptible:      true,
		MaxTimeToFirstByte: 2_000,
	}})
}
