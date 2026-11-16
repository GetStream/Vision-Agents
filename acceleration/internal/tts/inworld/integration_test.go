//go:build integration

package inworld

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type InworldIntegrationSuite struct {
	ttssuite.Suite
}

func TestInworldIntegrationSuite(t *testing.T) {
	suite.Run(t, &InworldIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"INWORLD_API_KEY"},
		// 25ms is a server-side P90 that excludes the network; this is the wait a
		// listener may still find acceptable on a live call.
		MaxTimeToFirstByte: 2_000,
		Interruptible:      true,
		Timeout:            time.Minute,
	}})
}

// TestInworldTTS2IntegrationSuite runs the same suite on the full model, which is slower
// to first byte than Flash but well inside what a live call tolerates.
func TestInworldTTS2IntegrationSuite(t *testing.T) {
	suite.Run(t, &InworldIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{Model: "inworld-tts-2"})
			require.NoError(t, err)
			return provider
		},
		Requires:           []string{"INWORLD_API_KEY"},
		MaxTimeToFirstByte: 2_000,
		Interruptible:      true,
		Timeout:            time.Minute,
	}})
}
