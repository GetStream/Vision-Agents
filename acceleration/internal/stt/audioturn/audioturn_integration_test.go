//go:build integration

package audioturn

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

func TestAudioTurnIntegrationSuite(t *testing.T) {
	suite.Run(t, &sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		// Select a transcript-capable deployment explicitly for live audio tests.
		Requires: []string{"ROUTER_EOT_URL"},
	})
}
