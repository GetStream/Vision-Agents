//go:build integration

package openaicompat

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

type BasetenRealtimeIntegrationSuite struct{ sttsuite.Suite }

func TestBasetenRealtimeIntegrationSuite(t *testing.T) {
	if os.Getenv("OPENAI_STT_REALTIME_PROTOCOL") != "baseten" {
		t.Skip("requires a configured Baseten streaming endpoint")
	}
	suite.Run(t, &BasetenRealtimeIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := NewRealtime(Options{Model: "qwen3-asr-1.7b"})
			require.NoError(t, err)
			return provider
		},
		Requires:       []string{"OPENAI_STT_BASE_URL", "OPENAI_STT_API_KEY"},
		SettlesOnClose: true,
	}})
}
