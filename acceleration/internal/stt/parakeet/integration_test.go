//go:build integration

package parakeet

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// coldStartTimeout is how long a scaled-to-zero L4 deployment may take to answer. The
// first connection after an idle spell pays for a GPU cold start and needs far longer
// than the timeout a live call would accept.
const coldStartTimeout = 6 * time.Minute

// ParakeetIntegrationSuite inherits what every provider owes a call from sttsuite. What
// is particular about this provider is mostly that it may not be running yet, which is a
// matter of timeouts rather than of behaviour.
type ParakeetIntegrationSuite struct {
	sttsuite.Suite
}

func TestParakeetIntegrationSuite(t *testing.T) {
	suite.Run(t, &ParakeetIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{HandshakeTimeout: coldStartTimeout})
			require.NoError(t, err)
			return provider
		},
		Requires:       []string{"PARAKEET_WS_URL", "BASETEN_API_KEY"},
		ChunkMs:        80,
		SessionTimeout: coldStartTimeout,
		// Closing flushes the audio the server is still holding.
		SettlesOnClose: true,
	}})
}

// TestTheFinalReportsWhatItCostToDecode is the deployment's own accounting: how much audio
// the turn covered and how long the GPU spent on it, which is how a model running on our
// own hardware is watched.
func (s *ParakeetIntegrationSuite) TestTheFinalReportsWhatItCostToDecode() {
	provider := s.Started()
	defer s.Hangup(provider)

	var final stt.Transcript
	for _, event := range s.Collect(provider) {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			final = transcript
		}
	}

	s.Positive(final.AudioDurationMs, "the final should report how much audio it covered")
	s.Positive(final.ProcessingTimeMs, "the final should report the server's decode time")
}
