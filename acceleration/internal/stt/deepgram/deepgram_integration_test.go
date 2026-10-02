//go:build integration

package deepgram

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// DeepgramIntegrationSuite inherits what every provider owes a call from sttsuite. Flux
// prefers roughly 80 ms of audio at a time, and ends a turn on about two seconds of
// silence of its own accord, so it is given more than that and the turn ends because Flux
// decided it had rather than because the audio ran out.
type DeepgramIntegrationSuite struct {
	sttsuite.Suite
}

func TestDeepgramIntegrationSuite(t *testing.T) {
	suite.Run(t, &DeepgramIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:  []string{"DEEPGRAM_API_KEY"},
		ChunkMs:   80,
		SilenceMs: 3000,
	}})
}

// TestAnIdleSessionOutlivesTheFluxPingDeadline is slow by nature: the deadline it is about
// is a minute long, so there is no shorter way to see the session survive it.
func (s *DeepgramIntegrationSuite) TestAnIdleSessionOutlivesTheFluxPingDeadline() {
	provider := s.Started()
	defer s.Hangup(provider)

	finals := make(chan stt.Transcript, 8)
	failures := make(chan error, 8)
	go func() {
		for event := range provider.Events() {
			switch typed := event.(type) {
			case stt.Transcript:
				if typed.Final() {
					finals <- typed
				}
			case stt.Error:
				failures <- typed.Err
			}
		}
	}()

	// Flux closes a session it has not been pinged on for 60 seconds, and the audio a
	// conversation sends does not count towards that.
	select {
	case err := <-failures:
		s.FailNowf("the idle session was closed", "%v", err)
	case <-time.After(75 * time.Second):
	}

	s.Speak(provider, 0)
	s.Quiet(provider)

	select {
	case final := <-finals:
		s.RequireAccurate(final.Text)
	case err := <-failures:
		s.FailNowf("the session failed after the pause", "%v", err)
	case <-time.After(20 * time.Second):
		s.FailNow("nothing came back after the pause")
	}
}

// TestTheFinalReportsTheAudioItCovered is what usage is charted against. Flux reports the
// window each turn was decoded from, which not every provider does.
func (s *DeepgramIntegrationSuite) TestTheFinalReportsTheAudioItCovered() {
	provider := s.Started()
	defer s.Hangup(provider)

	var final stt.Transcript
	for _, event := range s.Collect(provider) {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			final = transcript
		}
	}

	s.Positive(final.AudioDurationMs, "final transcripts should report the audio window")
}
