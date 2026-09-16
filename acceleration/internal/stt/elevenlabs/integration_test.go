//go:build integration

package elevenlabs

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// ElevenlabsIntegrationSuite inherits what every provider owes a call from sttsuite. What
// is particular about this one is that a segment is settled by a commit rather than by the
// model deciding the turn is over, so the tests here are about what moves that commit.
type ElevenlabsIntegrationSuite struct {
	sttsuite.Suite
}

func TestElevenlabsIntegrationSuite(t *testing.T) {
	suite.Run(t, &ElevenlabsIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"ELEVENLABS_API_KEY"},
		// Closing commits the audio the server was still holding.
		SettlesOnClose: true,
		// Scribe does not begin transcribing until it has about two seconds of audio, so
		// the first partial lands at 2.2s to 2.4s where the other providers answer inside
		// one. That is the model warming up rather than the hypotheses drying up, and the
		// segment still settles well inside the shared MaxSettle once they start.
		MaxToFirstWords: 3 * time.Second,
	}})
}

// TestKeytermsAreAcceptedByTheSession covers the vocabulary biasing, which this endpoint
// takes as repeated query parameters. A server that rejected them would refuse the session
// rather than quietly ignore them, so a transcript coming back at all is the assertion.
func (s *ElevenlabsIntegrationSuite) TestKeytermsAreAcceptedByTheSession() {
	provider, err := New(Options{Keyterms: []string{"Mia", "Scribe"}})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}

// TestALessPatientDetectorStillSettlesTheSameTurn is the endpointing this model can be
// told about. Shortening the silence it waits through must not cost the end of what the
// caller said.
func (s *ElevenlabsIntegrationSuite) TestALessPatientDetectorStillSettlesTheSameTurn() {
	provider, err := New(Options{VadSilenceThresholdSecs: 0.4, MinSilenceDurationMs: 200})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}
