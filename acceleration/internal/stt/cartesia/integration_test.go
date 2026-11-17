//go:build integration

package cartesia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// CartesiaIntegrationSuite inherits what every provider owes a call from sttsuite. What is
// particular about this one is that the turn boundary is the model's own judgement rather
// than a silence threshold, so the tests here are about the settings that move it.
type CartesiaIntegrationSuite struct {
	sttsuite.Suite
}

func TestCartesiaIntegrationSuite(t *testing.T) {
	suite.Run(t, &CartesiaIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"CARTESIA_API_KEY"},
		// The close command makes the server transcribe the audio it had buffered, so the
		// tail of a call that was cut off is not lost.
		SettlesOnClose: true,
	}})
}

// TestKeytermsAreAcceptedByTheSession covers the vocabulary biasing, which this endpoint
// takes as repeated query parameters. A server that rejected them would refuse the session
// rather than quietly ignore them, so a transcript coming back at all is the assertion.
func (s *CartesiaIntegrationSuite) TestKeytermsAreAcceptedByTheSession() {
	provider, err := New(Options{Keyterms: []string{"Mia", "Ink 2"}})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}

// TestAShorterEndTimeoutStillSettlesTheSameTurn is the endpointing this model can be told
// about. The timeout caps how long it waits after the caller stops, so shortening it must
// not cost the end of what they said.
func (s *CartesiaIntegrationSuite) TestAShorterEndTimeoutStillSettlesTheSameTurn() {
	provider, err := New(Options{TurnEndTimeoutMs: 800})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}
