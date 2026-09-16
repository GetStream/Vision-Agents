//go:build integration

package inworld

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// InworldIntegrationSuite inherits what every provider owes a call from sttsuite. What is
// particular about this one is the turn detection it can be tuned on, which is the half of
// the vocabulary it declares that the shared contract has no place for.
type InworldIntegrationSuite struct {
	sttsuite.Suite
}

func TestInworldIntegrationSuite(t *testing.T) {
	suite.Run(t, &InworldIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"INWORLD_API_KEY"},
		// Ending the turn and the stream makes the server transcribe what it was holding.
		SettlesOnClose: true,
	}})
}

// TestKeytermsAreAcceptedByTheSession covers the custom vocabulary, which this API calls
// prompts. The gateway refuses a malformed term outright rather than ignoring it, so a
// transcript coming back at all is the assertion.
func (s *InworldIntegrationSuite) TestKeytermsAreAcceptedByTheSession() {
	provider, err := New(Options{Keyterms: []string{"Mia"}})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}

// TestATighterTurnBoundaryStillSettlesTheSameTurn is the endpointing this model can be
// told about: how sure it has to be the caller has finished, and how much silence it
// wants before acting on that. Moving both must not cost the end of what was said.
func (s *InworldIntegrationSuite) TestATighterTurnBoundaryStillSettlesTheSameTurn() {
	provider, err := New(Options{
		EndOfTurnConfidenceThreshold: 0.7,
		MinEndOfTurnSilenceMs:        480,
	})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}
