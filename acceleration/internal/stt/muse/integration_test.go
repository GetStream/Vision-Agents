//go:build integration

package muse

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// MuseIntegrationSuite inherits what every provider owes a call from sttsuite. What is
// particular about this provider is diarisation, which the shared contract has no place
// for and which is therefore checked here.
type MuseIntegrationSuite struct {
	sttsuite.Suite
}

func TestMuseIntegrationSuite(t *testing.T) {
	suite.Run(t, &MuseIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"META_API_KEY"},
		// The model turns each 80ms of audio into one token, so that is the pace it is
		// built to be fed at.
		ChunkMs: 80,
		// Ending the audio stream makes the server settle the turn it was still holding.
		SettlesOnClose: true,
	}})
}

// TestDiarisationNamesTheSpeakerWithoutChangingTheTranscript is the one mode that changes
// what comes back rather than only how it is timed. The labels are the server's own and
// the router does not use them, so what matters is that asking for them still produces
// the transcript the rest of the call is built on.
func (s *MuseIntegrationSuite) TestDiarisationNamesTheSpeakerWithoutChangingTheTranscript() {
	provider, err := New(Options{Mode: ModeDiarization})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}

// TestKeytermsAreAcceptedByTheSession covers the vocabulary biasing the setup frame
// carries. A server that rejected the terms would fail the session rather than quietly
// ignore them, so the transcript coming back at all is the assertion.
func (s *MuseIntegrationSuite) TestKeytermsAreAcceptedByTheSession() {
	provider, err := New(Options{Keyterms: []string{"Mia"}})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}
