//go:build integration

package gemini

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// GeminiIntegrationSuite inherits what every provider owes a call from sttsuite and adds
// the two options of this one that only the server can answer for.
type GeminiIntegrationSuite struct {
	sttsuite.Suite
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, &GeminiIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
		// Ending the audio stream makes the server transcribe what it is still holding.
		SettlesOnClose: true,
	}})
}

// TestAKeytermIsSpelledTheWayItWasGiven exercises the custom vocabulary, which is where
// keyterms go. "Mia" is a name the model is free to hear as Mya or Maya.
func (s *GeminiIntegrationSuite) TestAKeytermIsSpelledTheWayItWasGiven() {
	provider := s.started(Options{Keyterms: []string{"Mia"}})
	defer s.Hangup(provider)

	s.Contains(s.SettledText(provider), "Mia")
}

// TestEitherTranscriptionModeIsAcceptedAndStillWritesTheWordsDown covers the mode
// reaching the server at all: a field it did not recognise would be rejected in the setup
// exchange rather than ignored. What each mode does with filler and false starts is the
// model's business; that the words themselves survive either one is ours.
func (s *GeminiIntegrationSuite) TestEitherTranscriptionModeIsAcceptedAndStillWritesTheWordsDown() {
	for _, mode := range []TranscriptionMode{ModeVerbatim, ModeSmart} {
		s.Run(string(mode), func() {
			provider := s.started(Options{Mode: mode})
			defer s.Hangup(provider)

			s.RequireAccurate(s.SettledText(provider))
		})
	}
}

// started opens a provider with options of its own, for the tests that are about those
// options rather than about a plain call.
func (s *GeminiIntegrationSuite) started(options Options) *STT {
	provider, err := New(options)
	s.Require().NoError(err)
	s.Start(provider)
	return provider
}
