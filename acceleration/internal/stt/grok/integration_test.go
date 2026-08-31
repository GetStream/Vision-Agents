//go:build integration

package grok

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// GrokIntegrationSuite inherits what every provider owes a call from sttsuite. xAI ask for
// 100ms of audio at a time, which is what the suite sends by default, and end a turn on
// 400ms of silence of their own accord.
type GrokIntegrationSuite struct {
	sttsuite.Suite
}

func TestGrokIntegrationSuite(t *testing.T) {
	suite.Run(t, &GrokIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"XAI_API_KEY"},
		// audio.done makes the server transcribe what it is still holding.
		SettlesOnClose: true,
	}})
}

// TestTheFinalReportsTheAudioItCovered is what usage is charted against: this provider is
// billed by the hour of audio, so a turn that does not say how much it covered cannot be
// costed.
func (s *GrokIntegrationSuite) TestTheFinalReportsTheAudioItCovered() {
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

// TestSmartTurnHoldsOffOnAPauseMidSentence is the option worth having on a call. Without
// it, 400ms of silence ends the turn, so a caller thinking between clauses gets answered
// mid-sentence. The turn should still settle, and it should still settle promptly.
func (s *GrokIntegrationSuite) TestSmartTurnHoldsOffOnAPauseMidSentence() {
	provider, err := New(Options{SmartTurn: 0.7, SmartTurnTimeoutMs: 3000})
	s.Require().NoError(err)
	s.Start(provider)
	defer s.Hangup(provider)

	timing := s.MeasureOn(provider)

	s.RequireAccurate(timing.Text)
	s.Less(timing.ToSettle, s.MaxSettle,
		"holding off on a mid-sentence pause should not stop the turn settling at the end")
}
