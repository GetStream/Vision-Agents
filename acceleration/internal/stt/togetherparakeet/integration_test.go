//go:build integration

package togetherparakeet

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// TogetherParakeetIntegrationSuite inherits what every provider owes a call from sttsuite.
// Together's own examples send roughly 250ms of audio at a time; the suite's 100ms is
// closer to what a call delivers and is what this is held to.
type TogetherParakeetIntegrationSuite struct {
	sttsuite.Suite
}

func TestTogetherParakeetIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherParakeetIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{"TOGETHER_API_KEY"},
		// Committing the buffer makes the server transcribe what it is still holding.
		SettlesOnClose: true,
	}})
}

// TestAShortTurnSettlesAsTheWholePhraseAndNotItsTail is the "can you hear me" report,
// against the real socket. It is the same decoder and the same defect as Nemotron's: the
// flush lands mid-phrase, the deltas after it carry only the words since, and publishing
// that as a finished turn reached the agent as the last word of the question.
//
// A short turn is the case that catches it. Speaking the whole fixture does not: the flush
// falls somewhere in the middle of a long sentence and the turn still settles on plenty of
// words, which scores as accurate while being the wrong half of what was said.
func (s *TogetherParakeetIntegrationSuite) TestAShortTurnSettlesAsTheWholePhraseAndNotItsTail() {
	provider := s.Started()

	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
		}
		collected <- events
	}()

	s.Speak(provider, shortTurnMs/s.ChunkMs)
	s.Quiet(provider)
	s.Hangup(provider)

	var events []stt.Event
	select {
	case events = <-collected:
	case <-time.After(90 * time.Second):
		s.FailNow("the event channel was not closed")
	}

	var finals []stt.Transcript
	for _, event := range events {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			finals = append(finals, transcript)
		}
	}
	s.Require().NotEmpty(finals, "a short turn never settled")

	settled := finals[len(finals)-1].Text
	s.T().Logf("%d finals, settled on %q", len(finals), settled)
	s.Contains(strings.ToLower(settled), s.Opening(),
		"the turn should settle on the whole phrase, opening words included, rather than "+
			"on the fragment after the decoder's flush")

	// The other half of the same bug, and the half the router acts on. Nothing here paused,
	// so every one of these belongs to one run of speech.
	for _, final := range finals {
		s.Equal(finals[0].Utterance, final.Utterance,
			"one uninterrupted phrase should be one utterance, however often the decoder flushed")
	}
}

// shortTurnMs is how much of the fixture makes a turn short enough that the decoder's
// flush lands inside it rather than after it.
const shortTurnMs = 2500

// TestTheModelIsTheStreamingOneRatherThanTheBatchOne is worth an assertion because the two
// differ only by a suffix, and the batch model cannot serve a call. A session that opened
// on the wrong one would look like a slow provider rather than a misconfigured one.
func (s *TogetherParakeetIntegrationSuite) TestTheModelIsTheStreamingOneRatherThanTheBatchOne() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.Equal(DefaultModel, provider.Model())
	s.Contains(provider.Model(), "-realtime")
}
