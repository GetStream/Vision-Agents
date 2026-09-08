//go:build integration

package togethernemotron

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// TogetherNemotronIntegrationSuite inherits what every provider owes a call from sttsuite,
// against Nemotron 3 ASR, the English model an English call gets by default.
type TogetherNemotronIntegrationSuite struct {
	sttsuite.Suite
}

func TestTogetherNemotronIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherNemotronIntegrationSuite{Suite: sttsuite.Suite{
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

// TestTheDocumentedModelIdOpensARealtimeSession is the one thing about this provider that
// could not be settled from Together's documentation. Their realtime Parakeet is a
// suffixed variant of the batch model id, and no equivalent suffix is named anywhere for
// Nemotron, so this asserts that the id their catalogue publishes is one the realtime
// socket will actually open a session on. A session that opened on nothing would look
// like an outage rather than a wrong model string.
func (s *TogetherNemotronIntegrationSuite) TestTheDocumentedModelIdOpensARealtimeSession() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.Equal(DefaultModel, provider.Model())
	s.Equal("nvidia/nemotron-3-asr-streaming-0.6b", provider.Model())
}

// TestAShortTurnSettlesAsTheWholePhraseAndNotItsTail is the "can you hear me" report,
// against the real socket.
//
// Nemotron's decoder flushes on its own schedule and the flush lands wherever it lands:
// speaking for a couple of seconds reliably gets one mid-phrase, after which the deltas
// begin again from nothing and carry only the words since. The bug was publishing that as
// a finished turn, so a four-word question reached the agent as its last word.
//
// A short turn is the case that catches it. Speaking the whole fixture does not: the flush
// falls somewhere in the middle of a long sentence and the turn still settles on plenty of
// words, which scores as accurate while being the wrong half of what was said.
func (s *TogetherNemotronIntegrationSuite) TestAShortTurnSettlesAsTheWholePhraseAndNotItsTail() {
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
	// so every one of these belongs to one run of speech. A flush reported as a new
	// utterance is this provider telling the router the caller started a second sentence,
	// which is what makes the fragment an answer to itself rather than a revision.
	for _, final := range finals {
		s.Equal(finals[0].Utterance, final.Utterance,
			"one uninterrupted phrase should be one utterance, however often the decoder flushed")
	}
}

// shortTurnMs is how much of the fixture makes a turn short enough that the decoder's
// flush lands inside it rather than after it.
const shortTurnMs = 2500

// TogetherNemotronMultilingualIntegrationSuite holds the multilingual model to the same
// bar on the same English fixture. It is a second suite rather than a case in the one
// above because a provider suite is built around one session's worth of configuration,
// and the model is chosen before the session opens.
type TogetherNemotronMultilingualIntegrationSuite struct {
	sttsuite.Suite
}

func TestTogetherNemotronMultilingualIntegrationSuite(t *testing.T) {
	suite.Run(t, &TogetherNemotronMultilingualIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{Model: MultilingualModel})
			require.NoError(t, err)
			return provider
		},
		Requires:       []string{"TOGETHER_API_KEY"},
		SettlesOnClose: true,
	}})
}

// TestTheMultilingualModelIsTheOneTheSessionOpenedOn guards the same thing as its English
// counterpart for the second of the two ids this provider serves.
func (s *TogetherNemotronMultilingualIntegrationSuite) TestTheMultilingualModelIsTheOneTheSessionOpenedOn() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.Equal(MultilingualModel, provider.Model())
	s.Equal("nvidia/nemotron-3.5-asr-streaming-0.6b", provider.Model())
}

// TestTheMultilingualModelStillTranscribesEnglish is worth asserting because the English
// model is the default and this one is what a multilingual call gets: English is in its
// 40 locales, and a routing decision made on the language of the call should not cost the
// caller their transcript.
func (s *TogetherNemotronMultilingualIntegrationSuite) TestTheMultilingualModelStillTranscribesEnglish() {
	provider := s.Started()
	defer s.Hangup(provider)

	s.RequireAccurate(s.SettledText(provider))
}
