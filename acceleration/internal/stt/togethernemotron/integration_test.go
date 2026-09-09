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
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
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

// TestASecondTurnDoesNotBeginWithTheEndOfTheFirst is the "boulder" report against the real
// socket: a word ending one sentence opened the next one, which the caller never said
// twice.
//
// It is the server's segmentation showing through. A segment does not end when the caller
// stops talking - it runs on into whatever they say next, and its deltas restate it from
// the beginning, so the tail of the finished turn is still on the front of every delta of
// the new one. Two turns of different fixtures is what makes the leak legible: none of the
// words of the first belong in the second.
func (s *TogetherNemotronIntegrationSuite) TestASecondTurnDoesNotBeginWithTheEndOfTheFirst() {
	second, err := testaudio.Load16kMono("saturday_seven_thirty.wav")
	s.Require().NoError(err)

	provider := s.Started()

	collected := make(chan []stt.Transcript, 1)
	go func() {
		var finals []stt.Transcript
		for event := range provider.Events() {
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				finals = append(finals, transcript)
			}
		}
		collected <- finals
	}()

	s.Speak(provider, 0)
	s.Quiet(provider)
	s.stream(provider, second)
	s.Quiet(provider)
	s.Hangup(provider)

	var finals []stt.Transcript
	select {
	case finals = <-collected:
	case <-time.After(90 * time.Second):
		s.FailNow("the event channel was not closed")
	}

	s.Require().Len(finals, 2, "two turns separated by silence are two turns")
	for i, final := range finals {
		s.T().Logf("turn %d (utterance %d): %q", i+1, final.Utterance, final.Text)
	}

	s.NotEqual(finals[0].Utterance, finals[1].Utterance, "a second turn is a second utterance")
	s.Contains(strings.ToLower(finals[0].Text), s.Opening())

	// The leak, in the words it would arrive as. The reported symptom is the first turn's
	// last word opening the second, so what the second begins with is the assertion; the
	// distinctive words guard against more of the first turn than its tail coming over.
	// Common words are no use here, since a booking says "a" and "the" too.
	booking := strings.ToLower(finals[1].Text)
	s.True(strings.HasPrefix(booking, "hi"),
		"the second turn should begin with its own first word, not the last of the first")
	for _, word := range []string{"village", "treasures", "gold"} {
		s.NotContains(booking, word, "the second turn should hold none of the first")
	}
	s.Contains(booking, "book a table")
}

// stream sends audio the suite did not load, at the pace a call delivers it.
func (s *TogetherNemotronIntegrationSuite) stream(provider stt.STT, audio stt.PcmData) {
	chunks := testaudio.Chunks(audio, s.ChunkMs)
	started := time.Now()
	for i, chunk := range chunks {
		if wait := time.Until(started.Add(time.Duration(i*s.ChunkMs) * time.Millisecond)); wait > 0 {
			time.Sleep(wait)
		}
		s.Require().NoError(provider.ProcessAudio(chunk, stt.Participant{ID: "p", UserID: "u"}))
	}
}

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
