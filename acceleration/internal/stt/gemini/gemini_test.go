package gemini

import (
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type GeminiSuite struct {
	suite.Suite
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *GeminiSuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key", Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *GeminiSuite) drain(provider *STT) []stt.Event {
	var events []stt.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		default:
			return events
		}
	}
}

// hearing is a frame carrying the server's hypothesis of the turn so far.
func hearing(text string) serverMessage {
	return serverMessage{ServerContent: &serverContent{
		InterimInputTranscription: &transcription{Text: text},
	}}
}

// settled is a frame carrying the transcript the server has committed to.
func settled(text string) serverMessage {
	return serverMessage{ServerContent: &serverContent{InputTranscription: &transcription{Text: text}}}
}

// stopped is the frame that ends a turn.
func stopped() serverMessage {
	return serverMessage{ServerContent: &serverContent{TurnComplete: true}}
}

// transcripts narrows drained events to the transcripts among them.
func transcripts(events []stt.Event) []stt.Transcript {
	var found []stt.Transcript
	for _, event := range events {
		if transcript, ok := event.(stt.Transcript); ok {
			found = append(found, transcript)
		}
	}
	return found
}

func (s *GeminiSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("GOOGLE_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *GeminiSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("GOOGLE_API_KEY", "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
}

func (s *GeminiSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://example.invalid/live"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *GeminiSuite) TestNewRejectsMoreKeytermsThanTheContractAllows() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = "term"
	}

	_, err := New(Options{APIKey: "k", Keyterms: terms})
	s.ErrorContains(err, "keyterms")
}

func (s *GeminiSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *GeminiSuite) TestTheKeyTravelsOnTheQueryStringBecauseTheresNowhereElse() {
	provider, err := New(Options{APIKey: "sec ret"})
	s.Require().NoError(err)

	s.Contains(provider.endpoint(), "key=sec+ret", "the key must be escaped, not pasted")
	s.True(strings.HasPrefix(provider.endpoint(), DefaultURL+"?"))
}

func (s *GeminiSuite) TestAHypothesisReplacesTheOneBeforeItRatherThanAddingToIt() {
	provider := s.newSTT()

	provider.handleMessage(hearing("in a"))
	provider.handleMessage(hearing("in a quiet village"))

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2)
	s.Equal(stt.ModeReplacement, said[0].Mode, "each hypothesis restates the turn so far")
	s.Equal("in a", said[0].Text)
	s.Equal("in a quiet village", said[1].Text)
	s.False(said[1].Final(), "the caller has not stopped talking yet")
	s.Equal(ProviderName, said[0].Provider)
	s.Equal(DefaultModel, said[0].Model)
}

func (s *GeminiSuite) TestTheFinalizedTranscriptSettlesTheTurn() {
	provider := s.newSTT()

	provider.handleMessage(hearing("in a quiet"))
	provider.handleMessage(settled("In a quiet village."))
	provider.handleMessage(stopped())

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2, "the boundary after a finalized turn settles nothing further")
	s.True(said[1].Final())
	s.Equal("In a quiet village.", said[1].Text)
}

func (s *GeminiSuite) TestTheNextTurnStartsAfreshRatherThanCarryingWordsOver() {
	provider := s.newSTT()

	provider.handleMessage(settled("hello"))
	provider.handleMessage(settled("goodbye"))

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2)
	s.Equal("goodbye", said[1].Text, "the second turn must not repeat the first")
}

func (s *GeminiSuite) TestTheHypothesesOfATurnShareTheUtteranceOfTheFinalTheySettleInto() {
	provider := s.newSTT()

	provider.handleMessage(hearing("in a"))
	provider.handleMessage(hearing("in a quiet village"))
	provider.handleMessage(settled("In a quiet village."))
	provider.handleMessage(hearing("forgotten"))

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 4)
	s.Equal(int64(1), said[0].Utterance)
	s.Equal(int64(1), said[1].Utterance)
	s.Equal(int64(1), said[2].Utterance)
	s.Equal(int64(2), said[3].Utterance, "the words after a settled turn are a new one")
}

func (s *GeminiSuite) TestATurnBoundaryWithNothingBeforeItIsNotATurn() {
	provider := s.newSTT()

	provider.handleMessage(stopped())

	s.Empty(s.drain(provider), "silence between turns is not something the caller said")
}

func (s *GeminiSuite) TestAnEmptyChunkIsNotEmitted() {
	provider := s.newSTT()

	provider.handleMessage(hearing(""))
	provider.handleMessage(settled("   "))

	s.Empty(s.drain(provider))
}

func (s *GeminiSuite) TestBeingInterruptedStillSettlesWhatWasHeard() {
	provider := s.newSTT()

	provider.handleMessage(hearing("wait actually"))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{Interrupted: true}})

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2)
	s.True(said[1].Final(), "words cut short are still words that were said")
	s.Equal("wait actually", said[1].Text)
}

func (s *GeminiSuite) TestAWarningThatTheSessionIsEndingIsNotATranscript() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{GoAway: &goAway{TimeLeft: "30s"}})

	s.Empty(s.drain(provider))
}

func (s *GeminiSuite) TestKeytermsBecomeTheCustomVocabulary() {
	provider, err := New(Options{
		APIKey:        "k",
		Keyterms:      []string{"Vision Agents", "  ", "Stream"},
		LanguageHints: []string{"en-US"},
		Mode:          ModeSmart,
	})
	s.Require().NoError(err)

	asked := provider.transcription()
	s.Equal([]string{"Vision Agents", "Stream"}, asked.CustomVocabulary,
		"a term nobody meant to add should not take up one of the places")
	s.Equal([]string{"en-US"}, asked.LanguageCodes)
	s.Equal("SMART", asked.Mode)
}

func (s *GeminiSuite) TestTranscriptionIsAskedForEvenWithNothingToConfigure() {
	asked := s.newSTT().transcription()

	s.Require().NotNil(asked, "sending this at all is what turns transcription on")
	s.Empty(asked.CustomVocabulary)
	s.Empty(asked.LanguageCodes)
	s.Empty(asked.Mode, "an unset mode leaves the server on its own default")
}

func (s *GeminiSuite) TestNewRejectsAModeTheAPIDoesNotHave() {
	_, err := New(Options{APIKey: "k", Mode: "smart"})
	s.ErrorContains(err, "mode must be VERBATIM or SMART")
}

func (s *GeminiSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT()

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *GeminiSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT()

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *GeminiSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT()
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *GeminiSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT()

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *GeminiSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT()
}
