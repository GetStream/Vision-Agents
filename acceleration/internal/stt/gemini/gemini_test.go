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

// heard is a frame carrying more of what the caller is saying.
func heard(text string) serverMessage {
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

func (s *GeminiSuite) TestATranscriptionChunkAppendsRatherThanReplacing() {
	provider := s.newSTT()

	provider.handleMessage(heard("in a quiet "))
	provider.handleMessage(heard("village"))

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2)
	s.Equal(stt.ModeDelta, said[0].Mode, "Gemini sends the words in pieces that append")
	s.Equal("in a quiet ", said[0].Text)
	s.Equal("village", said[1].Text)
	s.Equal(ProviderName, said[0].Provider)
	s.Equal(DefaultModel, said[0].Model)
}

func (s *GeminiSuite) TestTheEndOfATurnSettlesTheWholeUtterance() {
	provider := s.newSTT()

	provider.handleMessage(heard("in a quiet "))
	provider.handleMessage(heard("village"))
	provider.handleMessage(stopped())

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 3)
	s.True(said[2].Final())
	s.Equal("in a quiet village", said[2].Text,
		"the final is the one frame carrying the utterance as a whole")
}

func (s *GeminiSuite) TestTheNextTurnStartsAfreshRatherThanCarryingWordsOver() {
	provider := s.newSTT()

	provider.handleMessage(heard("hello"))
	provider.handleMessage(stopped())
	provider.handleMessage(heard("goodbye"))
	provider.handleMessage(stopped())

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 4)
	s.Equal("goodbye", said[3].Text, "the second turn must not repeat the first")
}

func (s *GeminiSuite) TestTheDeltasOfATurnShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT()

	provider.handleMessage(heard("in a quiet "))
	provider.handleMessage(heard("village"))
	provider.handleMessage(stopped())
	provider.handleMessage(heard("forgotten"))

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

	provider.handleMessage(heard(""))

	s.Empty(s.drain(provider))
}

func (s *GeminiSuite) TestBeingInterruptedStillSettlesWhatWasHeard() {
	provider := s.newSTT()

	provider.handleMessage(heard("wait actually"))
	provider.handleMessage(serverMessage{ServerContent: &serverContent{Interrupted: true}})

	said := transcripts(s.drain(provider))
	s.Require().Len(said, 2)
	s.True(said[1].Final(), "words cut short are still words that were said")
}

func (s *GeminiSuite) TestAWarningThatTheSessionIsEndingIsNotATranscript() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{GoAway: &goAway{TimeLeft: "30s"}})

	s.Empty(s.drain(provider))
}

func (s *GeminiSuite) TestKeytermsAreToldToTheModelBecauseTheresNoFieldForThem() {
	provider, err := New(Options{APIKey: "k", Keyterms: []string{"Vision Agents", "  ", "Stream"}})
	s.Require().NoError(err)

	said := provider.instruction()
	s.Require().NotNil(said)
	s.Contains(said.Parts[0].Text, "Vision Agents, Stream",
		"a term nobody meant to add should not be passed on")
}

func (s *GeminiSuite) TestAVoiceWithNothingToSayIsGivenNoInstruction() {
	s.Nil(s.newSTT().instruction(), "an empty instruction is worse than none")
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
