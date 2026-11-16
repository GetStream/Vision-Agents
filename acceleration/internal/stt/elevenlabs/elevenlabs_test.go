package elevenlabs

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type ElevenlabsSuite struct {
	suite.Suite
}

func TestElevenlabsSuite(t *testing.T) {
	suite.Run(t, new(ElevenlabsSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *ElevenlabsSuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key"})
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// transcripts collects the transcripts emitted so far without blocking on an empty
// channel.
func (s *ElevenlabsSuite) transcripts(provider *STT) []stt.Transcript {
	var heard []stt.Transcript
	for {
		select {
		case event := <-provider.Events():
			if transcript, ok := event.(stt.Transcript); ok {
				heard = append(heard, transcript)
			}
		default:
			return heard
		}
	}
}

func (s *ElevenlabsSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *ElevenlabsSuite) TestNewFallsBackToEnv() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultURL, provider.options.URL)
	s.Equal(DefaultModel, provider.options.Model)
}

func (s *ElevenlabsSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.elevenlabs.io/v1/speech-to-text/realtime"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *ElevenlabsSuite) TestNewLetsTheServerFindTheTurnBoundaryUnlessToldOtherwise() {
	// Committing manually means committing on a signal the router does not have: it has
	// no voice activity detector of its own, so nothing would ever settle.
	provider := s.newSTT()
	s.Equal(CommitOnVAD, provider.options.CommitStrategy)
}

func (s *ElevenlabsSuite) TestNewRejectsACommitStrategyTheServerDoesNotKnow() {
	_, err := New(Options{APIKey: "k", CommitStrategy: "automatic"})
	s.ErrorContains(err, "commit strategy must be")
}

func (s *ElevenlabsSuite) TestNewDropsBlankKeyterms() {
	provider, err := New(Options{APIKey: "k", Keyterms: []string{" Scribe ", "  ", ""}})
	s.Require().NoError(err)
	s.Equal([]string{"Scribe"}, provider.options.Keyterms)
}

func (s *ElevenlabsSuite) TestNewRefusesMoreKeytermsThanTheRealtimeModelTakes() {
	// The batch model takes a thousand of up to fifty characters and this one takes fifty
	// of up to twenty, so a list the rest of the router considers small is refused here
	// rather than by a failed dial.
	terms := make([]string, maxKeyterms+1)
	for i := range terms {
		terms[i] = "term"
	}

	_, err := New(Options{APIKey: "k", Keyterms: terms})
	s.ErrorContains(err, "at most 50 keyterms")
}

func (s *ElevenlabsSuite) TestNewRefusesAKeytermLongerThanTheModelTakes() {
	_, err := New(Options{APIKey: "k", Keyterms: []string{strings.Repeat("a", maxKeytermRunes+1)}})
	s.ErrorContains(err, "keyterms are at most 20 characters")
}

func (s *ElevenlabsSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *ElevenlabsSuite) TestPartialsRestateTheSegmentRatherThanAppendToIt() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "how is"})
	provider.handleMessage(serverMessage{
		MessageType: eventPartialTranscript, Text: "how is the weather",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	for _, transcript := range heard {
		s.Equal(stt.ModeReplacement, transcript.Mode)
		s.Equal("alice", transcript.Participant.UserID)
		s.Equal(ProviderName, transcript.Provider)
		s.Equal(DefaultModel, transcript.Model)
	}
	s.Equal("how is the weather", heard[1].Text)
}

func (s *ElevenlabsSuite) TestACommittedTranscriptSettlesTheTurn() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "how is"})
	provider.handleMessage(serverMessage{
		MessageType: eventCommittedTranscript, Text: "How is the weather?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.False(heard[0].Final())
	s.True(heard[1].Final())
	s.Equal("How is the weather?", heard[1].Text)
}

func (s *ElevenlabsSuite) TestASegmentAnnotatedAfterwardsIsNotReportedTwice() {
	// Asking for timestamps or entities makes the server restate a committed segment in
	// a second frame. Reporting both would tell the rest of the call the caller said it
	// twice.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		MessageType: eventCommittedTranscript, Text: "How is the weather?",
	})
	provider.handleMessage(serverMessage{
		MessageType:  eventCommittedTranscriptWithTimings,
		Text:         "How is the weather?",
		LanguageCode: "en",
	})
	provider.handleMessage(serverMessage{
		MessageType: eventCommittedTranscriptEntities, Text: "How is the weather?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.True(heard[0].Final())
}

func (s *ElevenlabsSuite) TestEachRunOfSpeechIsNumberedOnce() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "hello"})
	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: "Hello."})
	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "goodbye"})
	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: "Goodbye."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal([]int64{1, 1, 2, 2}, []int64{
		heard[0].Utterance, heard[1].Utterance, heard[2].Utterance, heard[3].Utterance,
	})
}

func (s *ElevenlabsSuite) TestTheSameWordsSaidTwiceAreTwoTurns() {
	// The dedupe is about one segment being restated, not about a caller repeating
	// themselves, and a partial in between is what tells the two apart.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: "Yes."})
	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "yes"})
	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: "Yes."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(2), heard[2].Utterance)
	s.True(heard[2].Final())
}

func (s *ElevenlabsSuite) TestTheDetectedLanguageIsCarriedWhenTheServerSaysIt() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		MessageType:  eventCommittedTranscriptWithTimings,
		Text:         "How is the weather?",
		LanguageCode: "en",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("en", heard[0].Language)
}

func (s *ElevenlabsSuite) TestEmptyTranscriptsAreNotReported() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventPartialTranscript, Text: "   "})
	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: ""})

	s.Empty(s.transcripts(provider))
}

func (s *ElevenlabsSuite) TestACommitOfNothingStillAnswersAWaitingClose() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{MessageType: eventCommittedTranscript, Text: ""})

	select {
	case <-provider.settled:
	default:
		s.Fail("a commit that settled nothing should still end the wait for the tail")
	}
}

func (s *ElevenlabsSuite) TestAnAuthFailureEndsTheSession() {
	// Scribe has a failure frame per reason and they all carry it in the same field, so
	// reading them by the field rather than by name reports one added later too.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		MessageType: "auth_error", Error: "invalid api key",
	})

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure, "invalid api key")
	s.True(failure.Fatal)
	s.Equal("auth_error", failure.Context)
}

func (s *ElevenlabsSuite) TestOneRefusedCommitDoesNotEndTheSession() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		MessageType: eventCommitThrottled, Error: "too many commits",
	})

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.False(failure.Fatal, "the caller is still talking and the session can still hear them")
}

func (s *ElevenlabsSuite) TestAudioBeforeStartIsRefused() {
	provider := s.newSTT()

	err := provider.ProcessAudio(
		stt.PcmData{Samples: []int16{1, 2}, SampleRate: stt.SampleRate, Channels: 1},
		stt.Participant{ID: "alice"},
	)

	s.ErrorContains(err, "not started")
}

func (s *ElevenlabsSuite) TestSilenceIsTwoBytesPerSample() {
	s.Len(silence(20), 2*stt.SampleRate*20/1000)
}
