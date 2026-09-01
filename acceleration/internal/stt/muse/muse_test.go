package muse

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type MuseSuite struct {
	suite.Suite
}

func TestMuseSuite(t *testing.T) {
	suite.Run(t, new(MuseSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *MuseSuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key"})
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// transcripts collects the transcripts emitted so far without blocking on an empty
// channel.
func (s *MuseSuite) transcripts(provider *STT) []stt.Transcript {
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

func (s *MuseSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *MuseSuite) TestNewFallsBackToEnv() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultURL, provider.options.URL)
}

func (s *MuseSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.meta.ai/v1/asr/realtime"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *MuseSuite) TestNewRejectsAModeTheServerDoesNotKnow() {
	_, err := New(Options{APIKey: "k", Mode: "AUTOMATIC"})
	s.ErrorContains(err, "mode must be one of")
}

func (s *MuseSuite) TestNewListensForTurnBoundariesUnlessToldOtherwise() {
	// Push to talk leaves the turn boundaries to the caller, which on a call means
	// nothing ever settles.
	provider := s.newSTT()
	s.Equal(ModeEndpointing, provider.options.Mode)
}

func (s *MuseSuite) TestNewDropsBlankKeyterms() {
	provider, err := New(Options{APIKey: "k", Keyterms: []string{" eSIM ", "  ", ""}})
	s.Require().NoError(err)
	s.Equal([]string{"eSIM"}, provider.options.Keyterms)
}

func (s *MuseSuite) TestLanguageHintsAreRenderedTheWayTheApiNamesLanguages() {
	// The rest of the router speaks ISO codes and this API names languages in full, so a
	// hint passed straight through would be ignored rather than rejected.
	s.Equal([]string{"English", "Mandarin Chinese"}, languageBias([]string{"en", "ZH "}))
}

func (s *MuseSuite) TestALanguageNamedOutrightIsLeftAlone() {
	s.Equal([]string{"French"}, languageBias([]string{"French"}))
}

func (s *MuseSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *MuseSuite) TestPartialsRestateTheTurnRatherThanAppendToIt() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 1})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "how is"})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "how is the weather"})

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

func (s *MuseSuite) TestSpeechCompleteSettlesTheTurn() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 1})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "how is the weather"})
	provider.handleMessage(serverMessage{Type: messageSpeechEnd, TurnID: 1})
	provider.handleMessage(serverMessage{
		Type: messageSpeechComplete, TurnID: 1, Transcript: "How is the weather?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.False(heard[0].Final(), "speechEnd is a boundary, not a transcript")
	s.True(heard[1].Final())
	s.Equal("How is the weather?", heard[1].Text)
}

func (s *MuseSuite) TestATurnSettlesOnceEvenWhenBothFramesSayItHas() {
	// The server flags the last transcript final and then repeats the settled turn in
	// speechComplete. Emitting both would tell the call the caller said it twice.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 1})
	provider.handleMessage(serverMessage{
		Type: messageTranscript, Transcript: "How is the weather?", Final: true,
	})
	provider.handleMessage(serverMessage{
		Type: messageSpeechComplete, TurnID: 1, Transcript: "How is the weather?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.True(heard[0].Final())
}

func (s *MuseSuite) TestEachRunOfSpeechIsNumberedOnce() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 1})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "hello"})
	provider.handleMessage(serverMessage{Type: messageSpeechComplete, TurnID: 1, Transcript: "Hello."})
	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 2})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "goodbye"})
	provider.handleMessage(serverMessage{Type: messageSpeechComplete, TurnID: 2, Transcript: "Goodbye."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal([]int64{1, 1, 2, 2}, []int64{
		heard[0].Utterance, heard[1].Utterance, heard[2].Utterance, heard[3].Utterance,
	})
}

func (s *MuseSuite) TestASecondTurnStillSettlesWhenItsStartWasNeverAnnounced() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeechStart, TurnID: 1})
	provider.handleMessage(serverMessage{Type: messageSpeechComplete, TurnID: 1, Transcript: "Hello."})
	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "goodbye"})
	provider.handleMessage(serverMessage{Type: messageSpeechComplete, TurnID: 2, Transcript: "Goodbye."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.True(heard[2].Final(), "the second turn settled, so it should be reported as settled")
	s.Equal(int64(2), heard[2].Utterance)
}

func (s *MuseSuite) TestTheTranscriptReportsTheAudioItsOwnTurnCovers() {
	// The server counts audio from the start of the session, so the count alone would
	// have every turn look longer than the last.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		Type: messageSpeechStart, TurnID: 2, AudioProcessedMs: 9000,
	})
	provider.handleMessage(serverMessage{
		Type: messageSpeechComplete, TurnID: 2, Transcript: "Goodbye.", AudioProcessedMs: 11400,
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal(float64(2400), heard[0].AudioDurationMs)
}

func (s *MuseSuite) TestEmptyTranscriptsAreNotReported() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageTranscript, Transcript: "   "})
	provider.handleMessage(serverMessage{Type: messageSpeechComplete, TurnID: 1})

	s.Empty(s.transcripts(provider))
}

func (s *MuseSuite) TestAServerErrorEndsTheSession() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageError, Message: "invalid api key"})

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure, "invalid api key")
	s.True(failure.Fatal)
}

func (s *MuseSuite) TestDiarisedSpeakersDoNotEnterTheSharedContract() {
	// The router already knows who is speaking from the track it fed the audio in on.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageSpeaker, Label: "B"})
	provider.handleMessage(serverMessage{Type: messageAudioProgress, AudioProcessedMs: 2400})

	s.Empty(s.transcripts(provider))
}

func (s *MuseSuite) TestAudioBeforeStartIsRefused() {
	provider := s.newSTT()

	err := provider.ProcessAudio(
		stt.PcmData{Samples: []int16{1, 2}, SampleRate: stt.SampleRate, Channels: 1},
		stt.Participant{ID: "alice"},
	)

	s.ErrorContains(err, "not started")
}
