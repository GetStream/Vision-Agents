package deepgram

import (
	"testing"

	msginterfaces "github.com/deepgram/deepgram-go-sdk/v3/pkg/api/listen/v2/websocket/interfaces"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type DeepgramSuite struct {
	suite.Suite
}

func TestDeepgramSuite(t *testing.T) {
	suite.Run(t, new(DeepgramSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *DeepgramSuite) newSTT(options Options) *STT {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *DeepgramSuite) drain(provider *STT) []stt.Event {
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

func (s *DeepgramSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("DEEPGRAM_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *DeepgramSuite) TestNewFallsBackToEnvAPIKey() {
	s.T().Setenv("DEEPGRAM_API_KEY", "from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *DeepgramSuite) TestNewDefaultsToEnglishFluxModel() {
	s.Equal(DefaultModel, s.newSTT(Options{}).Model())
}

func (s *DeepgramSuite) TestNewRejectsLanguageHintsOnEnglishModel() {
	_, err := New(Options{APIKey: "k", LanguageHints: []string{"en", "es"}})
	s.ErrorContains(err, "language hints require model flux-general-multi")
}

func (s *DeepgramSuite) TestNewAcceptsLanguageHintsOnMultilingualModel() {
	provider := s.newSTT(Options{Model: MultilingualModel, LanguageHints: []string{"en", "es"}})
	s.Equal(MultilingualModel, provider.Model())
}

func (s *DeepgramSuite) TestProviderIsReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
}

func (s *DeepgramSuite) TestStartOfTurnDoesNotEnterTheSharedContract() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:           msginterfaces.TurnEventStartOfTurn,
		EndOfTurnConfidence: 0.1,
	})

	s.Empty(s.drain(provider))
}

func (s *DeepgramSuite) TestOneRunOfSpeechKeepsOneUtteranceNumber() {
	// Flux revises the same words many times over, and whoever is deciding when to answer
	// has to be able to tell that from the speaker saying them again.
	provider := s.newSTT(Options{})

	for range 3 {
		provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
			EventType:  msginterfaces.TurnEventUpdate,
			Transcript: "hey",
		})
	}
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEndOfTurn,
		Transcript: "Hey.",
	})

	for _, event := range s.drain(provider) {
		transcript, ok := event.(stt.Transcript)
		s.Require().True(ok)
		s.Equal(int64(1), transcript.Utterance)
	}
}

func (s *DeepgramSuite) TestASecondRunOfSpeechIsNumberedApartFromTheFirst() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventUpdate,
		Transcript: "hey",
	})
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEndOfTurn,
		Transcript: "Hey.",
	})
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType: msginterfaces.TurnEventStartOfTurn,
	})
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventUpdate,
		Transcript: "hey",
	})

	events := s.drain(provider)
	s.Require().Len(events, 3)
	s.Equal(int64(1), events[0].(stt.Transcript).Utterance)
	s.Equal(int64(1), events[1].(stt.Transcript).Utterance,
		"the end of a run belongs to the run it ends")
	s.Equal(int64(2), events[2].(stt.Transcript).Utterance,
		"an end followed by a start is one boundary, not two")
}

func (s *DeepgramSuite) TestAResumedTurnStaysTheSameUtterance() {
	// Flux revokes an eager end of turn when the speaker was only pausing, and what
	// follows is the same sentence carrying on rather than a new one.
	provider := s.newSTT(Options{EagerEotThreshold: 0.5})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEagerEndOfTurn,
		Transcript: "book a table",
	})
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{EventType: msginterfaces.TurnEventTurnResumed})
	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventUpdate,
		Transcript: "book a table for four",
	})

	events := s.drain(provider)
	s.Require().Len(events, 2)
	s.Equal(int64(1), events[0].(stt.Transcript).Utterance)
	s.Equal(int64(1), events[1].(stt.Transcript).Utterance)
}

func (s *DeepgramSuite) TestUpdateProducesAReplacementTranscript() {
	provider := s.newSTT(Options{})
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:           msginterfaces.TurnEventUpdate,
		Transcript:          "  hello wor  ",
		EndOfTurnConfidence: 0.4,
		AudioWindowStart:    1.0,
		AudioWindowEnd:      2.5,
		Languages:           []string{"en"},
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)

	s.Equal("hello wor", transcript.Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, transcript.Mode, "Flux updates replace rather than append")
	s.False(transcript.Final())
	s.Equal(speaker, transcript.Participant)
	s.Equal("en", transcript.Language)
	s.Equal(ProviderName, transcript.Provider)
	s.Equal(DefaultModel, transcript.Model)
	s.InDelta(1500.0, transcript.AudioDurationMs, 0.001, "audio window is reported in seconds")
}

func (s *DeepgramSuite) TestEndOfTurnProducesAFinalTranscript() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:           msginterfaces.TurnEventEndOfTurn,
		Transcript:          "hello world",
		EndOfTurnConfidence: 0.95,
		AudioWindowStart:    0,
		AudioWindowEnd:      2,
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)

	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)
	s.Equal("hello world", transcript.Text)
	s.True(transcript.Final())
	s.InDelta(2000.0, transcript.AudioDurationMs, 0.001)
}

func (s *DeepgramSuite) TestEagerEndOfTurnStaysAReplacementTranscript() {
	provider := s.newSTT(Options{EagerEotThreshold: 0.5})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEagerEndOfTurn,
		Transcript: "hello world",
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)

	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)
	s.False(transcript.Final(), "an eager end of turn can still be revoked")
	s.Equal(stt.ModeReplacement, transcript.Mode)
}

func (s *DeepgramSuite) TestTurnResumedDoesNotEnterTheSharedContract() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{EventType: msginterfaces.TurnEventTurnResumed})

	s.Empty(s.drain(provider))
}

func (s *DeepgramSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventUpdate,
		Transcript: "   ",
	})

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *DeepgramSuite) TestEndOfTurnWithoutTextEmitsNothing() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEndOfTurn,
		Transcript: "",
	})

	s.Empty(s.drain(provider))
}

func (s *DeepgramSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *DeepgramSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *DeepgramSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT(Options{})
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *DeepgramSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *DeepgramSuite) TestTransportErrorsAreReported() {
	provider := s.newSTT(Options{})
	handler := &callbacks{stt: provider}

	s.Require().NoError(handler.Error(&msginterfaces.ErrorResponse{ErrCode: "NET-0001", ErrMsg: "socket died"}))

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure.Err, "socket died")
}

func (s *DeepgramSuite) TestTeardownErrorsAreNotReportedAsFailures() {
	provider := s.newSTT(Options{})
	handler := &callbacks{stt: provider}

	// Close, then deliver the read error that closing the socket provokes.
	provider.mu.Lock()
	provider.closed = true
	provider.mu.Unlock()

	s.Require().NoError(handler.Error(&msginterfaces.ErrorResponse{ErrCode: "NET-0001", ErrMsg: "connection closed"}))

	s.Empty(s.drain(provider), "our own teardown should not be billed as a failed request")
}

func (s *DeepgramSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT(Options{})
}
