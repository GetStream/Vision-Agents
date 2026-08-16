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

func (s *DeepgramSuite) TestEagerTurnDetectionGetsAThreshold() {
	// Asking for eager turns without a threshold is meaningless, so one is supplied.
	s.Equal(defaultEagerEotThreshold, s.newSTT(Options{EagerTurnDetection: true}).options.EagerEotThreshold)

	// An explicit threshold is respected.
	s.Equal(0.8, s.newSTT(Options{EagerTurnDetection: true, EagerEotThreshold: 0.8}).options.EagerEotThreshold)

	// Without eager turn detection there is no threshold at all.
	s.Zero(s.newSTT(Options{}).options.EagerEotThreshold)
}

func (s *DeepgramSuite) TestProviderAndTurnDetectionAreReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
	s.True(provider.TurnDetection(), "Flux detects turns server-side")
}

func (s *DeepgramSuite) TestStartOfTurnBeginsATurnWithoutATranscript() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:           msginterfaces.TurnEventStartOfTurn,
		EndOfTurnConfidence: 0.1,
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	started, ok := events[0].(stt.TurnStarted)
	s.Require().True(ok)
	s.InDelta(0.1, started.Confidence, 0.001)
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

func (s *DeepgramSuite) TestEndOfTurnProducesAFinalTranscriptAndEndsTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:           msginterfaces.TurnEventEndOfTurn,
		Transcript:          "hello world",
		EndOfTurnConfidence: 0.95,
		AudioWindowStart:    0,
		AudioWindowEnd:      2,
	})

	events := s.drain(provider)
	s.Require().Len(events, 2)

	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)
	s.Equal("hello world", transcript.Text)
	s.True(transcript.Final())

	ended, ok := events[1].(stt.TurnEnded)
	s.Require().True(ok)
	s.False(ended.Eager, "a real end of turn is not eager")
	s.InDelta(0.95, ended.Confidence, 0.001)
	s.InDelta(2000.0, ended.DurationMs, 0.001)
}

func (s *DeepgramSuite) TestEagerEndOfTurnIsMarkedEagerAndStaysNonFinal() {
	provider := s.newSTT(Options{EagerTurnDetection: true})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEagerEndOfTurn,
		Transcript: "hello world",
	})

	events := s.drain(provider)
	s.Require().Len(events, 2)

	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)
	s.False(transcript.Final(), "an eager end of turn can still be revoked")
	s.Equal(stt.ModeReplacement, transcript.Mode)

	ended, ok := events[1].(stt.TurnEnded)
	s.Require().True(ok)
	s.True(ended.Eager)
}

func (s *DeepgramSuite) TestTurnResumedReopensTheTurn() {
	provider := s.newSTT(Options{EagerTurnDetection: true})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{EventType: msginterfaces.TurnEventTurnResumed})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	_, ok := events[0].(stt.TurnStarted)
	s.True(ok, "resumed speech should look like a turn starting again")
}

func (s *DeepgramSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventUpdate,
		Transcript: "   ",
	})

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *DeepgramSuite) TestEndOfTurnWithoutTextStillEndsTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleTurnInfo(&msginterfaces.TurnInfoResponse{
		EventType:  msginterfaces.TurnEventEndOfTurn,
		Transcript: "",
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	_, ok := events[0].(stt.TurnEnded)
	s.True(ok)
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
