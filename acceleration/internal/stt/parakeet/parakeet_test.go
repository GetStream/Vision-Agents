package parakeet

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type ParakeetSuite struct {
	suite.Suite
}

func TestParakeetSuite(t *testing.T) {
	suite.Run(t, new(ParakeetSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *ParakeetSuite) newSTT() *STT {
	provider, err := New(Options{URL: "wss://example.invalid/websocket", APIKey: "test-key"})
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *ParakeetSuite) drain(provider *STT) []stt.Event {
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

func (s *ParakeetSuite) TestNewRequiresURL() {
	s.T().Setenv("PARAKEET_WS_URL", "")

	_, err := New(Options{APIKey: "k"})
	s.ErrorContains(err, "websocket url is required")
}

func (s *ParakeetSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("PARAKEET_WS_URL", "wss://from-env/websocket")
	s.T().Setenv("BASETEN_API_KEY", "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("wss://from-env/websocket", provider.options.URL)
	s.Equal("key-from-env", provider.options.APIKey)
}

func (s *ParakeetSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{URL: "https://example.invalid/websocket", APIKey: "k"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *ParakeetSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("BASETEN_API_KEY", "")

	_, err := New(Options{URL: "wss://example.invalid/websocket"})
	s.ErrorContains(err, "api key is required")
}

func (s *ParakeetSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *ParakeetSuite) TestStartOfTurnDoesNotEnterTheSharedContract() {
	provider := s.newSTT()
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleMessage(serverMessage{Type: messageStartOfTurn})

	s.Empty(s.drain(provider))
}

func (s *ParakeetSuite) TestPartialProducesAReplacementTranscript() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		Type:             messagePartial,
		Text:             "  in a quiet village ",
		AudioDurationMs:  1500,
		ProcessingTimeMs: 120,
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)

	s.Equal("in a quiet village", transcript.Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, transcript.Mode, "Parakeet re-decodes the whole utterance")
	s.False(transcript.Final())
	s.Equal(ProviderName, transcript.Provider)
	s.Equal(DefaultModel, transcript.Model)
	s.InDelta(1500.0, transcript.AudioDurationMs, 0.001)
	s.InDelta(120.0, transcript.ProcessingTimeMs, 0.001, "the server's own decode time is preferred")
}

func (s *ParakeetSuite) TestFinalProducesAFinalTranscript() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		Type:            messageFinal,
		Text:            "forgotten treasures",
		AudioDurationMs: 9800,
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)

	transcript, ok := events[0].(stt.Transcript)
	s.Require().True(ok)
	s.Equal("forgotten treasures", transcript.Text)
	s.True(transcript.Final())
	s.InDelta(9800.0, transcript.AudioDurationMs, 0.001)
}

func (s *ParakeetSuite) TestPartialsShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messagePartial, Text: "in a quiet"})
	provider.handleMessage(serverMessage{Type: messagePartial, Text: "in a quiet village"})
	provider.handleMessage(serverMessage{Type: messageFinal, Text: "In a quiet village."})
	provider.handleMessage(serverMessage{Type: messageStartOfTurn})
	provider.handleMessage(serverMessage{Type: messagePartial, Text: "forgotten"})

	events := s.drain(provider)
	s.Require().Len(events, 4)
	s.Equal(int64(1), events[0].(stt.Transcript).Utterance)
	s.Equal(int64(1), events[1].(stt.Transcript).Utterance)
	s.Equal(int64(1), events[2].(stt.Transcript).Utterance)
	s.Equal(int64(2), events[3].(stt.Transcript).Utterance,
		"a final followed by a start of turn is one boundary, not two")
}

func (s *ParakeetSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messagePartial, Text: "   "})

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *ParakeetSuite) TestServerErrorIsFatal() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: messageError, Error: "sample_rate must be 16000"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "sample_rate must be 16000")
}

func (s *ParakeetSuite) TestFinishedStatusEmitsNothing() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Status: statusFinished})

	s.Empty(s.drain(provider), "the flush acknowledgement is not an STT event")
}

func (s *ParakeetSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT()

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *ParakeetSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT()

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *ParakeetSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT()
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *ParakeetSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT()

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *ParakeetSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT()
}
