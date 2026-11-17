package inworld

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type InworldSuite struct {
	suite.Suite
}

func TestInworldSuite(t *testing.T) {
	suite.Run(t, new(InworldSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *InworldSuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key"})
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// heard is the frame the server sends for one transcription result.
func heard(text string, final bool) serverMessage {
	return serverMessage{Result: &serverResult{
		Transcription: &transcription{Transcript: text, IsFinal: final},
	}}
}

// transcripts collects the transcripts emitted so far without blocking on an empty
// channel.
func (s *InworldSuite) transcripts(provider *STT) []stt.Transcript {
	var collected []stt.Transcript
	for {
		select {
		case event := <-provider.Events():
			if transcript, ok := event.(stt.Transcript); ok {
				collected = append(collected, transcript)
			}
		default:
			return collected
		}
	}
}

func (s *InworldSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *InworldSuite) TestNewFallsBackToEnv() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultURL, provider.options.URL)
	s.Equal(DefaultModel, provider.options.Model)
}

func (s *InworldSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.inworld.ai/stt/v1/transcribe"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *InworldSuite) TestNewRejectsAVadThresholdOutsideItsRange() {
	over := 1.5

	_, err := New(Options{APIKey: "k", VadThreshold: &over})
	s.ErrorContains(err, "vad threshold must be between 0 and 1")
}

func (s *InworldSuite) TestNewKeepsAVadThresholdOfZeroBecauseItMeansSomething() {
	// Zero turns the server's turn detection off, so it is a request rather than an
	// unset field and cannot be treated as one.
	off := 0.0

	provider, err := New(Options{APIKey: "k", VadThreshold: &off})
	s.Require().NoError(err)
	s.Require().NotNil(provider.options.VadThreshold)
	s.Equal(0.0, *provider.options.VadThreshold)
}

func (s *InworldSuite) TestNewDropsKeytermsTheGatewayWouldRefuse() {
	// The gateway answers invalid-argument for a term with a slash or an at-sign in it,
	// which would cost the session rather than the one word that was spelt oddly.
	provider, err := New(Options{
		APIKey:   "k",
		Keyterms: []string{" eSIM ", "TCP/IP", "support@acme", "  "},
	})
	s.Require().NoError(err)
	s.Equal([]string{"eSIM"}, provider.options.Keyterms)
}

func (s *InworldSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *InworldSuite) TestAModelThatNamesItsOwnVendorIsLeftAlone() {
	// The gateway fronts other vendors too, and one reached deliberately already names
	// itself the way the gateway does.
	s.Equal("inworld/inworld-stt-1", modelID(DefaultModel))
	s.Equal("deepgram/flux-general-en", modelID("deepgram/flux-general-en"))
}

func (s *InworldSuite) TestInterimResultsRestateTheTurnRatherThanAppendToIt() {
	provider := s.newSTT()

	provider.handleMessage(heard("Open the door.", false))
	provider.handleMessage(heard("Open the door and let me in.", false))

	collected := s.transcripts(provider)
	s.Require().Len(collected, 2)
	for _, transcript := range collected {
		s.Equal(stt.ModeReplacement, transcript.Mode)
		s.Equal("alice", transcript.Participant.UserID)
		s.Equal(ProviderName, transcript.Provider)
		s.Equal(DefaultModel, transcript.Model)
	}
	s.Equal("Open the door and let me in.", collected[1].Text)
}

func (s *InworldSuite) TestAFinalResultSettlesTheTurn() {
	provider := s.newSTT()

	provider.handleMessage(heard("Open the door.", false))
	provider.handleMessage(heard("Open the door and let me in, please.", true))

	collected := s.transcripts(provider)
	s.Require().Len(collected, 2)
	s.False(collected[0].Final())
	s.True(collected[1].Final())
	s.Equal("Open the door and let me in, please.", collected[1].Text)
}

func (s *InworldSuite) TestEachRunOfSpeechIsNumberedOnce() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Result: &serverResult{SpeechStarted: &speechStarted{}}})
	provider.handleMessage(heard("hello", false))
	provider.handleMessage(heard("Hello.", true))
	provider.handleMessage(serverMessage{Result: &serverResult{SpeechStarted: &speechStarted{}}})
	provider.handleMessage(heard("goodbye", false))
	provider.handleMessage(heard("Goodbye.", true))

	collected := s.transcripts(provider)
	s.Require().Len(collected, 4)
	s.Equal([]int64{1, 1, 2, 2}, []int64{
		collected[0].Utterance, collected[1].Utterance,
		collected[2].Utterance, collected[3].Utterance,
	})
}

func (s *InworldSuite) TestATurnTheServerSaysNothingInDoesNotUseUpANumber() {
	// Voice activity is not speech the model could transcribe, so a turn announced and
	// then left empty would otherwise make the next thing the caller says the third
	// utterance of a call with one in it.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Result: &serverResult{SpeechStarted: &speechStarted{}}})
	provider.handleMessage(serverMessage{Result: &serverResult{SpeechStopped: &speechStopped{}}})
	provider.handleMessage(serverMessage{Result: &serverResult{SpeechStarted: &speechStarted{}}})
	provider.handleMessage(heard("Hello.", true))

	collected := s.transcripts(provider)
	s.Require().Len(collected, 1)
	s.Equal(int64(1), collected[0].Utterance)
}

func (s *InworldSuite) TestEmptyTranscriptsAreNotReported() {
	provider := s.newSTT()

	provider.handleMessage(heard("   ", false))
	provider.handleMessage(heard("", true))

	s.Empty(s.transcripts(provider))
}

func (s *InworldSuite) TestAServerErrorEndsTheSession() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		Error: &serverError{Code: 3, Message: "invalid transcribe config"},
	})

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure, "invalid transcribe config")
	s.True(failure.Fatal)
}

func (s *InworldSuite) TestTheUsageFrameEndsTheWaitForTheTail() {
	// It arrives before the socket closes, so a teardown that waited for the hangup
	// instead would spend the whole flush timeout on every call.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Result: &serverResult{
		Usage: &usage{TranscribedAudioMs: 2400, ModelID: DefaultModel},
	}})

	select {
	case <-provider.finished:
	default:
		s.Fail("the session should be finished once the server has accounted for it")
	}
}

func (s *InworldSuite) TestAudioBeforeStartIsRefused() {
	provider := s.newSTT()

	err := provider.ProcessAudio(
		stt.PcmData{Samples: []int16{1, 2}, SampleRate: stt.SampleRate, Channels: 1},
		stt.Participant{ID: "alice"},
	)

	s.ErrorContains(err, "not started")
}
