package togetherparakeet

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type TogetherParakeetSuite struct {
	suite.Suite
}

func TestTogetherParakeetSuite(t *testing.T) {
	suite.Run(t, new(TogetherParakeetSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *TogetherParakeetSuite) newSTT(options Options) *STT {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *TogetherParakeetSuite) drain(provider *STT) []stt.Event {
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

// transcripts is the drained events as transcripts, which is what most of these are about.
func (s *TogetherParakeetSuite) transcripts(provider *STT) []stt.Transcript {
	var found []stt.Transcript
	for _, event := range s.drain(provider) {
		transcript, ok := event.(stt.Transcript)
		s.Require().True(ok, "expected only transcripts, got %T", event)
		found = append(found, transcript)
	}
	return found
}

// delta is more of what the caller seems to be saying.
func delta(text string) serverMessage {
	return serverMessage{Type: eventDelta, Delta: text}
}

// completed is the settled utterance.
func completed(text string) serverMessage {
	return serverMessage{Type: eventCompleted, Transcript: text}
}

func (s *TogetherParakeetSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *TogetherParakeetSuite) TestNewFallsBackToEnvAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *TogetherParakeetSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.together.ai/v1/realtime"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *TogetherParakeetSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *TogetherParakeetSuite) TestTheProviderIsNamedApartFromTheSelfHostedParakeet() {
	// The same weights on our own deployment are a different provider: the bill and the
	// pager are not shared, and routing has to be able to pick between them.
	s.NotEqual("parakeet", ProviderName)
}

func (s *TogetherParakeetSuite) TestADeltaProducesAReplacementTranscript() {
	provider := s.newSTT(Options{})
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleMessage(serverMessage{Type: eventDelta, Delta: "  in a quiet vill  "})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("in a quiet vill", heard[0].Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, heard[0].Mode, "each delta restates the utterance")
	s.False(heard[0].Final())
	s.Equal(speaker, heard[0].Participant)
	s.Equal(ProviderName, heard[0].Provider)
	s.Equal(DefaultModel, heard[0].Model)
}

func (s *TogetherParakeetSuite) TestACompletedTranscriptSettlesTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("In a quiet village.", heard[0].Text)
	s.True(heard[0].Final())
}

func (s *TogetherParakeetSuite) TestDeltasShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("in a quiet"))
	provider.handleMessage(delta("in a quiet village"))
	provider.handleMessage(completed("In a quiet village."))
	provider.handleMessage(delta("forgotten"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(1), heard[1].Utterance)
	s.Equal(int64(1), heard[2].Utterance, "the end of a run belongs to the run it ends")
	s.Equal(int64(2), heard[3].Utterance)
}

func (s *TogetherParakeetSuite) TestEmptyTranscriptsAreNotEmitted() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("   "))
	provider.handleMessage(completed("  "))

	s.Empty(s.drain(provider), "whitespace-only transcripts carry no information")
}

func (s *TogetherParakeetSuite) TestASettledUtteranceReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("In a quiet village."))

	select {
	case <-provider.settled:
	default:
		s.Fail("a settled utterance should release a Close waiting for the tail")
	}
}

func (s *TogetherParakeetSuite) TestAnEmptyCommitStillReleasesAWaitingClose() {
	// There was nothing left to transcribe. Close has its answer all the same, and waiting
	// out the timeout would spend it on every hangup.
	provider := s.newSTT(Options{})

	provider.handleMessage(completed(""))

	select {
	case <-provider.settled:
	default:
		s.Fail("an empty transcript is still an answer")
	}
}

func (s *TogetherParakeetSuite) TestAFailedUtteranceIsNotFatal() {
	// Together's protocol says the session carries on, so tearing it down would end a call
	// over one utterance that could not be transcribed.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed, Message: "decode failed"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.False(failure.Fatal)
	s.ErrorContains(failure, "decode failed")
}

func (s *TogetherParakeetSuite) TestAFailedUtteranceReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed, Message: "decode failed"})

	select {
	case <-provider.settled:
	default:
		s.Fail("a hangup should not wait out the timeout for words the server has given up on")
	}
}

func (s *TogetherParakeetSuite) TestServerErrorIsFatal() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventError, Message: "model not available"})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "model not available")
}

func (s *TogetherParakeetSuite) TestAnErrorNestedTheRealtimeWayIsStillReported() {
	// Together documents the message at the top level; the realtime protocol they mirror
	// nests it. A failure nobody can read is worse than either.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{
		Type:  eventError,
		Error: &serverError{Message: "invalid api key"},
	})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	s.ErrorContains(events[0].(stt.Error), "invalid api key")
}

func (s *TogetherParakeetSuite) TestTheSessionCreatedFrameIsNotAnSTTEvent() {
	// The handshake already reported the session as connected.
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventSessionCreated})

	s.Empty(s.drain(provider))
}

func (s *TogetherParakeetSuite) TestTheEndpointNamesTheModelAndAsksToTranscribe() {
	endpoint := s.newSTT(Options{}).endpoint()

	s.Contains(endpoint, "intent=transcription")
	s.Contains(endpoint, "model=nvidia%2Fparakeet-tdt-0.6b-v3-realtime")
	s.Contains(endpoint, "input_audio_format=pcm_s16le_16000")
}

func (s *TogetherParakeetSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *TogetherParakeetSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *TogetherParakeetSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT(Options{})
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *TogetherParakeetSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *TogetherParakeetSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT(Options{})
}
