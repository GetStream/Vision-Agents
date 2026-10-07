package microsoft

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type MicrosoftSuite struct {
	suite.Suite
}

func TestMicrosoftSuite(t *testing.T) {
	suite.Run(t, new(MicrosoftSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *MicrosoftSuite) newSTT(options Options) *STT {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	if options.Endpoint == "" {
		options.Endpoint = "https://example.services.ai.azure.com"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel. A closed
// channel reads forever, so a session that has been hung up ends the drain.
func (s *MicrosoftSuite) drain(provider *STT) []stt.Event {
	var events []stt.Event
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
		default:
			return events
		}
	}
}

func (s *MicrosoftSuite) transcripts(provider *STT) []stt.Transcript {
	var found []stt.Transcript
	for _, event := range s.drain(provider) {
		transcript, ok := event.(stt.Transcript)
		s.Require().True(ok, "expected only transcripts, got %T", event)
		found = append(found, transcript)
	}
	return found
}

func (s *MicrosoftSuite) texts(provider *STT) []string {
	var said []string
	for _, transcript := range s.transcripts(provider) {
		said = append(said, transcript.Text)
	}
	return said
}

func delta(text string) serverMessage {
	return serverMessage{Type: eventDelta, Delta: text}
}

func intermediate(text string) serverMessage {
	return serverMessage{Type: eventIntermediate, Intermediate: text}
}

func completed(text string) serverMessage {
	return serverMessage{Type: eventCompleted, Transcript: text}
}

func (s *MicrosoftSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{Endpoint: "https://example.services.ai.azure.com"})
	s.ErrorContains(err, "api key is required")
}

func (s *MicrosoftSuite) TestNewRequiresAnEndpoint() {
	s.T().Setenv(endpointEnvVar, "")

	_, err := New(Options{APIKey: "k"})
	s.ErrorContains(err, "endpoint is required")
}

func (s *MicrosoftSuite) TestNewFallsBackToTheEnvironment() {
	s.T().Setenv(apiKeyEnvVar, "from-env")
	s.T().Setenv(endpointEnvVar, "https://mine.services.ai.azure.com")
	s.T().Setenv(deploymentEnvVar, "my-deployment")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
	s.Equal("https://mine.services.ai.azure.com", provider.options.Endpoint)
	s.Equal("my-deployment", provider.options.Deployment)
}

func (s *MicrosoftSuite) TestTheDeploymentDefaultsToTheModelName() {
	// Foundry names a deployment after its model unless told otherwise.
	s.T().Setenv(deploymentEnvVar, "")

	s.Equal(DefaultModel, s.newSTT(Options{}).options.Deployment)
}

func (s *MicrosoftSuite) TestNewRefusesAnEndpointThatIsNotAResourceRoot() {
	_, err := New(Options{APIKey: "k", Endpoint: "https://example.services.ai.azure.com/mai/v1/realtime"})
	s.ErrorContains(err, "resource root")
}

func (s *MicrosoftSuite) TestNewRefusesAnEndpointThatIsNotAURLForTheSocket() {
	_, err := New(Options{APIKey: "k", Endpoint: "example.services.ai.azure.com"})
	s.ErrorContains(err, "https://")
}

func (s *MicrosoftSuite) TestTheSocketIsTheRealtimePathAskingToTranscribe() {
	endpoint, err := socketURL("https://example.services.ai.azure.com/")
	s.Require().NoError(err)
	s.Equal("wss://example.services.ai.azure.com/mai/v1/realtime?intent=transcription", endpoint)
}

func (s *MicrosoftSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT(Options{})
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *MicrosoftSuite) TestTheSessionUpdateLeavesTurnsToTheClient() {
	payload, err := json.Marshal(s.newSTT(Options{Deployment: "mai-stream"}).sessionUpdate())
	s.Require().NoError(err)

	s.JSONEq(`{
		"type": "session.update",
		"session": {
			"type": "transcription",
			"audio": {"input": {
				"format": {"type": "audio/pcm", "rate": 16000},
				"transcription": {"model": "mai-stream", "language": null},
				"turn_detection": null,
				"noise_reduction": null
			}}
		}
	}`, string(payload))
}

func (s *MicrosoftSuite) TestTheSessionUpdateNamesALanguageWhenGivenOne() {
	update := s.newSTT(Options{Language: "fr"}).sessionUpdate()

	s.Require().NotNil(update.Session.Audio.Input.Transcription.Language)
	s.Equal("fr", *update.Session.Audio.Input.Transcription.Language)
}

func (s *MicrosoftSuite) TestADeltaProducesAReplacementTranscript() {
	provider := s.newSTT(Options{})
	speaker := stt.Participant{ID: "p1", UserID: "u1"}
	provider.participant = speaker

	provider.handleMessage(delta("  in a quiet  "))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal("in a quiet", heard[0].Text, "surrounding whitespace should be trimmed")
	s.Equal(stt.ModeReplacement, heard[0].Mode)
	s.Equal(speaker, heard[0].Participant)
	s.Equal(ProviderName, heard[0].Provider)
	s.Equal(DefaultModel, heard[0].Model)
}

// TestTheHypothesisIsTheDeltasPlusTheLatestIntermediate is Microsoft's own example off
// their page: deltas append, an intermediate replaces the one before it, and a delta
// overtakes the intermediate it finalized.
func (s *MicrosoftSuite) TestTheHypothesisIsTheDeltasPlusTheLatestIntermediate() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Hello"))
	provider.handleMessage(intermediate(" world"))
	provider.handleMessage(intermediate(" there"))
	provider.handleMessage(delta(" there!"))

	s.Equal([]string{"Hello", "Hello world", "Hello there", "Hello there!"}, s.texts(provider))
}

func (s *MicrosoftSuite) TestDeltasAreJoinedWithTheSpacingTheServerChose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("pat"))
	provider.handleMessage(delta("io"))
	provider.handleMessage(delta(" chairs"))

	heard := s.texts(provider)
	s.Equal("patio chairs", heard[len(heard)-1], "adding or removing a space would split or fuse words")
}

func (s *MicrosoftSuite) TestAFrameThatChangesNothingIsNotRepeated() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Hello"))
	provider.handleMessage(intermediate(""))

	s.Equal([]string{"Hello"}, s.texts(provider))
}

func (s *MicrosoftSuite) TestACompletedTranscriptSettlesTheTurn() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Hello"))
	provider.handleMessage(intermediate(" there"))
	provider.handleMessage(completed("Hello there!"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.True(heard[2].Final())
	s.Equal(stt.ModeFinal, heard[2].Mode)
	s.Equal("Hello there!", heard[2].Text)
}

func (s *MicrosoftSuite) TestHypothesesShareTheUtteranceOfTheFinalTheyBecome() {
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("in a quiet"))
	provider.handleMessage(intermediate(" village"))
	provider.handleMessage(completed("in a quiet village."))
	provider.handleMessage(delta("Young"))

	heard := s.transcripts(provider)
	s.Require().Len(heard, 4)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(1), heard[1].Utterance)
	s.Equal(int64(1), heard[2].Utterance, "the end of a run belongs to the run it ends")
	s.Equal(int64(2), heard[3].Utterance)
}

func (s *MicrosoftSuite) TestATurnDoesNotBeginWithTheOneBefore() {
	// A completed is the whole of the audio since the previous commit, so nothing heard
	// before it belongs to the turn after.
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("In a quiet village."))
	provider.handleMessage(completed("In a quiet village."))
	provider.handleMessage(delta("Hi"))

	heard := s.transcripts(provider)
	s.Equal("Hi", heard[len(heard)-1].Text)
}

func (s *MicrosoftSuite) TestAnEmptyCommitPublishesNothing() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed("  "))

	s.Empty(s.drain(provider))
}

func (s *MicrosoftSuite) TestAnAnsweredCommitReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(completed(""))

	select {
	case <-provider.settled:
	default:
		s.Fail("an answered commit, even an empty one, should release a Close waiting for the tail")
	}
}

func (s *MicrosoftSuite) TestHangingUpSettlesWhatTheCallerHadJustSaid() {
	// The grace period has not run out when somebody hangs up mid-sentence, and the words
	// they got out are still owed to whoever was listening.
	provider := s.newSTT(Options{})

	provider.handleMessage(delta("Can you"))
	provider.handleMessage(intermediate(" hear"))
	s.Require().NoError(provider.Close())

	heard := s.transcripts(provider)
	s.Require().NotEmpty(heard)
	last := heard[len(heard)-1]
	s.True(last.Final())
	s.Equal("Can you hear", last.Text)
}

func (s *MicrosoftSuite) TestAFailedTranscriptionIsNotFatal() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed, Error: &serverError{Message: "decode failed"}})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.False(failure.Fatal)
	s.ErrorContains(failure, "decode failed")
}

func (s *MicrosoftSuite) TestAFailedTranscriptionReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventFailed})

	select {
	case <-provider.settled:
	default:
		s.Fail("a hangup should not wait out the timeout for words the server has given up on")
	}
}

func (s *MicrosoftSuite) TestServerErrorIsFatal() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventError, Error: &serverError{Message: "deployment not found"}})

	events := s.drain(provider)
	s.Require().Len(events, 1)
	failure, ok := events[0].(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "deployment not found")
}

func (s *MicrosoftSuite) TestSessionAndCommitAcknowledgementsAreNotSTTEvents() {
	provider := s.newSTT(Options{})

	provider.handleMessage(serverMessage{Type: eventSessionCreated})
	provider.handleMessage(serverMessage{Type: eventSessionUpdated})
	provider.handleMessage(serverMessage{Type: eventCommitted})

	s.Empty(s.drain(provider))
}

func (s *MicrosoftSuite) TestProcessAudioRejectsWrongAudioFormat() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: 48000, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "sample rate must be 16000")
}

func (s *MicrosoftSuite) TestProcessAudioFailsBeforeStart() {
	provider := s.newSTT(Options{})

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "not started")
}

func (s *MicrosoftSuite) TestProcessAudioFailsAfterClose() {
	provider := s.newSTT(Options{})
	s.Require().NoError(provider.Close())

	err := provider.ProcessAudio(stt.PcmData{SampleRate: stt.SampleRate, Channels: 1}, stt.Participant{})
	s.ErrorContains(err, "session closed")
}

func (s *MicrosoftSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newSTT(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	_, open := <-provider.Events()
	s.False(open, "closing the session should close the event channel")
}

func (s *MicrosoftSuite) TestSatisfiesSTTInterface() {
	var _ stt.STT = s.newSTT(Options{})
}
