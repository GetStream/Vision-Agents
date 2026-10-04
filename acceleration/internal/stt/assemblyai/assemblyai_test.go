package assemblyai

import (
	"embed"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// frames are server messages as AssemblyAI send them, most captured from a live session.
//
//go:embed testdata/*.json
var frames embed.FS

// frame returns one of the captured server messages as it arrives on the wire.
func frame(name string) []byte {
	raw, err := frames.ReadFile("testdata/" + name)
	if err != nil {
		panic(err)
	}
	return raw
}

type AssemblyAISuite struct {
	suite.Suite
}

func TestAssemblyAISuite(t *testing.T) {
	suite.Run(t, new(AssemblyAISuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *AssemblyAISuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key"})
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// receive hands the provider a frame the way its read loop would.
func (s *AssemblyAISuite) receive(provider *STT, names ...string) {
	for _, name := range names {
		var message serverMessage
		s.Require().NoError(json.Unmarshal(frame(name), &message))
		provider.handleMessage(message)
	}
}

// transcripts collects the transcripts emitted so far without blocking on an empty
// channel.
func (s *AssemblyAISuite) transcripts(provider *STT) []stt.Transcript {
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

func (s *AssemblyAISuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *AssemblyAISuite) TestNewFallsBackToEnv() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultURL, provider.options.URL)
	s.Equal(DefaultModel, provider.options.Model)
	s.Equal(DefaultMode, provider.options.Mode)
}

func (s *AssemblyAISuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://streaming.assemblyai.com/v3/ws"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *AssemblyAISuite) TestNewRefusesAPresetTheServerDoesNotHave() {
	_, err := New(Options{APIKey: "k", Mode: "fastest"})
	s.ErrorContains(err, "mode must be")
}

func (s *AssemblyAISuite) TestNewDropsBlankKeyterms() {
	provider, err := New(Options{APIKey: "k", Keyterms: []string{" Mia ", "  ", ""}})
	s.Require().NoError(err)
	s.Equal([]string{"Mia"}, provider.options.Keyterms)
}

func (s *AssemblyAISuite) TestNewRefusesMoreKeytermsThanASessionTakes() {
	terms := make([]string, maxKeyterms+1)
	for i := range terms {
		terms[i] = "term"
	}

	_, err := New(Options{APIKey: "k", Keyterms: terms})
	s.ErrorContains(err, "at most 100 keyterms")
}

func (s *AssemblyAISuite) TestNewRefusesAKeytermLongerThanTheModelTakes() {
	// The server accepts a longer one without complaint, so the limit it documents is
	// only ever enforced here.
	_, err := New(Options{APIKey: "k", Keyterms: []string{strings.Repeat("a", maxKeytermRunes+1)}})
	s.ErrorContains(err, "keyterms are at most 50 characters")
}

func (s *AssemblyAISuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *AssemblyAISuite) TestEachTurnFrameRestatesTheTurnRatherThanAppendingToIt() {
	provider := s.newSTT()

	s.receive(provider, "speech_started.json", "turn_partial.json", "turn_partial_longer.json")

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2, "SpeechStarted only says a turn is coming")
	for _, transcript := range heard {
		s.Equal(stt.ModeReplacement, transcript.Mode)
		s.Equal("alice", transcript.Participant.UserID)
		s.Equal(ProviderName, transcript.Provider)
		s.Equal(DefaultModel, transcript.Model)
	}
	s.Equal("In a quiet village where the sky brushes", heard[1].Text)
}

func (s *AssemblyAISuite) TestTheFrameThatEndsTheTurnSettlesIt() {
	provider := s.newSTT()

	s.receive(provider, "turn_partial.json", "turn_final.json")

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.False(heard[0].Final())
	s.True(heard[1].Final())
	s.Equal("In a quiet village where the sky brushes the fields in hues of gold,", heard[1].Text)
}

func (s *AssemblyAISuite) TestTurnsAreNumberedAsTheServerNumbersThem() {
	provider := s.newSTT()

	s.receive(provider, "turn_partial.json", "turn_final.json", "turn_next_partial.json")

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.Equal([]int64{1, 1, 2}, []int64{heard[0].Utterance, heard[1].Utterance, heard[2].Utterance},
		"turn_order counts from zero and utterances from one")
}

func (s *AssemblyAISuite) TestAnEmptyTurnIsNotReported() {
	provider := s.newSTT()

	s.receive(provider, "turn_empty.json")

	s.Empty(s.transcripts(provider))
}

func (s *AssemblyAISuite) TestAnErrorFrameEndsTheSessionWithTheServersReason() {
	provider := s.newSTT()

	s.receive(provider, "error_input_duration.json")

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure, "Expected between 50 and 1000 ms")
	s.ErrorContains(failure, "code 3007")
	s.True(failure.Fatal, "the server closes the socket straight after an Error frame")
}

func (s *AssemblyAISuite) TestTerminationTellsAWaitingCloseNothingMoreIsComing() {
	provider := s.newSTT()

	s.receive(provider, "termination.json")

	select {
	case <-provider.terminated:
	default:
		s.Fail("Termination is the last frame the server sends, so Close can stop waiting")
	}
}

func (s *AssemblyAISuite) TestAudioBeforeStartIsRefused() {
	provider := s.newSTT()

	err := provider.ProcessAudio(
		stt.PcmData{Samples: []int16{1, 2}, SampleRate: stt.SampleRate, Channels: 1},
		stt.Participant{ID: "alice"},
	)

	s.ErrorContains(err, "not started")
}
