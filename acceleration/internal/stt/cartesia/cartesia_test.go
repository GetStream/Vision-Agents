package cartesia

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type CartesiaSuite struct {
	suite.Suite
}

func TestCartesiaSuite(t *testing.T) {
	suite.Run(t, new(CartesiaSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *CartesiaSuite) newSTT() *STT {
	provider, err := New(Options{APIKey: "test-key"})
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// transcripts collects the transcripts emitted so far without blocking on an empty
// channel.
func (s *CartesiaSuite) transcripts(provider *STT) []stt.Transcript {
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

func (s *CartesiaSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *CartesiaSuite) TestNewFallsBackToEnv() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultURL, provider.options.URL)
	s.Equal(DefaultModel, provider.options.Model)
}

func (s *CartesiaSuite) TestNewRejectsNonWebSocketURL() {
	_, err := New(Options{APIKey: "k", URL: "https://api.cartesia.ai/stt/turns/websocket"})
	s.ErrorContains(err, "url must be ws:// or wss://")
}

func (s *CartesiaSuite) TestNewDropsBlankKeyterms() {
	provider, err := New(Options{APIKey: "k", Keyterms: []string{" eSIM ", "  ", ""}})
	s.Require().NoError(err)
	s.Equal([]string{"eSIM"}, provider.options.Keyterms)
}

func (s *CartesiaSuite) TestNewRefusesMoreKeytermsThanTheEndpointTakes() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = "term"
	}

	_, err := New(Options{APIKey: "k", Keyterms: terms})
	s.ErrorContains(err, "at most 100 keyterms")
}

func (s *CartesiaSuite) TestNewRefusesKeytermsPastTheCharacterBudget() {
	// The endpoint caps the total as well as the count, so a handful of long phrases is
	// refused here rather than by a failed dial.
	_, err := New(Options{
		APIKey:   "k",
		Keyterms: []string{strings.Repeat("a", maxKeytermChars+1)},
	})
	s.ErrorContains(err, "keyterms total at most 1200 characters")
}

func (s *CartesiaSuite) TestProviderAndModelAreReported() {
	provider := s.newSTT()
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultModel, provider.Model())
}

func (s *CartesiaSuite) TestTurnUpdatesRestateTheTurnRatherThanAppendToIt() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnStart})
	provider.handleMessage(serverMessage{Type: eventTurnUpdate, Transcript: "Hey can you help"})
	provider.handleMessage(serverMessage{
		Type: eventTurnUpdate, Transcript: "Hey can you help me with something",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	for _, transcript := range heard {
		s.Equal(stt.ModeReplacement, transcript.Mode)
		s.Equal("alice", transcript.Participant.UserID)
		s.Equal(ProviderName, transcript.Provider)
		s.Equal(DefaultModel, transcript.Model)
	}
	s.Equal("Hey can you help me with something", heard[1].Text)
}

func (s *CartesiaSuite) TestTurnEndSettlesTheTurn() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnStart})
	provider.handleMessage(serverMessage{Type: eventTurnUpdate, Transcript: "Hey can you help"})
	provider.handleMessage(serverMessage{
		Type: eventTurnEnd, Transcript: "Hey can you help me with something?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.False(heard[0].Final())
	s.True(heard[1].Final())
	s.Equal("Hey can you help me with something?", heard[1].Text)
}

func (s *CartesiaSuite) TestAnEagerEndDoesNotSettleTheTurnItGuessedAt() {
	// The model fires this when the caller may be done and takes it back with a resume
	// when they were not. Settling on it is how the first half of a question reaches an
	// agent as the whole of it.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnStart})
	provider.handleMessage(serverMessage{
		Type: eventTurnEagerEnd, Transcript: "Hey can you help me",
	})
	provider.handleMessage(serverMessage{Type: eventTurnResume})
	provider.handleMessage(serverMessage{
		Type: eventTurnUpdate, Transcript: "Hey can you help me with something",
	})
	provider.handleMessage(serverMessage{
		Type: eventTurnEnd, Transcript: "Hey can you help me with something?",
	})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.False(heard[0].Final(), "the caller had not finished, and a resume said so")
	s.False(heard[1].Final())
	s.True(heard[2].Final())
	s.Equal("Hey can you help me with something?", heard[2].Text)
	// One turn throughout: the eager end and the resume are a guess and its retraction,
	// not a boundary.
	s.Equal([]int64{1, 1, 1}, []int64{
		heard[0].Utterance, heard[1].Utterance, heard[2].Utterance,
	})
}

func (s *CartesiaSuite) TestEachRunOfSpeechIsNumbered() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnStart})
	provider.handleMessage(serverMessage{Type: eventTurnEnd, Transcript: "Hello."})
	provider.handleMessage(serverMessage{Type: eventTurnStart})
	provider.handleMessage(serverMessage{Type: eventTurnEnd, Transcript: "Goodbye."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.Equal(int64(1), heard[0].Utterance)
	s.Equal(int64(2), heard[1].Utterance)
}

func (s *CartesiaSuite) TestATurnWhoseStartWasNeverAnnouncedIsStillNumbered() {
	// The audio buffered when the session is closed is transcribed into a turn the server
	// never announced the start of, and an unnumbered transcript reads as a provider that
	// cannot see utterance boundaries at all.
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnEnd, Transcript: "Goodbye."})

	heard := s.transcripts(provider)
	s.Require().Len(heard, 1)
	s.Equal(int64(1), heard[0].Utterance)
}

func (s *CartesiaSuite) TestEmptyTranscriptsAreNotReported() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{Type: eventTurnUpdate, Transcript: "   "})
	provider.handleMessage(serverMessage{Type: eventTurnEnd})

	s.Empty(s.transcripts(provider))
}

func (s *CartesiaSuite) TestAServerErrorEndsTheSession() {
	provider := s.newSTT()

	provider.handleMessage(serverMessage{
		Type:       eventError,
		Title:      "Invalid model",
		Message:    "The model is not valid, make sure it is a valid model ID.",
		ErrorCode:  "model_not_found",
		StatusCode: 400,
	})

	event := <-provider.Events()
	failure, ok := event.(stt.Error)
	s.Require().True(ok)
	s.ErrorContains(failure, "Invalid model")
	s.ErrorContains(failure, "model_not_found")
	s.True(failure.Fatal)
}

func (s *CartesiaSuite) TestAudioBeforeStartIsRefused() {
	provider := s.newSTT()

	err := provider.ProcessAudio(
		stt.PcmData{Samples: []int16{1, 2}, SampleRate: stt.SampleRate, Channels: 1},
		stt.Participant{ID: "alice"},
	)

	s.ErrorContains(err, "not started")
}
