package fish

import (
	"bytes"
	"embed"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// frames are server messages as Fish send them, most captured from a live session. They
// are kept as JSON to be readable, and sent as the MessagePack the server writes.
//
//go:embed testdata/*.json
var frames embed.FS

// frame returns one of the captured server messages as it arrives on the wire.
func frame(name string) []byte {
	raw, err := frames.ReadFile("testdata/" + name)
	if err != nil {
		panic(err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var message any
	if err := decoder.Decode(&message); err != nil {
		panic(err)
	}
	encoded, err := msgpack.Marshal(integers(message))
	if err != nil {
		panic(err)
	}
	return encoded
}

// integers turns the whole numbers JSON decoded into integers, which is how the server
// writes them.
func integers(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			typed[key] = integers(item)
		}
	case []any:
		for i, item := range typed {
			typed[i] = integers(item)
		}
	case json.Number:
		if whole, err := typed.Int64(); err == nil {
			return whole
		}
		fractional, err := typed.Float64()
		if err != nil {
			panic(err)
		}
		return fractional
	}
	return value
}

type FishSuite struct {
	suite.Suite
}

func TestFishSuite(t *testing.T) {
	suite.Run(t, new(FishSuite))
}

// newSTT returns a provider that is wired up but never connected, so the event mapping
// can be exercised without touching the network.
func (s *FishSuite) newSTT(options Options) *STT {
	options.APIKey = "test-key"
	provider, err := New(options)
	s.Require().NoError(err)
	provider.participant = stt.Participant{ID: "alice", UserID: "alice"}
	return provider
}

// receive hands the provider a frame the way its read loop would.
func (s *FishSuite) receive(provider *STT, names ...string) {
	for _, name := range names {
		var message serverMessage
		s.Require().NoError(msgpack.Unmarshal(frame(name), &message))
		provider.handleMessage(message)
	}
}

// events collects what was emitted so far without blocking on an empty channel.
func (s *FishSuite) events(provider *STT) []stt.Event {
	var emitted []stt.Event
	for {
		select {
		case event := <-provider.Events():
			emitted = append(emitted, event)
		default:
			return emitted
		}
	}
}

func (s *FishSuite) transcripts(provider *STT) []stt.Transcript {
	var heard []stt.Transcript
	for _, event := range s.events(provider) {
		if transcript, ok := event.(stt.Transcript); ok {
			heard = append(heard, transcript)
		}
	}
	return heard
}

func (s *FishSuite) errors(provider *STT) []stt.Error {
	var failures []stt.Error
	for _, event := range s.events(provider) {
		if failure, ok := event.(stt.Error); ok {
			failures = append(failures, failure)
		}
	}
	return failures
}

func (s *FishSuite) TestNewRequiresAPIKey() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required (set FISH_API_KEY)")
}

func (s *FishSuite) TestNewFallsBackToEnvAndTheRealtimeModel() {
	s.T().Setenv(apiKeyEnvVar, "key-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(DefaultModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.Equal(ModeAuto, provider.options.Mode)
}

func (s *FishSuite) TestNewRefusesAModeThatWouldNeverEndATurn() {
	_, err := New(Options{APIKey: "test-key", Mode: "manual"})
	s.ErrorContains(err, "mode must be auto or vad, got manual")
}

func (s *FishSuite) TestNewRefusesASilenceTheModeDoesNotTake() {
	// The server closes a session configured with one, so it is refused before dialing.
	_, err := New(Options{APIKey: "test-key", SilenceTimeoutMs: 700})
	s.ErrorContains(err, "silence timeout must be between 3000 and 60000 ms in auto mode, got 700")

	_, err = New(Options{APIKey: "test-key", Mode: ModeVAD, SilenceTimeoutMs: 700})
	s.NoError(err, "vad mode ends every turn on a silence, so it takes a short one")
}

func (s *FishSuite) TestNewRefusesKeytermsTheServerWouldNot() {
	_, err := New(Options{APIKey: "test-key", Keyterms: []string{strings.Repeat("a", 33)}})
	s.ErrorContains(err, "keyterms are at most 32 characters")

	_, err = New(Options{APIKey: "test-key", Keyterms: strings.Fields(strings.Repeat("Mia ", maxKeyterms+1))})
	s.ErrorContains(err, "at most 100 keyterms")
}

func (s *FishSuite) TestTheCallersTurnIsRestatedUntilItSettles() {
	provider := s.newSTT(Options{})

	s.receive(provider, "speech_start.json", "user_turn_taken.json", "draft.json",
		"draft_longer.json", "committed.json")

	heard := s.transcripts(provider)
	s.Require().Len(heard, 3)
	s.Equal(stt.ModeReplacement, heard[0].Mode)
	s.Equal("In a quiet village where the", heard[0].Text)
	s.Equal(stt.ModeReplacement, heard[1].Mode)
	s.Equal(stt.ModeFinal, heard[2].Mode)
	for _, transcript := range heard {
		s.Equal(int64(1), transcript.Utterance, "turn_index counts from zero and Utterance from one")
		s.Equal("alice", transcript.Participant.UserID)
	}

	final := heard[2]
	s.Equal("In a quiet village where the sky brushes the fields in hues of gold, young Mia "+
		"discovered a map leading to forgotten treasures.", final.Text,
		"a sound tag is something heard rather than said")
	s.Equal("en", final.Language)
	s.InDelta(8960, final.AudioDurationMs, 0.001)
	s.Equal(ProviderName, final.Provider)
	s.Equal(DefaultModel, final.Model)
}

func (s *FishSuite) TestDraftsBeforeTheTurnIsTheCallersAreNotReported() {
	// The next draft restates the whole turn, so nothing said is lost by waiting.
	provider := s.newSTT(Options{})

	s.receive(provider, "speech_start.json", "draft_before_taken.json")
	s.Empty(s.transcripts(provider))

	s.receive(provider, "user_turn_taken.json", "draft.json")
	s.Len(s.transcripts(provider), 1)
}

func (s *FishSuite) TestASoundThatIsNotTheCallerIsNeverReported() {
	provider := s.newSTT(Options{})

	s.receive(provider, "speech_start_noise.json", "draft_noise.json", "committed_noise.json")

	s.Empty(s.transcripts(provider), "a television is not something to answer")
}

func (s *FishSuite) TestInVADModeEveryTurnIsTheCallers() {
	provider := s.newSTT(Options{Mode: ModeVAD})

	s.receive(provider, "speech_start.json", "draft.json")

	s.Len(s.transcripts(provider), 1, "vad mode sends no user_turn_taken, since every turn is the caller's")
}

func (s *FishSuite) TestATurnOfOnlySoundsSettlesNothing() {
	provider := s.newSTT(Options{})

	s.receive(provider, "committed_laughter.json")

	s.Empty(s.transcripts(provider), "a turn that is only [laughter] has nothing in it to answer")
}

func (s *FishSuite) TestAnEagerEndOfTurnRestatesTheTurnAndAResumedOneCarriesOn() {
	provider := s.newSTT(Options{Eager: true})

	s.receive(provider, "speech_start.json", "user_turn_taken.json", "eager_end_of_turn.json",
		"turn_resumed.json", "committed.json")

	heard := s.transcripts(provider)
	s.Require().Len(heard, 2)
	s.Equal(stt.ModeReplacement, heard[0].Mode, "the model only guesses the turn is over")
	s.Equal(stt.ModeFinal, heard[1].Mode)
	s.Equal(heard[0].Utterance, heard[1].Utterance, "a resumed turn keeps its number")
}

func (s *FishSuite) TestARefusedConfigureIsReportedWithoutEndingTheSession() {
	provider := s.newSTT(Options{})

	s.receive(provider, "configured_refused.json")

	failures := s.errors(provider)
	s.Require().Len(failures, 1)
	s.False(failures[0].Fatal, "nothing in a refused configure is applied, and the session goes on")
	s.ErrorContains(failures[0], "silence_timeout_ms must be between 3000 and 60000 in auto mode")
}

func (s *FishSuite) TestAnErrorIsFatalOnlyWhenTheServerSaysItIs() {
	provider := s.newSTT(Options{})

	s.receive(provider, "error_engine.json", "error_invalid_config.json")

	failures := s.errors(provider)
	s.Require().Len(failures, 2)
	s.False(failures[0].Fatal, "a degraded commit follows an engine error and the session continues")
	s.ErrorContains(failures[0], "engine_error: final pass timed out")
	s.True(failures[1].Fatal)
	s.ErrorContains(failures[1], "invalid_config")
}

func (s *FishSuite) TestSessionDoneReleasesAWaitingClose() {
	provider := s.newSTT(Options{})

	s.receive(provider, "session_done.json")

	select {
	case <-provider.terminated:
	default:
		s.Fail("Close should stop waiting once the server has said its last")
	}
}

func (s *FishSuite) TestTheRouterMetersTheAudio() {
	s.Equal(stt.MeteredByRouter, stt.MeteringOf(s.newSTT(Options{})))
}
