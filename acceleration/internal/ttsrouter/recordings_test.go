package ttsrouter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// heldRecorder is the recorder ElevenLabs' own factory built, kept from speaking: what it
// is asked to record is held rather than sent, so nothing is dialled.
type heldRecorder struct {
	tts.Recorder
	asked *[]tts.Recording
}

func (h heldRecorder) Record(_ context.Context, recording tts.Recording) (tts.Recorded, error) {
	*h.asked = append(*h.asked, recording)
	return tts.Recorded{Audio: []byte("audio")}, nil
}

// Voice is the voice the recorder ElevenLabs built speaks in.
func (h heldRecorder) Voice() string { return h.Recorder.(tts.Voiced).Voice() }

// unvoicedRecorder records and cannot say in which voice.
type unvoicedRecorder struct{}

func (unvoicedRecorder) Record(context.Context, tts.Recording) (tts.Recorded, error) {
	return tts.Recorded{Audio: []byte("audio")}, nil
}
func (unvoicedRecorder) Start(context.Context) error { return nil }
func (unvoicedRecorder) Close() error                { return nil }
func (unvoicedRecorder) Provider() string            { return "elevenlabs" }
func (unvoicedRecorder) Model() string               { return "eleven_v3" }

// ownVoices is a customer with one voice of their own, founder, which ElevenLabs knows as
// el-founder. Any other name is not theirs and comes back as asked.
type ownVoices struct{}

func (ownVoices) ResolveVoice(_ context.Context, _, _, voice string) (string, error) {
	if voice == "founder" {
		return "el-founder", nil
	}
	return voice, nil
}

type RecordingsSuite struct {
	suite.Suite
	ctx context.Context
	// built is every recorder ElevenLabs' factory built, and asked every recording handed
	// to one.
	built []tts.Recorder
	asked []tts.Recording
}

func TestRecordingsSuite(t *testing.T) {
	suite.Run(t, new(RecordingsSuite))
}

func (s *RecordingsSuite) SetupTest() {
	s.ctx = context.Background()
	s.built, s.asked = nil, nil
}

// newRecordings routes over the built-in config with ElevenLabs' own recorder factory,
// whose recorders are held rather than let speak.
func (s *RecordingsSuite) newRecordings() *Recordings {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	vendor := DefaultRecorderRegistry()
	registry := NewRecorderRegistry()
	registry.Register("elevenlabs", func(spec routing.Spec) (tts.Recorder, error) {
		recorder, err := vendor.Build("elevenlabs", spec)
		if err != nil {
			return nil, err
		}
		s.built = append(s.built, recorder)
		return heldRecorder{Recorder: recorder, asked: &s.asked}, nil
	})

	recordings, err := NewRecordings(Options{Config: config[routing.TTS], Recorders: registry, Voices: ownVoices{}})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)
	return recordings
}

func (s *RecordingsSuite) record(voice string, overwrite string) error {
	asked := options.TTS{Target: "en-recorded", Voice: voice}
	if overwrite != "" {
		asked.Overwrites = map[string]json.RawMessage{"elevenlabs": json.RawMessage(overwrite)}
	}
	_, _, err := s.newRecordings().Record(s.ctx, Recording{CustomerID: "acme", Options: asked, Text: "hello"})
	return err
}

// spokenIn is the voice the one recording was spoken in: the one it named, or failing
// that the one its recorder was built for.
func (s *RecordingsSuite) spokenIn() string {
	s.Require().Len(s.asked, 1)
	if s.asked[0].Voice != "" {
		return s.asked[0].Voice
	}
	s.Require().Len(s.built, 1)
	voiced, ok := s.built[0].(tts.Voiced)
	s.Require().True(ok, "the recorder says which voice it speaks in")
	return voiced.Voice()
}

func (s *RecordingsSuite) TestARecordingWithNoOverwriteIsSpokenInTheVoiceItAskedFor() {
	// On the commit before overwrites were read here (a5721c90) this recording was handed
	// library-voice and spoken in it.
	s.Require().NoError(s.record("library-voice", ""))

	s.Equal("library-voice", s.spokenIn())
}

func (s *RecordingsSuite) TestACustomersOwnVoiceIsSpokenInTheIdElevenLabsKnowsItBy() {
	// On a5721c90 the recording was handed founder, the name as asked, and spoken in that.
	s.Require().NoError(s.record("founder", ""))

	s.Equal("el-founder", s.spokenIn())
}

func (s *RecordingsSuite) TestAVoiceIdOverwriteIsWhatARecordingIsSpokenIn() {
	s.Require().NoError(s.record("library-voice", `{"voice_id":"el-1"}`))

	s.Equal("el-1", s.spokenIn(), "the voice named for this vendor wins, as on the live path")
}

func (s *RecordingsSuite) TestAnOverwriteTheRecorderHasNoFieldForIsRefused() {
	err := s.record("library-voice", `{"voice_settings":{"speed":1.2}}`)

	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorIs(err, routing.ErrRequest, "a request error is not counted against the model's health")
	s.ErrorContains(err, `unknown field "speed"`)
	s.Empty(s.asked, "nothing is spoken with the overwrite dropped")
}

func (s *RecordingsSuite) TestAStyleTheRecordingModelHasNoPlaceForIsRefused() {
	err := s.record("library-voice", `{"voice_settings":{"style":0.4}}`)

	// The built-in recording voice is eleven_v3, which has no style.
	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorContains(err, "style is not taken by eleven_v3")
	s.Empty(s.asked, "nothing is spoken with the style dropped")
}

func (s *RecordingsSuite) TestARecordingAskingForASpeedIsRefusedRatherThanSpokenAtItsOwnPace() {
	// The built-in recording voice is eleven_v3, which ignores speed (live, 2026-10-16: a
	// speed of 0.7 to 1.5 all gave a 200), so it does not declare it.
	speed := 0.8
	_, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options:    options.TTS{Target: "en-recorded", Voice: "library-voice", Speed: &speed},
		Text:       "hello",
	})

	s.Require().Error(err)
	s.ErrorContains(err, "speed")
	s.Empty(s.asked, "nothing is spoken with the speed dropped")
}

func (s *RecordingsSuite) TestAStabilityOrSimilarityOutsideZeroToOneIsRefusedBeforeTheCall() {
	// Live on 2026-10-16 eleven_v3 answered 422 to 1.5, which would count against its health.
	above := 1.5
	for name, asked := range map[string]options.TTS{
		"stability":  {Target: "en-recorded", Voice: "library-voice", Stability: &above},
		"similarity": {Target: "en-recorded", Voice: "library-voice", Similarity: &above},
	} {
		s.Run(name, func() {
			s.asked = nil
			_, _, err := s.newRecordings().Record(s.ctx, Recording{CustomerID: "acme", Options: asked, Text: "hello"})

			s.Require().ErrorIs(err, routing.ErrRequest, "the caller's mistake, not counted against the model")
			s.ErrorContains(err, name+" 1.5 is outside 0 to 1")
			s.Empty(s.asked, "nothing is recorded")
		})
	}
}

func (s *RecordingsSuite) TestARecordingWithAVoiceOnlyForAVendorWithNoRecorderIsRefused() {
	// On b1e563f98 ElevenLabs v4 recorded this in its default voice: only ElevenLabs records.
	_, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "tts-quality", Overwrites: map[string]json.RawMessage{
			"cartesia": json.RawMessage(`{"voice_id":"db6b0ed5-cartesia"}`),
		}},
		Text: "hello",
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "a recording nothing here can speak in that voice is the caller's to change")
	s.ErrorContains(err, "cartesia/sonic-preview has no factory registered here")
	s.Empty(s.built, "ElevenLabs was never built without the voice")
}

func (s *RecordingsSuite) TestARecordingSaysItWasSpokenInTheDefaultVoiceWhenNoneWasAsked() {
	// Probed on b1e563f98: the recorder was built for elevenlabs.DefaultVoiceID.
	recorded, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme", Options: options.TTS{Target: "en-recorded"}, Text: "hello",
	})
	s.Require().NoError(err)

	s.Equal(elevenlabs.DefaultVoiceID, recorded.Voice)
}

func (s *RecordingsSuite) TestARecorderThatCannotSayItsVoiceReportsNoneRatherThanTheOneAsked() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	registry := NewRecorderRegistry()
	registry.Register("elevenlabs", func(routing.Spec) (tts.Recorder, error) { return unvoicedRecorder{}, nil })
	recordings, err := NewRecordings(Options{Config: config[routing.TTS], Recorders: registry})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)

	recorded, _, err := recordings.Record(s.ctx, Recording{
		CustomerID: "acme", Options: options.TTS{Target: "en-recorded", Voice: "library-voice"}, Text: "hello",
	})
	s.Require().NoError(err)

	s.Empty(recorded.Voice, "a voice the recorder may have dropped is not claimed")
}
