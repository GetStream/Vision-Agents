package ttsrouter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
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
