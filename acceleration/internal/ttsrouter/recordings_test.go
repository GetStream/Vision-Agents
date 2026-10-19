package ttsrouter

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// heldRecorder is the recorder a vendor's own factory built, kept from speaking: what it
// is asked to record is held rather than sent, so nothing is dialled.
type heldRecorder struct {
	tts.Recorder
	asked *[]tts.Recording
}

func (h heldRecorder) Record(_ context.Context, recording tts.Recording) (tts.Recorded, error) {
	*h.asked = append(*h.asked, recording)
	return tts.Recorded{Audio: []byte("audio")}, nil
}

// Voice is the voice the recorder the vendor built speaks in.
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
	// built is every recorder ElevenLabs' or Cartesia's factory built, and asked every
	// recording handed to one.
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

// newRecordings routes over the built-in config with ElevenLabs' and Cartesia's own
// recorder factories, whose recorders are held rather than let speak.
func (s *RecordingsSuite) newRecordings() *Recordings {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	s.T().Setenv("CARTESIA_API_KEY", "test-key")
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	vendor := DefaultRecorderRegistry()
	registry := NewRecorderRegistry()
	for _, provider := range []string{"elevenlabs", "cartesia"} {
		registry.Register(provider, func(spec routing.Spec) (tts.Recorder, error) {
			recorder, err := vendor.Build(provider, spec)
			if err != nil {
				return nil, err
			}
			s.built = append(s.built, recorder)
			return heldRecorder{Recorder: recorder, asked: &s.asked}, nil
		})
	}

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
	// Probed on 41cb93cba: refused in these words, nothing built.
	_, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "tts-quality", Overwrites: map[string]json.RawMessage{
			"breeze": json.RawMessage(`{"voice_id":"some-voice"}`),
		}},
		Text: "hello",
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "a recording nothing here can speak in that voice is the caller's to change")
	s.ErrorContains(err, "breeze/breeze-tts-2 has no factory registered here")
	s.Empty(s.built, "no other vendor was built without the voice")
}

func (s *RecordingsSuite) TestARecordingWithAVoiceOnlyForCartesiaIsSpokenByCartesiaInIt() {
	// On 41cb93cba this was refused: "cartesia/sonic-preview has no factory registered here".
	_, served, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "tts-quality", Overwrites: map[string]json.RawMessage{
			"cartesia": json.RawMessage(`{"voice_id":"db6b0ed5-cartesia"}`),
		}},
		Text: "hello",
	})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-preview", served.Name())
	s.Equal("db6b0ed5-cartesia", s.spokenIn())
}

// recordOn checks a recording and then records it, as RecordSpeech does, and says which
// model spoke it.
func (s *RecordingsSuite) recordOn(asked options.TTS) (string, error) {
	recordings := s.newRecordings()
	recording := Recording{CustomerID: "acme", Options: asked, Text: "hello"}
	if err := recordings.Check(s.ctx, recording); err != nil {
		return "", err
	}
	_, served, err := recordings.Record(s.ctx, recording)
	return served.Name(), err
}

// recordingModels is every ElevenLabs and Cartesia model in the built-in config, which is
// every model a recording can go to.
func (s *RecordingsSuite) recordingModels() string {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	var models []string
	for _, provider := range config[routing.TTS].Providers {
		if provider.Provider == "elevenlabs" || provider.Provider == "cartesia" {
			models = append(models, provider.Name())
		}
	}
	return strings.Join(models, ", ")
}

func (s *RecordingsSuite) TestARecordingPinnedToAModelThatOnlyStreamsIsRefusedNamingTheModelsThatRecord() {
	// Probed on 41cb93cba: refused in these words, naming the ElevenLabs models only.
	for _, target := range []string{"inworld/inworld-tts-2", "breeze/breeze-tts-2"} {
		s.Run(target, func() {
			_, err := s.recordOn(options.TTS{Target: target})

			s.Require().ErrorIs(err, routing.ErrRequest, "the caller's to change")
			s.EqualError(err, "routing: "+target+" can only stream speech, not record it; models that record: "+
				s.recordingModels())
		})
	}
}

func (s *RecordingsSuite) TestCartesiaIsAVendorThatRecords() {
	recordings := s.newRecordings()

	s.True(recordings.Records("cartesia"), "can_record is true for its models")
	s.Contains(s.recordingModels(), "cartesia/sonic-preview")
}

func (s *RecordingsSuite) TestARecordingPinnedToCartesiaIsSpokenByCartesiaInItsDefaultVoice() {
	// On 41cb93cba this was refused: "cartesia/sonic-preview can only stream speech".
	s.T().Setenv("CARTESIA_VOICE_ID", "")
	served, err := s.recordOn(options.TTS{Target: "cartesia/sonic-preview", Format: "pcm_16000"})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-preview", served)
	s.Equal(cartesia.DefaultVoiceID, s.spokenIn())
	s.Equal("pcm_16000", s.asked[0].Format, "the format asked for is the one recorded")
}

func (s *RecordingsSuite) TestARecordingInAFormatCartesiaCannotProduceIsRefusedNamingIt() {
	_, err := s.recordOn(options.TTS{Target: "cartesia/sonic-preview", Format: "opus_48000_64"})

	s.Require().ErrorIs(err, routing.ErrRequest, "the caller's to change, not counted against the model")
	s.ErrorIs(err, cartesia.ErrFormat)
	s.ErrorContains(err, "opus_48000_64")
	s.Empty(s.asked, "no other file is made in its place")
}

func (s *RecordingsSuite) TestACartesiaRecorderThatCannotBeBuiltForAnotherReasonCountsAgainstTheModel() {
	recordings := s.newRecordings()
	s.T().Setenv("CARTESIA_API_KEY", "")
	recording := Recording{CustomerID: "acme", Options: options.TTS{Target: "cartesia/sonic-preview"}, Text: "hello"}

	s.Require().NoError(recordings.Check(s.ctx, recording))
	_, _, err := recordings.Record(s.ctx, recording)

	s.Require().Error(err)
	s.NotErrorIs(err, routing.ErrRequest, "a missing key is the provider's failure, not the caller's")
	s.ErrorContains(err, "api key is required")
}

func (s *RecordingsSuite) TestARecordingAskingCartesiaForWhatItCannotExpressIsRefused() {
	// On 41cb93cba these were refused because Cartesia did not record at all.
	speed, volume := 0.8, 1.2
	for name, asked := range map[string]options.TTS{
		"speed":      {Target: "cartesia/sonic-preview", Speed: &speed},
		"volume":     {Target: "cartesia/sonic-preview", Volume: &volume},
		"emotion":    {Target: "cartesia/sonic-preview", Emotion: "calm"},
		"stability":  {Target: "cartesia/sonic-preview", Stability: &speed},
		"similarity": {Target: "cartesia/sonic-preview", Similarity: &speed},
	} {
		s.Run(name, func() {
			s.asked = nil
			_, err := s.recordOn(asked)

			s.EqualError(err, "routing: cartesia/sonic-preview does not take "+name)
			s.Empty(s.asked, "nothing is spoken with the "+name+" dropped")
		})
	}
}

func (s *RecordingsSuite) TestAPriorityListOfModelsThatOnlyStreamIsRefused() {
	// mock is the one batch model in the built-in config with no recorder.
	_, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"mock"}})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, "mock/mock-tts can only stream speech, not record it")
}

func (s *RecordingsSuite) TestAPriorityListNamingCartesiaIsServedByItsBatchModel() {
	// On ef4a4cad8 this was refused: Cartesia had no batch entry, so the recording filter
	// dropped sonic-preview.
	for _, providers := range [][]string{{"cartesia"}, {"cartesia/sonic-3.6"}} {
		s.Run(strings.Join(providers, ","), func() {
			s.built = nil
			served, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: providers})

			s.Require().NoError(err)
			s.Equal("cartesia/sonic-3.6", served)
			s.Require().Len(s.built, 1)
			s.Equal("sonic-3.6", s.built[0].(interface{ Model() string }).Model(), "a model /tts/bytes lists")
		})
	}
}

func (s *RecordingsSuite) TestAPriorityListOfElevenLabsThenCartesiaFallsBackToCartesia() {
	recordings := s.newRecordings()
	s.T().Setenv("ELEVENLABS_API_KEY", "")
	providers := []string{"elevenlabs", "cartesia"}

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: providers}, Text: "hello"})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-3.6", served.Name())
}

func (s *RecordingsSuite) TestAPriorityListOfElevenLabsAloneIsStillServedByV3() {
	// Probed on 41cb93cba: served by elevenlabs/eleven_v3.
	served, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs"}})

	s.Require().NoError(err)
	s.Equal("elevenlabs/eleven_v3", served)
}

func (s *RecordingsSuite) TestTheRecordedShortcutsAreStillServedByElevenLabs() {
	// Probed on 41cb93cba: en-recorded and multilingual-recorded were served by
	// elevenlabs/eleven_v3, the one batch model; Cartesia's batch entry must not take them.
	for _, target := range []string{"en-recorded", "multilingual-recorded"} {
		s.Run(target, func() {
			served, err := s.recordOn(options.TTS{Target: target})

			s.Require().NoError(err)
			s.Equal("elevenlabs/eleven_v3", served)
		})
	}
}

func (s *RecordingsSuite) TestAPriorityListWithOneModelThatRecordsIsServedByIt() {
	// Probed on da97ee28: served by elevenlabs/eleven_v3.
	served, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"mock", "elevenlabs"}})

	s.Require().NoError(err)
	s.Equal("elevenlabs/eleven_v3", served)
}

func (s *RecordingsSuite) TestARecordingOnTheQualityShortcutIsServedByCartesiaWhenElevenLabsCannotBeBuilt() {
	// On 41cb93cba: every candidate for "tts-quality" failed, ElevenLabs having no key.
	recordings := s.newRecordings()
	s.T().Setenv("ELEVENLABS_API_KEY", "")

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme", Options: options.TTS{Target: "tts-quality"}, Text: "hello"})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-preview", served.Name())
}

func (s *RecordingsSuite) TestARecordingOnTheQualityShortcutIsServedByElevenLabs() {
	// Probed on da97ee28: served by elevenlabs/eleven_v4, the one of the six that records.
	served, err := s.recordOn(options.TTS{Target: "tts-quality"})

	s.Require().NoError(err)
	s.Equal("elevenlabs/eleven_v4", served)
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
