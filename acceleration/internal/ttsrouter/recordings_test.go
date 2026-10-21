package ttsrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/inworld"
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

// inworldStub is Inworld's /tts/v1/voice as the http transport the recorders are built on:
// it keeps the bodies it was sent and answers a short file, or a 500 when failing.
type inworldStub struct {
	bodies  []string
	failing bool
}

func (t *inworldStub) RoundTrip(request *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	t.bodies = append(t.bodies, string(body))
	status, answer := http.StatusOK, `{"audioContent":"YXVkaW8="}`
	if t.failing {
		status, answer = http.StatusInternalServerError, `{"code":13,"message":"down"}`
	}
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(answer)),
		Request:    request,
	}, nil
}

// vendorStub is ElevenLabs', Cartesia's and Inworld's recording endpoints as the http
// transport the recorders are built on. It answers a short file, or for a host it holds a
// status for, that status and its body. It keeps the hosts it was asked, in order, and
// cancels the recording's context at the cancelAt-th request when set.
type vendorStub struct {
	asked    []string
	statuses map[string]int
	body     string
	cancelAt int
	cancel   context.CancelFunc
}

func (t *vendorStub) RoundTrip(request *http.Request) (*http.Response, error) {
	t.asked = append(t.asked, request.URL.Host)
	if len(t.asked) == t.cancelAt {
		t.cancel()
		return nil, context.Canceled
	}
	status, answer := http.StatusOK, `{"audioContent":"YXVkaW8="}`
	if failing, ok := t.statuses[request.URL.Host]; ok {
		status, answer = failing, t.body
	}
	return &http.Response{
		StatusCode: status,
		Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(answer)),
		Request:    request,
	}, nil
}

// The hosts the recorders send to (recorderURL in each vendor's recorder.go).
const (
	elevenLabsHost = "api.elevenlabs.io"
	cartesiaHost   = "api.cartesia.ai"
	inworldHost    = "api.inworld.ai"
)

type RecordingsSuite struct {
	suite.Suite
	ctx context.Context
	// built is every recorder ElevenLabs', Cartesia's or Inworld's factory built, and asked
	// every recording handed to one.
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

// newRecordings routes over the built-in config with ElevenLabs', Cartesia's and Inworld's
// own recorder factories, whose recorders are held rather than let speak.
func (s *RecordingsSuite) newRecordings() *Recordings {
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")
	s.T().Setenv("CARTESIA_API_KEY", "test-key")
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	vendor := DefaultRecorderRegistry()
	registry := NewRecorderRegistry()
	for _, provider := range []string{"elevenlabs", "cartesia", "inworld"} {
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

// speakingInworld routes over the built-in config with Inworld's own factory, whose recorder
// speaks to the stub that http.DefaultTransport is swapped for before it is built.
func (s *RecordingsSuite) speakingInworld() (*Recordings, *inworldStub) {
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	stub := &inworldStub{}
	real := http.DefaultTransport
	http.DefaultTransport = stub
	s.T().Cleanup(func() { http.DefaultTransport = real })
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	recordings, err := NewRecordings(Options{Config: config[routing.TTS], Recorders: DefaultRecorderRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)
	return recordings, stub
}

// speaking routes over the built-in config with every vendor's own factory, whose
// recorders speak to a vendorStub answering statuses, swapped in for http.DefaultTransport.
func (s *RecordingsSuite) speaking(statuses map[string]int) (*Recordings, *vendorStub) {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	return s.speakingWith(config[routing.TTS], nil, statuses)
}

// speakingWith is speaking over a config and behind a gate of the test's own.
func (s *RecordingsSuite) speakingWith(
	config routing.ModalityConfig, gate routing.Gate, statuses map[string]int,
) (*Recordings, *vendorStub) {
	for _, key := range []string{"ELEVENLABS_API_KEY", "CARTESIA_API_KEY", "INWORLD_API_KEY"} {
		s.T().Setenv(key, "test-key")
	}
	stub := &vendorStub{statuses: statuses, body: `{"detail":"down"}`}
	real := http.DefaultTransport
	http.DefaultTransport = stub
	s.T().Cleanup(func() { http.DefaultTransport = real })

	recordings, err := NewRecordings(Options{Config: config, Recorders: DefaultRecorderRegistry(), Gate: gate})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)
	return recordings, stub
}

// allowing is a policy that allows only its models.
type allowing []string

func (a allowing) Admit(context.Context, string) (routing.Admission, error) {
	return routing.Admission{Models: a}, nil
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

// recordingModels is every ElevenLabs, Cartesia and Inworld model in the built-in config,
// which is every model a recording can go to.
func (s *RecordingsSuite) recordingModels() string {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	var models []string
	for _, provider := range config[routing.TTS].Providers {
		if provider.Provider == "elevenlabs" || provider.Provider == "cartesia" || provider.Provider == "inworld" {
			models = append(models, provider.Name())
		}
	}
	return strings.Join(models, ", ")
}

func (s *RecordingsSuite) TestARecordingPinnedToAModelThatOnlyStreamsIsRefusedNamingTheModelsThatRecord() {
	// Probed on 41cb93cba: refused in these words, naming the ElevenLabs models only.
	for _, target := range []string{"breeze/breeze-tts-2"} {
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

func (s *RecordingsSuite) TestInworldIsAVendorThatRecords() {
	// Probed on b5a81650b: refused, "inworld/inworld-tts-2 can only stream speech".
	recordings := s.newRecordings()

	s.True(recordings.Records("inworld"), "can_record is true for its models")
	s.Contains(s.recordingModels(), "inworld/inworld-tts-2-batch")
}

func (s *RecordingsSuite) TestARecordingPinnedToALiveInworldModelIsSpokenByItInItsDefaultVoice() {
	// Probed on b5a81650b: refused, "inworld/inworld-tts-2 can only stream speech, not
	// record it".
	s.T().Setenv("INWORLD_VOICE_ID", "")
	for _, target := range []string{"inworld/inworld-tts-2", "inworld/inworld-tts-2-flash"} {
		s.Run(target, func() {
			s.built, s.asked = nil, nil
			served, err := s.recordOn(options.TTS{Target: target, Format: "pcm_16000"})

			s.Require().NoError(err)
			s.Equal(target, served)
			s.Equal(inworld.DefaultVoiceID, s.spokenIn())
			s.Equal("pcm_16000", s.asked[0].Format, "the format asked for is the one recorded")
		})
	}
}

func (s *RecordingsSuite) TestAPriorityListNamingInworldIsServedByItsBatchModel() {
	// Probed on b5a81650b: "routing: nothing in the priority list inworld can serve this
	// request", Inworld having no batch entry.
	for _, providers := range [][]string{{"inworld"}, {"inworld/inworld-tts-2-batch"}} {
		s.Run(strings.Join(providers, ","), func() {
			s.built, s.asked = nil, nil
			served, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: providers, Format: "ulaw_8000"})

			s.Require().NoError(err)
			s.Equal("inworld/inworld-tts-2-batch", served)
			s.Require().Len(s.built, 1)
			s.Equal("inworld-tts-2-batch", s.built[0].Model())
			s.Equal("ulaw_8000", s.asked[0].Format)
		})
	}
}

func (s *RecordingsSuite) TestAPriorityListOfElevenLabsThenInworldFallsBackToInworld() {
	// Probed on b5a81650b: "every candidate for "en-recorded" failed: elevenlabs/eleven_v3:
	// elevenlabs: api key is required".
	recordings := s.newRecordings()
	s.T().Setenv("ELEVENLABS_API_KEY", "")

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "inworld"}}, Text: "hello"})

	s.Require().NoError(err)
	s.Equal("inworld/inworld-tts-2-batch", served.Name())
}

func (s *RecordingsSuite) TestAPriorityListOfElevenLabsThenInworldIsStillServedByV3() {
	// Probed on b5a81650b: served by elevenlabs/eleven_v3.
	served, err := s.recordOn(options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "inworld"}})

	s.Require().NoError(err)
	s.Equal("elevenlabs/eleven_v3", served)
}

func (s *RecordingsSuite) TestALanguageV3LacksIsStillRecordedByCartesia() {
	// Probed on b5a81650b: multilingual-recorded asked for bn was served by
	// cartesia/sonic-3.6; Inworld's batch entry, declared after it, must not take it.
	served, err := s.recordOn(options.TTS{Target: "multilingual-recorded", Languages: []string{"bn"}})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-3.6", served)
}

func (s *RecordingsSuite) TestARecordingWithAVoiceOnlyForInworldIsSpokenByInworldInIt() {
	// Probed on b5a81650b: refused, "a voice was asked for only in overwrites.inworld, and
	// inworld/inworld-tts-2-flash, inworld/inworld-tts-2 has no factory registered here".
	_, served, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "tts-quality", Overwrites: map[string]json.RawMessage{
			"inworld": json.RawMessage(`{"voice_id":"Ashley","delivery_mode":"STABLE"}`),
		}},
		Text: "hello",
	})

	s.Require().NoError(err)
	s.Equal("inworld/inworld-tts-2-flash", served.Name(), "the first Inworld model tts-quality lists")
	s.Equal("Ashley", s.spokenIn())
}

func (s *RecordingsSuite) TestAnInworldVoiceIdOverwriteWinsOverTheVoiceAsked() {
	_, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "inworld/inworld-tts-2", Voice: "library-voice", Overwrites: map[string]json.RawMessage{
			"inworld": json.RawMessage(`{"voice_id":"Ashley"}`),
		}},
		Text: "hello",
	})

	s.Require().NoError(err)
	s.Equal("Ashley", s.spokenIn(), "the voice named for this vendor wins, as on the live path")
}

func (s *RecordingsSuite) TestAnInworldOverwriteTheRecorderHasNoFieldForIsRefused() {
	_, _, err := s.newRecordings().Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "inworld/inworld-tts-2", Overwrites: map[string]json.RawMessage{
			"inworld": json.RawMessage(`{"temperature":1.5}`),
		}},
		Text: "hello",
	})

	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorContains(err, `unknown field "temperature"`)
	s.Empty(s.asked, "nothing is spoken with the overwrite dropped")
}

func (s *RecordingsSuite) TestARecordingInAFormatInworldCannotProduceIsRefusedNamingIt() {
	_, err := s.recordOn(options.TTS{Target: "inworld/inworld-tts-2", Format: "opus_48000_64"})

	s.Require().ErrorIs(err, routing.ErrRequest, "the caller's to change, not counted against the model")
	s.ErrorIs(err, inworld.ErrFormat)
	s.ErrorContains(err, "opus_48000_64")
	s.Empty(s.asked, "no other file is made in its place")
}

func (s *RecordingsSuite) TestAnInworldDeliveryModeOverwriteReachesInworld() {
	recordings, stub := s.speakingInworld()

	_, _, err := recordings.Record(s.ctx, Recording{
		CustomerID: "acme",
		Options: options.TTS{Target: "inworld/inworld-tts-2", Overwrites: map[string]json.RawMessage{
			"inworld": json.RawMessage(`{"delivery_mode":"STABLE"}`),
		}},
		Text: "hello",
	})

	s.Require().NoError(err)
	s.Require().Len(stub.bodies, 1)
	s.Contains(stub.bodies[0], `"deliveryMode":"STABLE"`)
}

func (s *RecordingsSuite) TestATextOverInworldsLimitIsTheCallersToShortenAndSendsNothing() {
	recordings, stub := s.speakingInworld()

	_, _, err := recordings.Record(s.ctx, Recording{
		CustomerID: "acme",
		Options:    options.TTS{Target: "inworld/inworld-tts-2"},
		Text:       strings.Repeat("a", inworld.MaxRecordingText+1),
	})

	s.Require().ErrorIs(err, routing.ErrRequest, "not counted against the model")
	s.ErrorIs(err, inworld.ErrTextTooLong)
	s.ErrorContains(err, "at most 2000")
	s.Empty(stub.bodies, "no vendor call")
}

func (s *RecordingsSuite) TestARecordingAskingInworldForWhatItCannotExpressIsRefused() {
	// Probed on b5a81650b: a speed was refused because Inworld did not record at all.
	speed, volume := 0.8, 1.2
	for name, asked := range map[string]options.TTS{
		"speed":   {Target: "inworld/inworld-tts-2", Speed: &speed},
		"volume":  {Target: "inworld/inworld-tts-2", Volume: &volume},
		"emotion": {Target: "inworld/inworld-tts-2", Emotion: "calm"},
	} {
		s.Run(name, func() {
			s.asked = nil
			_, err := s.recordOn(asked)

			s.EqualError(err, "routing: inworld/inworld-tts-2 does not take "+name)
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

func (s *RecordingsSuite) TestARecordingInAFormatIsServedByCartesiasBatchModelPinnedOrListed() {
	for name, opts := range map[string]options.TTS{
		"pinned": {Target: "cartesia/sonic-3.6", Format: "pcm_16000"},
		"listed": {Target: "en-recorded", Providers: []string{"cartesia"}, Format: "pcm_16000"},
	} {
		s.Run(name, func() {
			s.asked = nil
			served, err := s.recordOn(opts)

			s.Require().NoError(err)
			s.Equal("cartesia/sonic-3.6", served)
			s.Require().Len(s.asked, 1)
			s.Equal("pcm_16000", s.asked[0].Format)
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

func (s *RecordingsSuite) TestAPriorityListFallsBackToCartesiaWhenElevenLabsFailsTheRecording() {
	// Probed on 1c5fa58db: failed, "elevenlabs: record: 500 Internal Server Error", and
	// Cartesia was never asked.
	recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusInternalServerError})

	recorded, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, Text: "hello"})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-3.6", served.Name(), "the job says who spoke it")
	s.Equal(cartesia.DefaultVoiceID, recorded.Voice)
	s.Equal([]string{elevenLabsHost, cartesiaHost}, stub.asked)
}

func (s *RecordingsSuite) TestTheRecordedShortcutFallsBackWhenElevenLabsIsOutOfCredits() {
	// Probed on 1c5fa58db: failed, "elevenlabs: record: 402 Payment Required".
	recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusPaymentRequired})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded"}, Text: "hello"})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-3.6", served.Name(), "the next en-recorded ranks")
	s.Equal([]string{elevenLabsHost, cartesiaHost}, stub.asked)
}

func (s *RecordingsSuite) TestATextOverInworldsLimitGoesToTheNextCandidate() {
	// Probed on 1c5fa58db: refused, "at most 2000", and Cartesia was never asked.
	recordings, stub := s.speaking(nil)

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"inworld", "cartesia"}},
		Text:    strings.Repeat("a", inworld.MaxRecordingText+1)})

	s.Require().NoError(err)
	s.Equal("cartesia/sonic-3.6", served.Name())
	s.Equal([]string{cartesiaHost}, stub.asked, "Inworld refused it before any call")
}

func (s *RecordingsSuite) TestACandidateThatCannotMakeTheFormatIsPassedOverForTheNext() {
	// Cartesia records no pcm_32000; Inworld does.
	recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusServiceUnavailable})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia", "inworld"}, Format: "pcm_32000"},
		Text:    "hello"})

	s.Require().NoError(err)
	s.Equal("inworld/inworld-tts-2-batch", served.Name())
	s.Equal([]string{elevenLabsHost, inworldHost}, stub.asked)
}

func (s *RecordingsSuite) TestARecordingEveryCandidateRefusesIsTheCallersToChange() {
	// Inworld takes pcm_32000 and no text over its limit; Cartesia takes the text and not
	// pcm_32000.
	recordings, stub := s.speaking(nil)

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"inworld", "cartesia"}, Format: "pcm_32000"},
		Text:    strings.Repeat("a", inworld.MaxRecordingText+1)})

	s.Require().ErrorIs(err, routing.ErrRequest, "nothing failed, so nothing is the provider's")
	s.ErrorIs(err, inworld.ErrTextTooLong)
	s.ErrorIs(err, cartesia.ErrFormat)
	s.Equal(routing.ProviderConfig{}, served, "nobody spoke it")
	s.Empty(stub.asked)
}

func (s *RecordingsSuite) TestARecordingOneCandidateRefusedAndTheNextFailedIsStillTheCallersToChange() {
	recordings, _ := s.speaking(map[string]int{cartesiaHost: http.StatusInternalServerError})

	_, _, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"inworld", "cartesia"}},
		Text:    strings.Repeat("a", inworld.MaxRecordingText+1)})

	s.Require().ErrorIs(err, routing.ErrRequest, "a refusal is the answer even when another failed, as in Select")
	s.ErrorIs(err, inworld.ErrTextTooLong)
	s.ErrorContains(err, "cartesia/sonic-3.6: cartesia: record: 500")
}

func (s *RecordingsSuite) TestARecordingEveryCandidateFailsNamesEachCandidatesError() {
	recordings, stub := s.speaking(map[string]int{
		elevenLabsHost: http.StatusInternalServerError, cartesiaHost: http.StatusTooManyRequests,
	})

	// mock has no recorder: it was never asked, so the walk does not name it.
	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "mock", "cartesia"}}, Text: "hello"})

	s.Require().Error(err)
	s.NotErrorIs(err, routing.ErrRequest, "the providers failed, not the caller")
	s.Equal("ttsrouter: no candidate recorded the text: "+
		`elevenlabs/eleven_v3: elevenlabs: record: 500 Internal Server Error: {"detail":"down"}`+"\n"+
		`cartesia/sonic-3.6: cartesia: record: 429 Too Many Requests: {"detail":"down"}`, err.Error())
	s.Equal(routing.ProviderConfig{}, served, "nobody spoke it")
	s.Equal([]string{elevenLabsHost, cartesiaHost}, stub.asked)
}

// The control below takes its expected values from 1c5fa58db, the base: a list whose only
// recorder is one model is the one-candidate case wherever the streaming-only vendor stands.
func (s *RecordingsSuite) TestARecordingListedWithAStreamingOnlyVendorFailsAsBeforeNamingOneModel() {
	for _, providers := range [][]string{{"mock", "elevenlabs"}, {"elevenlabs", "mock"}} {
		recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusInternalServerError})

		_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
			Options: options.TTS{Target: "en-recorded", Providers: providers}, Text: "hello"})

		s.Require().Error(err, providers)
		s.Equal("elevenlabs/eleven_v3", served.Name(), providers)
		s.Equal(`elevenlabs: record: 500 Internal Server Error: {"detail":"down"}`, err.Error(), providers)
		s.Equal([]string{elevenLabsHost}, stub.asked, providers)
	}
}

func (s *RecordingsSuite) TestAFailedRecordingSaysAtMostTheStartOfEachVendorsAnswer() {
	recordings, stub := s.speaking(map[string]int{
		elevenLabsHost: http.StatusInternalServerError, cartesiaHost: http.StatusInternalServerError,
	})
	stub.body = strings.Repeat("x", 10_000)

	_, _, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, Text: "hello"})

	s.Require().Error(err)
	for _, line := range strings.Split(err.Error(), "\n") {
		s.LessOrEqual(len(line),
			len("ttsrouter: no candidate recorded the text: ")+len("elevenlabs/eleven_v3: ")+maxTriedError+len("..."))
	}
}

func (s *RecordingsSuite) TestARecordingWhoseCallerHasGoneAsksNoFurtherVendor() {
	recordings, stub := s.speaking(nil)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	stub.cancelAt, stub.cancel = 1, cancel

	_, _, err := recordings.Record(ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, Text: "hello"})

	s.Require().ErrorIs(err, context.Canceled)
	s.Equal([]string{elevenLabsHost}, stub.asked)
}

// The control below takes its expected values from 1c5fa58db, the base: a pinned model
// has no next candidate, so it fails as it always has.
func (s *RecordingsSuite) TestARecordingPinnedToAModelThatFailsStillFailsAsBefore() {
	recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusInternalServerError})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "elevenlabs/eleven_v3"}, Text: "hello"})

	s.EqualError(err, `elevenlabs: record: 500 Internal Server Error: {"detail":"down"}`)
	s.Equal("elevenlabs/eleven_v3", served.Name())
	s.Equal([]string{elevenLabsHost}, stub.asked)
}

func (s *RecordingsSuite) TestARecordingWithNothingToSayIsRefusedWithoutAskingAVendor() {
	// Probed on 1c5fa58db: "tts: there is nothing to say", not a request error, so it was
	// counted against eleven_v3.
	recordings, stub := s.speaking(nil)

	_, _, err := recordings.Record(s.ctx, Recording{CustomerID: "acme",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}, Text: "  "})

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, "there is nothing to say")
	s.Empty(stub.asked)
}

// The controls below take their expected values from 1c5fa58db, the base: what Select
// rules out for a voice, a policy or an overwrite is never asked, so a walk that met one
// candidate fails as the base did, naming it.
func (s *RecordingsSuite) TestARecordingWhoseVoiceOnlyElevenLabsKnowsFailsWithElevenLabsAsBefore() {
	recordings, stub := s.speaking(map[string]int{elevenLabsHost: http.StatusInternalServerError})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme", Text: "hello",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"},
			Overwrites: map[string]json.RawMessage{"elevenlabs": json.RawMessage(`{"voice_id":"abcdefghijklmnopqrst"}`)}}})

	s.EqualError(err, `elevenlabs: record: 500 Internal Server Error: {"detail":"down"}`)
	s.NotErrorIs(err, routing.ErrRequest, "the vendor failed, not the caller")
	s.Equal("elevenlabs/eleven_v3", served.Name())
	s.Equal([]string{elevenLabsHost}, stub.asked, "Cartesia never had the voice")
}

func (s *RecordingsSuite) TestARecordingWhosePolicyAllowsOnlyElevenLabsFailsWithElevenLabsAsBefore() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	recordings, stub := s.speakingWith(config[routing.TTS], allowing{"elevenlabs/eleven_v3"},
		map[string]int{elevenLabsHost: http.StatusInternalServerError})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme", Text: "hello",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"}}})

	s.EqualError(err, `elevenlabs: record: 500 Internal Server Error: {"detail":"down"}`)
	s.Equal("elevenlabs/eleven_v3", served.Name())
	s.Equal([]string{elevenLabsHost}, stub.asked, "the policy does not allow Cartesia")
}

func (s *RecordingsSuite) TestAnOverwriteOneElevenLabsModelRefusedIsNotDroppedOnAnotherVendorWhenTheNextFails() {
	// A deployment with a second ElevenLabs batch model: eleven_v3 refuses the style
	// overwrite, eleven_multilingual_v2 takes it and fails at the vendor. Probed on
	// 1c5fa58db: failed, eleven_multilingual_v2's 500, and Cartesia was never asked.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	tts := config[routing.TTS]
	tts.Providers = slices.Clone(tts.Providers)
	for i := range tts.Providers {
		if tts.Providers[i].Name() == "elevenlabs/eleven_multilingual_v2" {
			tts.Providers[i].Realtime = false
		}
	}
	recordings, stub := s.speakingWith(tts, nil, map[string]int{elevenLabsHost: http.StatusInternalServerError})

	_, served, err := recordings.Record(s.ctx, Recording{CustomerID: "acme", Text: "hello",
		Options: options.TTS{Target: "en-recorded", Providers: []string{"elevenlabs", "cartesia"},
			Overwrites: map[string]json.RawMessage{"elevenlabs": json.RawMessage(`{"voice_settings":{"style":0.5}}`)}}})

	s.Require().ErrorIs(err, routing.ErrOverwrite)
	s.ErrorContains(err, `elevenlabs/eleven_multilingual_v2: elevenlabs: record: 500 Internal Server Error`)
	s.Equal(routing.ProviderConfig{}, served, "nobody spoke it")
	s.Equal([]string{elevenLabsHost}, stub.asked, "Cartesia would drop the style in silence")
}
