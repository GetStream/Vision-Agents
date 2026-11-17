package cartesia

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// sent is one request the stub /tts/bytes was sent.
type sent struct {
	path    string
	header  http.Header
	body    string
	arrived bool
}

type RecorderSuite struct {
	suite.Suite
	server *httptest.Server

	mu sync.Mutex
	// last is the request the stub was sent, and status and answer what it says back.
	last   sent
	status int
	answer string
}

func TestRecorderSuite(t *testing.T) {
	suite.Run(t, new(RecorderSuite))
}

func (s *RecorderSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		defer s.mu.Unlock()
		s.last = sent{path: r.URL.Path, header: r.Header.Clone(), body: string(body), arrived: true}
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.answer))
	}))
}

func (s *RecorderSuite) TearDownSuite() { s.server.Close() }

func (s *RecorderSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.last, s.status, s.answer = sent{}, http.StatusOK, "audio"
}

// newRecorder builds a recorder against the stub.
func (s *RecorderSuite) newRecorder(options RecorderOptions) *Recorder {
	options.APIKey = "k"
	options.BaseURL = s.server.URL + "/tts/bytes"
	recorder, err := NewRecorder(options)
	s.Require().NoError(err)
	return recorder
}

// record says hello through a recorder built from options and returns what it gave back
// and what the stub was sent.
func (s *RecorderSuite) record(options RecorderOptions, recording tts.Recording) (tts.Recorded, sent) {
	recording.Text = "hello"
	recorded, err := s.newRecorder(options).Record(context.Background(), recording)
	s.Require().NoError(err)

	s.mu.Lock()
	defer s.mu.Unlock()
	return recorded, s.last
}

// arrived reports whether the stub was sent anything.
func (s *RecorderSuite) arrived() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last.arrived
}

func (s *RecorderSuite) TestARecordingThatAsksForNothingIsAnMp3OfTheRecordersVoice() {
	recorded, request := s.record(RecorderOptions{VoiceID: "v1", Model: "sonic-preview", Language: "en"}, tts.Recording{})

	s.Equal("/tts/bytes", request.path)
	s.Equal("Bearer k", request.header.Get("Authorization"))
	s.Equal("2026-08-14", request.header.Get("Cartesia-Version"))
	s.Equal("application/json", request.header.Get("Content-Type"))
	s.JSONEq(`{"model_id":"sonic-preview","transcript":"hello","voice":{"id":"v1"},"language":"en",
		"output_format":{"container":"mp3","sample_rate":44100,"bit_rate":128000}}`, request.body)
	s.Equal(tts.Recorded{Audio: []byte("audio"), Format: "mp3_44100_128", Characters: 5}, recorded)
}

func (s *RecorderSuite) TestTheVoiceAndLanguageARecordingNamesWinOverTheRecorders() {
	_, request := s.record(RecorderOptions{VoiceID: "v1", Language: "en"}, tts.Recording{Voice: "v2", Language: "de"})

	s.JSONEq(`{"model_id":"sonic-preview","transcript":"hello","voice":{"id":"v2"},"language":"de",
		"output_format":{"container":"mp3","sample_rate":44100,"bit_rate":128000}}`, request.body)
}

func (s *RecorderSuite) TestALanguageIsSentInLowerCaseWhoseverNamedIt() {
	_, fromRecording := s.record(RecorderOptions{VoiceID: "v1"}, tts.Recording{Language: "FR"})
	_, fromRecorder := s.record(RecorderOptions{VoiceID: "v1", Language: "DE"}, tts.Recording{})

	s.Contains(fromRecording.body, `"language":"fr"`)
	s.Contains(fromRecorder.body, `"language":"de"`)
}

func (s *RecorderSuite) TestARecorderNamingNoVoiceSpeaksInTheVoiceOfTheEnvironment() {
	s.T().Setenv("CARTESIA_VOICE_ID", "env-voice")

	recorded, request := s.record(RecorderOptions{}, tts.Recording{})

	s.Equal("env-voice", s.newRecorder(RecorderOptions{}).Voice())
	s.Contains(request.body, `"voice":{"id":"env-voice"}`)
	s.Equal("mp3_44100_128", recorded.Format)
}

func (s *RecorderSuite) TestARecorderNamingNoVoiceSpeaksInCartesiasDefault() {
	s.T().Setenv("CARTESIA_VOICE_ID", "")

	s.Equal(DefaultVoiceID, s.newRecorder(RecorderOptions{}).Voice())
}

func (s *RecorderSuite) TestEachFormatIsAskedOfCartesiaAsTheFileItNames() {
	for format, want := range map[string]string{
		"mp3_22050_32": `{"container":"mp3","sample_rate":22050,"bit_rate":32000}`,
		"pcm_16000":    `{"container":"raw","encoding":"pcm_s16le","sample_rate":16000}`,
		"wav_24000":    `{"container":"wav","encoding":"pcm_s16le","sample_rate":24000}`,
		"ulaw_8000":    `{"container":"raw","encoding":"pcm_mulaw","sample_rate":8000}`,
		"alaw_8000":    `{"container":"raw","encoding":"pcm_alaw","sample_rate":8000}`,
	} {
		s.Run(format, func() {
			recorded, request := s.record(RecorderOptions{VoiceID: "v1", Format: format}, tts.Recording{Format: format})

			s.JSONEq(`{"model_id":"sonic-preview","transcript":"hello","voice":{"id":"v1"},"output_format":`+want+`}`,
				request.body)
			s.Equal(format, recorded.Format)
		})
	}
}

func (s *RecorderSuite) TestAFormatCartesiaCannotProduceIsRefusedWhenBuiltNamingIt() {
	// opus and a 48 kbps mp3 are ElevenLabs formats Cartesia has no container or bit rate
	// for; 32000 Hz is not one of its sample rates.
	for _, format := range []string{"opus_48000_64", "mp3_24000_48", "pcm_32000", "mp3_44100", "pcm_16000_128", "flac"} {
		s.Run(format, func() {
			_, err := NewRecorder(RecorderOptions{APIKey: "k", Format: format})

			s.ErrorIs(err, ErrFormat)
			s.ErrorContains(err, format)
		})
	}
}

func (s *RecorderSuite) TestAFormatCartesiaCannotProduceIsNeverSent() {
	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "hello", Format: "opus_48000_64"})

	s.ErrorIs(err, ErrFormat)
	s.False(s.arrived(), "a different file than was asked for is never made")
}

func (s *RecorderSuite) TestCartesiasRefusalIsPassedOnInItsOwnWords() {
	s.status, s.answer = http.StatusPaymentRequired, `{"error":"insufficient credits"}`

	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "hello"})

	s.EqualError(err, `cartesia: record: 402 Payment Required: {"error":"insufficient credits"}`)
}

func (s *RecorderSuite) TestNothingToSayIsRefusedBeforeTheCall() {
	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "  "})

	s.ErrorContains(err, "there is nothing to say")
	s.False(s.arrived(), "nothing is sent")
}

func (s *RecorderSuite) TestARecorderWithNoKeyFailsWhenBuilt() {
	s.T().Setenv("CARTESIA_API_KEY", "")

	_, err := NewRecorder(RecorderOptions{})

	s.ErrorContains(err, "cartesia: api key is required")
}
