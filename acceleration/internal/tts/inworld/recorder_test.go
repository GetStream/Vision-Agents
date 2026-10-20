package inworld

import (
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// sent is one request the stub /tts/v1/voice was sent.
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
	s.last, s.status = sent{}, http.StatusOK
	s.answer = `{"audioContent":"` + base64.StdEncoding.EncodeToString([]byte("audio")) +
		`","usage":{"processedCharactersCount":5,"modelId":"inworld-tts-2"}}`
}

// newRecorder builds a recorder against the stub.
func (s *RecorderSuite) newRecorder(options RecorderOptions) *Recorder {
	options.APIKey = "k"
	options.BaseURL = s.server.URL + "/tts/v1/voice"
	recorder, err := NewRecorder(options)
	s.Require().NoError(err)
	return recorder
}

// record says the text through a recorder built from options and returns what it gave
// back and what the stub was sent.
func (s *RecorderSuite) record(options RecorderOptions, recording tts.Recording) (tts.Recorded, sent) {
	if recording.Text == "" {
		recording.Text = "hello"
	}
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
	recorded, request := s.record(RecorderOptions{VoiceID: "Ashley", Model: "inworld-tts-2", Language: "en"}, tts.Recording{})

	s.Equal("/tts/v1/voice", request.path)
	s.Equal("Basic k", request.header.Get("Authorization"), "the key is Inworld's Base64 credential, sent as is")
	s.Equal("application/json", request.header.Get("Content-Type"))
	s.JSONEq(`{"text":"hello","voiceId":"Ashley","modelId":"inworld-tts-2","language":"en",
		"audioConfig":{"audioEncoding":"MP3","sampleRateHertz":44100,"bitRate":128000}}`, request.body)
	s.Equal(tts.Recorded{Audio: []byte("audio"), Format: "mp3_44100_128", Characters: 5}, recorded,
		"the audio is decoded from base64")
}

func (s *RecorderSuite) TestTheVoiceAndLanguageARecordingNamesWinOverTheRecorders() {
	_, request := s.record(RecorderOptions{VoiceID: "Ashley", Language: "en"}, tts.Recording{Voice: "Dennis", Language: "DE"})

	s.JSONEq(`{"text":"hello","voiceId":"Dennis","modelId":"inworld-tts-2-flash","language":"de",
		"audioConfig":{"audioEncoding":"MP3","sampleRateHertz":44100,"bitRate":128000}}`, request.body)
}

func (s *RecorderSuite) TestTheBatchEntryIsSentAsTheModelIdInworldKnows() {
	_, request := s.record(RecorderOptions{VoiceID: "Ashley", Model: "inworld-tts-2-batch"}, tts.Recording{})

	s.Contains(request.body, `"modelId":"inworld-tts-2"`)
	s.Equal("inworld-tts-2-batch", s.newRecorder(RecorderOptions{Model: "inworld-tts-2-batch"}).Model(),
		"stats keep the router's name")
}

func (s *RecorderSuite) TestADeliveryModeIsSentAsTheLiveVoiceSendsIt() {
	_, request := s.record(RecorderOptions{VoiceID: "Ashley", DeliveryMode: "STABLE"}, tts.Recording{})

	s.Contains(request.body, `"deliveryMode":"STABLE"`)
}

func (s *RecorderSuite) TestARecorderNamingNoVoiceSpeaksInTheVoiceOfTheEnvironment() {
	s.T().Setenv("INWORLD_VOICE_ID", "env-voice")

	_, request := s.record(RecorderOptions{}, tts.Recording{})

	s.Equal("env-voice", s.newRecorder(RecorderOptions{}).Voice())
	s.Contains(request.body, `"voiceId":"env-voice"`)
}

func (s *RecorderSuite) TestARecorderNamingNoVoiceSpeaksInInworldsDefault() {
	s.T().Setenv("INWORLD_VOICE_ID", "")

	s.Equal(DefaultVoiceID, s.newRecorder(RecorderOptions{}).Voice())
}

func (s *RecorderSuite) TestEachFormatIsAskedOfInworldAsTheFileItNames() {
	for format, want := range map[string]string{
		"mp3_22050_32": `{"audioEncoding":"MP3","sampleRateHertz":22050,"bitRate":32000}`,
		"pcm_16000":    `{"audioEncoding":"PCM","sampleRateHertz":16000}`,
		"wav_24000":    `{"audioEncoding":"WAV","sampleRateHertz":24000}`,
		"ulaw_8000":    `{"audioEncoding":"MULAW","sampleRateHertz":8000}`,
		"alaw_8000":    `{"audioEncoding":"ALAW","sampleRateHertz":8000}`,
		"pcm_48000":    `{"audioEncoding":"PCM","sampleRateHertz":48000}`,
	} {
		s.Run(format, func() {
			recorded, request := s.record(RecorderOptions{VoiceID: "Ashley", Format: format}, tts.Recording{Format: format})

			s.JSONEq(`{"text":"hello","voiceId":"Ashley","modelId":"inworld-tts-2-flash","audioConfig":`+want+`}`,
				request.body)
			s.Equal(format, recorded.Format)
		})
	}
}

func (s *RecorderSuite) TestAFormatInworldCannotProduceIsRefusedWhenBuiltNamingIt() {
	// opus is left out, 4000 and 96000 Hz are outside 8000 to 48000, and the rest are not
	// names of a file.
	for _, format := range []string{"opus_48000_64", "opus_48000", "flac_44100", "pcm_4000", "pcm_96000", "mp3_44100", "pcm_16000_128", "mp3_44100_0", "flac"} {
		s.Run(format, func() {
			_, err := NewRecorder(RecorderOptions{APIKey: "k", Format: format})

			s.ErrorIs(err, ErrFormat)
			s.ErrorContains(err, format)
		})
	}
}

func (s *RecorderSuite) TestAFormatInworldCannotProduceIsNeverSent() {
	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "hello", Format: "opus_48000_64"})

	s.ErrorIs(err, ErrFormat)
	s.False(s.arrived(), "a different file than was asked for is never made")
}

func (s *RecorderSuite) TestTwoThousandUTF16CodeUnitsAreOneRecordingBilledByCharacter() {
	// Each emoji is two UTF-16 code units and one character.
	recorded, request := s.record(RecorderOptions{}, tts.Recording{Text: strings.Repeat("😀", 1_000)})

	s.True(request.arrived)
	s.Equal(int64(1_000), recorded.Characters)
}

func (s *RecorderSuite) TestATextOverTwoThousandUTF16CodeUnitsIsRefusedNamingTheLimit() {
	// 2,000 characters, and 2,001 code units: Inworld counts as JavaScript's text.length.
	text := strings.Repeat("a", 1_999) + "😀"

	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: text})

	s.ErrorIs(err, ErrTextTooLong)
	s.ErrorContains(err, "2001 UTF-16 code units, at most 2000")
	s.False(s.arrived(), "nothing is sent, and no part of it recorded")
}

func (s *RecorderSuite) TestInworldsRefusalIsPassedOnInItsOwnWords() {
	s.status, s.answer = http.StatusBadRequest, `{"code":5,"message":"Unknown voice: John not found!","details":[]}`

	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "hello"})

	s.EqualError(err, `inworld: record: 400 Bad Request: {"code":5,"message":"Unknown voice: John not found!","details":[]}`)
}

func (s *RecorderSuite) TestAudioThatIsNotBase64FailsTheRecording() {
	s.answer = `{"audioContent":"not base64!"}`

	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "hello"})

	s.ErrorContains(err, "inworld: decode audio")
}

func (s *RecorderSuite) TestNothingToSayIsRefusedBeforeTheCall() {
	_, err := s.newRecorder(RecorderOptions{}).Record(context.Background(), tts.Recording{Text: "  "})

	s.ErrorContains(err, "there is nothing to say")
	s.False(s.arrived(), "nothing is sent")
}

func (s *RecorderSuite) TestARecorderWithNoKeyFailsWhenBuilt() {
	s.T().Setenv("INWORLD_API_KEY", "")

	_, err := NewRecorder(RecorderOptions{})

	s.ErrorContains(err, "inworld: api key is required")
}
