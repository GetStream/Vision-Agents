package elevenlabs

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

// recorded is one request the stub synthesis endpoint was sent.
type recorded struct {
	path string
	body string
}

type RecorderSuite struct {
	suite.Suite
	server *httptest.Server

	mu   sync.Mutex
	last recorded
}

func TestRecorderSuite(t *testing.T) {
	suite.Run(t, new(RecorderSuite))
}

func (s *RecorderSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.last = recorded{path: r.URL.Path, body: string(body)}
		s.mu.Unlock()
		_, _ = w.Write([]byte("audio"))
	}))
}

func (s *RecorderSuite) SetupTest() {
	s.mu.Lock()
	s.last = recorded{}
	s.mu.Unlock()
}

// record speaks hello through a recorder built from options, against the stub, and
// returns what the stub was sent.
func (s *RecorderSuite) record(options RecorderOptions, recording tts.Recording) recorded {
	options.APIKey = "k"
	options.BaseURL = s.server.URL
	recorder, err := NewRecorder(options)
	s.Require().NoError(err)

	recording.Text = "hello"
	_, err = recorder.Record(context.Background(), recording)
	s.Require().NoError(err)

	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

func (s *RecorderSuite) TestARecordingThatAsksForNothingSendsWhatItAlwaysHas() {
	// The request this recorder sent before a style could be asked for, captured on the
	// commit before it could (a5721c90).
	sent := s.record(RecorderOptions{VoiceID: "v1", Model: "eleven_v3"}, tts.Recording{})

	s.Equal("/v1", sent.path, "a recording that names no voice is spoken in the recorder's")
	s.JSONEq(`{"text":"hello","model_id":"eleven_v3"}`, sent.body)
}

func (s *RecorderSuite) TestAStyleIsSentToAModelThatTakesOne() {
	style := 0.4

	sent := s.record(RecorderOptions{VoiceID: "v1", Model: "eleven_multilingual_v2", Style: &style}, tts.Recording{})

	s.JSONEq(`{"text":"hello","model_id":"eleven_multilingual_v2","voice_settings":{"style":0.4}}`, sent.body)
}

func (s *RecorderSuite) TestAStyleIsRefusedByTheV3RecordingModel() {
	// No ElevenLabs doc says v3 reads a style, and live on 2026-10-16 its file endpoint took
	// 1.5 without a word, so a style sent there would be dropped in silence.
	style := 0.4

	_, err := NewRecorder(RecorderOptions{APIKey: "k", Model: "eleven_v3", Style: &style})

	s.ErrorIs(err, ErrStyle)
	s.ErrorContains(err, "style is not taken by eleven_v3")
}
