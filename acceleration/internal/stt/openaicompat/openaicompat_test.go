package openaicompat_test

import (
	"context"
	"embed"
	"encoding/binary"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/openaicompat"
)

//go:embed testdata/*.json
var fixtures embed.FS

type endpointReply struct {
	status     int
	body       []byte
	location   string
	waitCancel bool
}

type requestCapture struct {
	err           string
	method        string
	path          string
	authorization string
	contentType   string
	fileName      string
	audio         []byte
	model         string
	format        string
	language      string
}

type OpenAICompatSuite struct {
	suite.Suite

	server       *httptest.Server
	target       *httptest.Server
	mu           sync.RWMutex
	reply        endpointReply
	captured     chan requestCapture
	targetCalled atomic.Bool
}

func TestOpenAICompatSuite(t *testing.T) {
	suite.Run(t, new(OpenAICompatSuite))
}

func (s *OpenAICompatSuite) SetupSuite() {
	s.captured = make(chan requestCapture, 8)
	s.server = httptest.NewServer(http.HandlerFunc(s.serve))
	s.target = httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		s.targetCalled.Store(true)
	}))
}

func (s *OpenAICompatSuite) SetupTest() {
	transcript, err := fixtures.ReadFile("testdata/transcript.json")
	s.Require().NoError(err)
	s.setReply(endpointReply{status: http.StatusOK, body: transcript})
	s.targetCalled.Store(false)
	for {
		select {
		case <-s.captured:
		default:
			return
		}
	}
}

func (s *OpenAICompatSuite) TestNewRequiresExplicitEndpointAndKey() {
	s.T().Setenv("OPENAI_STT_BASE_URL", "")
	s.T().Setenv("OPENAI_STT_API_KEY", "")

	_, err := openaicompat.New(openaicompat.Options{})
	s.ErrorContains(err, "OPENAI_STT_BASE_URL")

	_, err = openaicompat.New(openaicompat.Options{BaseURL: s.server.URL})
	s.ErrorContains(err, "OPENAI_STT_API_KEY")
}

func (s *OpenAICompatSuite) TestNewUsesEnvironmentAndDefaultModel() {
	s.T().Setenv("OPENAI_STT_BASE_URL", s.server.URL+"/v1")
	s.T().Setenv("OPENAI_STT_API_KEY", "endpoint-key")

	provider, err := openaicompat.New(openaicompat.Options{})
	s.Require().NoError(err)
	s.Equal(openaicompat.DefaultModel, provider.Model())
	s.Equal(openaicompat.ProviderName, provider.Provider())
}

func (s *OpenAICompatSuite) TestNewRejectsCredentialsAndEscapedPathsInBaseURL() {
	_, err := openaicompat.New(openaicompat.Options{
		BaseURL: "http://user:password@endpoint.example/v1",
		APIKey:  "test-key",
	})
	s.ErrorContains(err, "base URL")

	_, err = openaicompat.New(openaicompat.Options{
		BaseURL: "http://endpoint.example/v1%2Fprivate",
		APIKey:  "test-key",
	})
	s.ErrorContains(err, "base URL")
}

func (s *OpenAICompatSuite) TestTranscribeUsesConfiguredMultipartEndpoint() {
	audio := pcmWAV(16000, 1600)
	provider := s.newTranscriber(s.server.URL + "/v1")
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio:     audio,
		Languages: []string{"en"},
		Channels:  1,
	})
	s.Require().NoError(err)
	s.Equal("recognized words", transcription.Text)
	s.Equal("en", transcription.Language)
	s.EqualValues(100, transcription.AudioDurationMs)

	request := s.receiveRequest()
	s.Empty(request.err)
	s.Equal(http.MethodPost, request.method)
	s.Equal("/v1/audio/transcriptions", request.path)
	s.Equal("Bearer test-key", request.authorization)
	s.Contains(request.contentType, "multipart/form-data; boundary=")
	s.Equal("audio.wav", request.fileName)
	s.Equal(audio, request.audio)
	s.Equal("whisper-1", request.model)
	s.Equal("json", request.format)
	s.Equal("en", request.language)
}

func (s *OpenAICompatSuite) TestTranscribeDoesNotFollowRedirects() {
	s.setReply(endpointReply{
		status:   http.StatusTemporaryRedirect,
		location: s.target.URL + "/collect",
		body:     []byte("redirect body"),
	})

	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "HTTP 307")
	s.False(s.targetCalled.Load(), "the configured bearer token must not reach a redirect target")
	s.Equal("Bearer test-key", s.receiveRequest().authorization)
}

func (s *OpenAICompatSuite) TestTranscribeHidesUpstreamErrorBody() {
	s.setReply(endpointReply{status: http.StatusTooManyRequests, body: []byte("private upstream detail")})
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "HTTP 429")
	s.NotContains(err.Error(), "private upstream detail")
}

func (s *OpenAICompatSuite) TestTranscribeRejectsNullTranscriptText() {
	body, err := fixtures.ReadFile("testdata/null_text.json")
	s.Require().NoError(err)
	s.setReply(endpointReply{status: http.StatusOK, body: body})

	provider := s.newTranscriber(s.server.URL)
	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "text must be a string")
}

func (s *OpenAICompatSuite) TestTranscribeBoundsTranscriptResponse() {
	s.setReply(endpointReply{
		status: http.StatusOK,
		body:   []byte(`{"text":"` + strings.Repeat("a", 2<<20) + `"}`),
	})
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "response exceeds")
}

func (s *OpenAICompatSuite) TestTranscribeStopsWhenRequestContextIsCanceled() {
	s.setReply(endpointReply{waitCancel: true})
	provider := s.newTranscriber(s.server.URL)
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		_, err := provider.Transcribe(ctx, stt.Recording{Audio: pcmWAV(16000, 1600)})
		result <- err
	}()
	<-s.captured
	cancel()

	select {
	case err := <-result:
		s.ErrorIs(err, context.Canceled)
	case <-time.After(time.Second):
		s.T().Fatal("transcription did not stop after context cancellation")
	}
}

func (s *OpenAICompatSuite) TestTranscribeRejectsUnsupportedSourcesAndFeatures() {
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{URL: "https://audio.example/clip.wav"})
	s.ErrorContains(err, "URL sources are not supported")

	_, err = provider.Transcribe(context.Background(), stt.Recording{
		Audio:           pcmWAV(16000, 1600),
		Diarize:         true,
		MaxSpeakers:     2,
		Words:           true,
		Format:          true,
		Redact:          true,
		Summary:         true,
		Entities:        true,
		Keyterms:        []string{"Vision Agents"},
		Channels:        2,
		ProfanityFilter: true,
		FillerWords:     true,
	})
	s.ErrorContains(err, "requested transcription feature is not supported")
	s.Empty(s.captured)
}

func (s *OpenAICompatSuite) TestTranscribeRequiresOneLowercaseLanguageCode() {
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio:     pcmWAV(16000, 1600),
		Languages: []string{"en-US"},
	})
	s.ErrorContains(err, "two-letter lowercase ISO 639-1")
	s.Empty(s.captured)
}

func (s *OpenAICompatSuite) TestTranscribeAcceptsSixtySecondsAndRejectsLongerAudio() {
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(24000, 24000*60)})
	s.NoError(err)
	s.receiveRequest()

	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(24000, 24000*61)})
	s.ErrorContains(err, "audio exceeds 60 seconds")
	s.Empty(s.captured)
}

func (s *OpenAICompatSuite) TestTranscribeRejectsUnsupportedWAVSampleRate() {
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(44100, 4410)})
	s.ErrorContains(err, "unsupported PCM WAV format")
	s.Empty(s.captured)
}

func (s *OpenAICompatSuite) serve(w http.ResponseWriter, r *http.Request) {
	captured := captureRequest(r)
	select {
	case s.captured <- captured:
	default:
	}
	reply := s.currentReply()
	if reply.waitCancel {
		<-r.Context().Done()
		return
	}
	if reply.location != "" {
		w.Header().Set("Location", reply.location)
	}
	if reply.status != 0 {
		w.WriteHeader(reply.status)
	}
	_, _ = w.Write(reply.body)
}

func captureRequest(r *http.Request) requestCapture {
	captured := requestCapture{
		method:        r.Method,
		path:          r.URL.Path,
		authorization: r.Header.Get("Authorization"),
		contentType:   r.Header.Get("Content-Type"),
	}
	if err := r.ParseMultipartForm(4 << 20); err != nil {
		captured.err = err.Error()
		return captured
	}
	captured.model = r.FormValue("model")
	captured.format = r.FormValue("response_format")
	captured.language = r.FormValue("language")
	file, header, err := r.FormFile("file")
	if err != nil {
		captured.err = err.Error()
		return captured
	}
	defer file.Close()
	captured.fileName = header.Filename
	captured.audio, err = io.ReadAll(file)
	if err != nil {
		captured.err = err.Error()
	}
	return captured
}

func (s *OpenAICompatSuite) currentReply() endpointReply {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.reply
}

func (s *OpenAICompatSuite) setReply(reply endpointReply) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reply = reply
}

func (s *OpenAICompatSuite) receiveRequest() requestCapture {
	s.T().Helper()
	select {
	case request := <-s.captured:
		return request
	case <-time.After(time.Second):
		s.T().Fatal("endpoint did not receive a request")
		return requestCapture{}
	}
}

func (s *OpenAICompatSuite) newTranscriber(baseURL string) *openaicompat.Transcriber {
	s.T().Helper()
	provider, err := openaicompat.New(openaicompat.Options{BaseURL: baseURL, APIKey: "test-key"})
	s.Require().NoError(err)
	return provider
}

func pcmWAV(sampleRate, frames int) []byte {
	dataBytes := frames * 2
	wav := make([]byte, 44+dataBytes)
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(len(wav)-8))
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(wav[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataBytes))
	return wav
}
