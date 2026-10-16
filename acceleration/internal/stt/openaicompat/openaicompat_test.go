package openaicompat_test

import (
	"context"
	"embed"
	"encoding/binary"
	"encoding/json"
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
	status      int
	body        []byte
	contentType string
	location    string
	waitCancel  bool
}

type requestCapture struct {
	err             string
	method          string
	path            string
	authorization   string
	contentType     string
	fileName        string
	fileType        string
	audio           []byte
	model           string
	format          string
	language        string
	languages       []string
	prompt          string
	keywords        []string
	temperature     string
	granularities   []string
	include         []string
	chunking        string
	chunkingType    string
	chunkingSilence string
	speakers        []string
	references      []string
	stream          string
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

func (s *OpenAICompatSuite) TestNewRequiresEndpointAndAllowsAKeylessExplicitEndpoint() {
	s.T().Setenv("OPENAI_STT_BASE_URL", "")
	s.T().Setenv("OPENAI_STT_API_KEY", "")

	_, err := openaicompat.New(openaicompat.Options{})
	s.ErrorContains(err, "OPENAI_STT_BASE_URL")

	s.T().Setenv("OPENAI_STT_API_KEY", "unrelated-env-key")
	provider, err := openaicompat.New(openaicompat.Options{BaseURL: s.server.URL})
	s.Require().NoError(err)
	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.NoError(err)
	s.Empty(s.receiveRequest().authorization, "an explicit endpoint must not inherit an environment key")
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
	s.Zero(transcription.AudioDurationMs, "the JSON response did not report duration")

	request := s.receiveRequest()
	s.Empty(request.err)
	s.Equal(http.MethodPost, request.method)
	s.Equal("/v1/audio/transcriptions", request.path)
	s.Equal("Bearer test-key", request.authorization)
	s.Contains(request.contentType, "multipart/form-data; boundary=")
	s.Equal("audio.wav", request.fileName)
	s.Equal("audio/wav", request.fileType)
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
	s.setReply(endpointReply{status: http.StatusTooManyRequests, body: []byte(`{"error":{"message":"private upstream detail","type":"rate_limit_error","code":"rate_limit_exceeded","param":"model"}}`)})
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "HTTP 429")
	s.ErrorContains(err, "type=rate_limit_error")
	s.ErrorContains(err, "code=rate_limit_exceeded")
	s.ErrorContains(err, "param=model")
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

func (s *OpenAICompatSuite) TestTranscribeAcceptsEmptyTextAsSilence() {
	s.setReply(endpointReply{status: http.StatusOK, body: []byte(`{"text":""}`)})
	provider := s.newTranscriber(s.server.URL)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.NoError(err)
	s.Empty(transcription.Text)
	s.receiveRequest()
}

func (s *OpenAICompatSuite) TestCanonicalEmptyTextDoesNotResurrectStaleSegments() {
	s.setReply(endpointReply{status: http.StatusOK, body: []byte(`{"text":"","segments":[{"text":"stale text"}]}`)})
	provider := s.newTranscriber(s.server.URL)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.NoError(err)
	s.Empty(transcription.Text)
	s.receiveRequest()
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

func (s *OpenAICompatSuite) TestTranscribeForwardsMultipleLanguageHints() {
	provider := s.newTranscriber(s.server.URL)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio:     pcmWAV(16000, 1600),
		Languages: []string{"en-US", "pt-BR"},
	})
	s.NoError(err)
	s.Equal("recognized words", transcription.Text)
	request := s.receiveRequest()
	s.Empty(request.language)
	s.Equal([]string{"en-US", "pt-BR"}, request.languages)
}

func (s *OpenAICompatSuite) TestBatchOptionsLanguagesAreFallbackHints() {
	for _, testCase := range []struct {
		name      string
		languages []string
		want      string
		wantMany  []string
	}{
		{name: "single", languages: []string{"pt-BR"}, want: "pt-BR"},
		{name: "multiple", languages: []string{"pt-BR", "en-US"}, wantMany: []string{"pt-BR", "en-US"}},
	} {
		s.Run(testCase.name, func() {
			provider, err := openaicompat.New(openaicompat.Options{
				BaseURL:   s.server.URL,
				Languages: testCase.languages,
			})
			s.Require().NoError(err)
			_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
			s.NoError(err)
			request := s.receiveRequest()
			s.Equal(testCase.want, request.language)
			s.Equal(testCase.wantMany, request.languages)
		})
	}
}

func (s *OpenAICompatSuite) TestTranscribeAcceptsAudioLongerThanSixtySeconds() {
	provider := s.newTranscriber(s.server.URL)
	audio := pcmWAV(24000, 24000*61)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: audio})
	s.NoError(err)
	s.Equal(audio, s.receiveRequest().audio)
}

func (s *OpenAICompatSuite) TestTranscribeForwardsOriginalStereoAndHighRateAudio() {
	provider := s.newTranscriber(s.server.URL)
	audio := pcmWAVChannels(96000, 2, 9600)
	_, err := provider.Transcribe(context.Background(), stt.Recording{Audio: audio})
	s.NoError(err)
	s.Equal(audio, s.receiveRequest().audio)
}

func (s *OpenAICompatSuite) TestTranscribeMapsSharedFeaturesAndTypedStandardSettings() {
	stream := true
	temperature := 0.25
	provider, err := openaicompat.New(openaicompat.Options{
		BaseURL:  s.server.URL,
		APIKey:   "test-key",
		Keyterms: []string{"Vision Agents"},
		Settings: openaicompat.TranscriptionSettings{
			Prompt:                 "clinical calls",
			Language:               "en-US",
			Keywords:               []string{"Stream"},
			Temperature:            &temperature,
			TimestampGranularities: []string{"segment"},
			Include:                []string{"logprobs"},
			ChunkingStrategy:       json.RawMessage(`"auto"`),
			KnownSpeakerNames:      []string{"agent"},
			KnownSpeakerReferences: []string{"data:audio/wav;base64,AAAA"},
			Stream:                 &stream,
		},
	})
	s.Require().NoError(err)
	_, err = provider.Transcribe(context.Background(), stt.Recording{
		Audio:     pcmWAV(16000, 1600),
		Languages: []string{"pt-BR", "en-US"},
		Keyterms:  []string{"Vision Agents"},
		Words:     true,
		Format:    true,
	})
	s.NoError(err)
	request := s.receiveRequest()
	s.Equal("en-US", request.language)
	s.Empty(request.languages)
	s.Equal("verbose_json", request.format)
	s.Equal("clinical calls\nVocabulary: Vision Agents.", request.prompt)
	s.Equal([]string{"Stream"}, request.keywords, "shared keyterms use the generic prompt mapping")
	s.Equal("0.25", request.temperature)
	s.Equal([]string{"segment", "word"}, request.granularities)
	s.Equal([]string{"logprobs"}, request.include)
	s.Equal("auto", request.chunking)
	s.Equal([]string{"agent"}, request.speakers)
	s.Equal([]string{"data:audio/wav;base64,AAAA"}, request.references)
	s.Equal("true", request.stream)
}

func (s *OpenAICompatSuite) TestTranscribeReturnsProviderDurationAndOnlyRealWordTimestamps() {
	s.setReply(endpointReply{status: http.StatusOK, body: []byte(`{"text":"go now","languages":[{"code":"fr"}],"duration":12.3,"usage":{"type":"duration","seconds":12.6},"words":[{"word":"go","start":0,"end":0.5},{"word":"now","start":0.6}]}`)})
	provider := s.newTranscriber(s.server.URL)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio: pcmWAV(16000, 1600), Words: true,
	})
	s.NoError(err)
	s.Equal("go now", transcription.Text)
	s.Equal("fr", transcription.Language, "single-item language arrays map to the router's singular language field")
	s.EqualValues(12300, transcription.AudioDurationMs, "duration is the recording length; usage seconds are a fallback")
	s.Equal([]stt.Word{{Text: "go", StartMs: 0, EndMs: 500}}, transcription.Words)
	s.receiveRequest()
}

func (s *OpenAICompatSuite) TestTranscribeReadsDiarizedSegmentsWithoutInventingWordTimings() {
	s.setReply(endpointReply{status: http.StatusOK, body: []byte(`{"text":"Agent: hello","duration":1.25,"segments":[{"type":"transcript.text.segment","start":0,"end":1.25,"text":"hello","speaker":"agent"}]}`)})
	provider := s.newTranscriber(s.server.URL)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio: pcmWAV(16000, 1600), Diarize: true,
	})
	s.NoError(err)
	s.Equal("diarized_json", s.receiveRequest().format)
	s.Equal("Agent: hello", transcription.Text)
	s.EqualValues(1250, transcription.AudioDurationMs)
	s.Equal([]string{"agent"}, transcription.Speakers)
	s.Empty(transcription.Words)
}

func (s *OpenAICompatSuite) TestTranscribeCollectsServerSentEvents() {
	s.setReply(endpointReply{
		status:      http.StatusOK,
		contentType: "text/event-stream",
		body: []byte("data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n" +
			"data: {\"type\":\"transcript.text.delta\",\"delta\":\" world\"}\n\n" +
			"data: {\"type\":\"transcript.text.segment\",\"start\":0,\"end\":1.5,\"text\":\"hello world\",\"speaker\":\"A\"}\n\n" +
			"data: {\"type\":\"transcript.text.done\",\"text\":\"hello world\",\"usage\":{\"seconds\":1.5}}\n\n"),
	})
	stream := true
	provider, err := openaicompat.New(openaicompat.Options{
		BaseURL: s.server.URL, APIKey: "test-key",
		Settings: openaicompat.TranscriptionSettings{Stream: &stream},
	})
	s.Require().NoError(err)
	transcription, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.NoError(err)
	request := s.receiveRequest()
	s.Equal("true", request.stream)
	s.Equal("hello world", transcription.Text)
	s.EqualValues(1500, transcription.AudioDurationMs)
	s.Equal([]string{"A"}, transcription.Speakers)
}

func (s *OpenAICompatSuite) TestTranscribeRejectsIncompleteServerSentEvents() {
	s.setReply(endpointReply{
		status:      http.StatusOK,
		contentType: "text/event-stream",
		body:        []byte("data: {\"type\":\"transcript.text.delta\",\"delta\":\"partial\"}\n\n"),
	})
	stream := true
	provider, err := openaicompat.New(openaicompat.Options{
		BaseURL: s.server.URL, Settings: openaicompat.TranscriptionSettings{Stream: &stream},
	})
	s.Require().NoError(err)
	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.ErrorContains(err, "transcript.text.done")
	s.receiveRequest()
}

func (s *OpenAICompatSuite) TestTranscribeUsesBracketedChunkingStrategyFields() {
	provider, err := openaicompat.New(openaicompat.Options{
		BaseURL: s.server.URL,
		Settings: openaicompat.TranscriptionSettings{
			ChunkingStrategy: json.RawMessage(`{"type":"server_vad","silence_duration_ms":400}`),
		},
	})
	s.Require().NoError(err)
	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
	s.NoError(err)
	request := s.receiveRequest()
	s.Equal("server_vad", request.chunkingType)
	s.Equal("400", request.chunkingSilence)
}

func (s *OpenAICompatSuite) TestTranscribeConsumesTextAndSubtitleFormats() {
	for _, reply := range []endpointReply{
		{status: http.StatusOK, contentType: "text/plain", body: []byte("recognized words")},
		{status: http.StatusOK, contentType: "application/x-subrip", body: []byte("1\n00:00:00,000 --> 00:00:01,000\nrecognized words\n")},
		{status: http.StatusOK, contentType: "text/vtt", body: []byte("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nrecognized words\n")},
	} {
		s.setReply(reply)
		provider := s.newTranscriber(s.server.URL)
		transcription, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
		s.NoError(err)
		s.Equal("recognized words", transcription.Text)
		s.Empty(transcription.Words, "subtitle cue boundaries are not word timestamps")
		s.receiveRequest()
	}
}

func (s *OpenAICompatSuite) TestTranscribeHonorsTextAndSubtitleResponseFormats() {
	for _, test := range []struct {
		format string
		body   string
		want   string
	}{
		{format: "text", body: "{hello}", want: "{hello}"},
		{format: "text", body: `"hello"`, want: `"hello"`},
		{format: "srt", body: "1\n00:00:00,000 --> 00:00:01,000\n42\n", want: "42"},
		{format: "vtt", body: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nrecognized words\n", want: "recognized words"},
		{format: "vtt", body: "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n42\n\n2\n00:00:01.000 --> 00:00:02.000\nnext\n", want: "42 next"},
	} {
		s.setReply(endpointReply{status: http.StatusOK, contentType: "text/plain", body: []byte(test.body)})
		provider, err := openaicompat.New(openaicompat.Options{
			BaseURL:  s.server.URL,
			Settings: openaicompat.TranscriptionSettings{ResponseFormat: test.format},
		})
		s.Require().NoError(err)
		transcription, err := provider.Transcribe(context.Background(), stt.Recording{Audio: pcmWAV(16000, 1600)})
		s.NoError(err)
		s.Equal(test.want, transcription.Text)
		s.receiveRequest()
	}
}

func (s *OpenAICompatSuite) TestTranscribePreservesCompressedAudioAndFileMetadata() {
	audio := append([]byte("ID3\x04\x00\x00"), []byte("compressed fixture")...)
	provider := s.newTranscriber(s.server.URL)
	_, err := provider.Transcribe(context.Background(), stt.Recording{
		Audio: audio, Filename: "call.mp3", ContentType: "audio/mpeg",
	})
	s.NoError(err)
	request := s.receiveRequest()
	s.Equal("call.mp3", request.fileName)
	s.Equal("audio/mpeg", request.fileType)
	s.Equal(audio, request.audio)

	_, err = provider.Transcribe(context.Background(), stt.Recording{Audio: audio, Filename: "call"})
	s.NoError(err)
	request = s.receiveRequest()
	s.Equal("call.mp3", request.fileName, "a metadata name without an extension gets the detected suffix")
	s.Equal("audio/mpeg", request.fileType)
	s.Equal(audio, request.audio)
}

func (s *OpenAICompatSuite) TestTranscribeForwardsOggOpusBytesAndFilename() {
	audio := append([]byte("OggS"), []byte("synthetic opus payload")...)
	for _, testCase := range []struct {
		name        string
		filename    string
		contentType string
		wantName    string
	}{
		{name: "known Opus filename", filename: "voice.opus", contentType: "audio/ogg; codecs=opus", wantName: "voice.opus"},
		{name: "sniffed Ogg filename", wantName: "audio.ogg"},
	} {
		s.Run(testCase.name, func() {
			provider := s.newTranscriber(s.server.URL)
			_, err := provider.Transcribe(context.Background(), stt.Recording{
				Audio: audio, Filename: testCase.filename, ContentType: testCase.contentType,
			})
			s.NoError(err)
			request := s.receiveRequest()
			s.Equal(testCase.wantName, request.fileName)
			s.Equal("audio/ogg", request.fileType)
			s.Equal(audio, request.audio)
		})
	}
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
	if reply.contentType != "" {
		w.Header().Set("Content-Type", reply.contentType)
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
	captured.languages = r.Form["languages[]"]
	captured.prompt = r.FormValue("prompt")
	captured.keywords = r.Form["keywords[]"]
	captured.temperature = r.FormValue("temperature")
	captured.granularities = r.Form["timestamp_granularities[]"]
	captured.include = r.Form["include[]"]
	captured.chunking = r.FormValue("chunking_strategy")
	captured.chunkingType = r.FormValue("chunking_strategy[type]")
	captured.chunkingSilence = r.FormValue("chunking_strategy[silence_duration_ms]")
	captured.speakers = r.Form["known_speaker_names[]"]
	captured.references = r.Form["known_speaker_references[]"]
	captured.stream = r.FormValue("stream")
	file, header, err := r.FormFile("file")
	if err != nil {
		captured.err = err.Error()
		return captured
	}
	defer file.Close()
	captured.fileName = header.Filename
	captured.fileType = header.Header.Get("Content-Type")
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
	return pcmWAVChannels(sampleRate, 1, frames)
}

func pcmWAVChannels(sampleRate, channels, frames int) []byte {
	blockAlign := channels * 2
	dataBytes := frames * blockAlign
	wav := make([]byte, 44+dataBytes)
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(len(wav)-8))
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(wav[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(wav[28:32], uint32(sampleRate*blockAlign))
	binary.LittleEndian.PutUint16(wav[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataBytes))
	return wav
}
