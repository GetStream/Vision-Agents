// Package gemini implements the tts.TTS contract on Google's Gemini TTS models, over
// streamGenerateContent.
//
// It is a request per utterance rather than a socket: the model takes a whole transcript
// and streams the speech for it back as server-sent events, each carrying a piece of raw
// PCM. There is no way to feed it text a delta at a time, so partial text is buffered until
// the utterance is final, and the audio is still played as it arrives.
//
// The request is framed the way the Live API under internal/sts frames its setup:
// contents and parts in, inlineData out. The two share no code, because one is a
// conversation on a socket and the other is one POST per sentence.
package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// DefaultModel is Gemini 3.8 Flash TTS, the expressive one of the two 3.8 voices.
const DefaultModel = "gemini-3.8-flash-tts"

// DefaultBaseURL is the Gemini API, versioned. The model and method are appended per
// request.
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// OutputSampleRate is what the provider asks for, and the one rate every Gemini voice
// speaks at.
const OutputSampleRate = 24_000

// apiKeyEnvVar holds the credentials when Options does not. It is the name the other
// Gemini packages and the Python side of this repository already use.
const apiKeyEnvVar = "GOOGLE_API_KEY"

// audioFormat is raw 16-bit PCM. It is the streaming default today, but 3.8 already moved
// the unary default from PCM to WAV, so it is asked for rather than assumed.
const audioFormat = "AUDIO_L16"

// Options configures the provider. APIKey falls back to GOOGLE_API_KEY.
type Options struct {
	APIKey string
	Model  string
	// Voice is a prebuilt voice such as Kore, or a designed or replicated voice_ id.
	// Empty leaves the model's own.
	Voice string
	// Language is the code to speak in, such as en or en-US. Empty lets the model detect
	// it from the text.
	Language string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// Timeout bounds one synthesis, including reading the audio back.
	Timeout time.Duration
	// HTTPClient overrides the client, for a custom transport.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// generateRequest is the body of streamGenerateContent.
type generateRequest struct {
	Contents         []content        `json:"contents"`
	GenerationConfig generationConfig `json:"generationConfig"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text       string `json:"text,omitempty"`
	InlineData *blob  `json:"inlineData,omitempty"`
}

type blob struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type generationConfig struct {
	ResponseModalities []string        `json:"responseModalities"`
	ResponseFormat     *responseFormat `json:"responseFormat,omitempty"`
	SpeechConfig       *speechConfig   `json:"speechConfig,omitempty"`
}

type responseFormat struct {
	Audio audioResponseFormat `json:"audio"`
}

type audioResponseFormat struct {
	MimeType   string `json:"mimeType"`
	SampleRate int    `json:"sampleRate"`
}

type speechConfig struct {
	VoiceConfig  *voiceConfig `json:"voiceConfig,omitempty"`
	LanguageCode string       `json:"languageCode,omitempty"`
}

// voiceConfig names the voice. The 3.8 models take it as a plain voice field, which is
// what accepts a designed or replicated voice_ id as well as a prebuilt name; the Live API
// still wants prebuiltVoiceConfig.
type voiceConfig struct {
	Voice string `json:"voice"`
}

// streamEvent is one server-sent event: a piece of the audio, or the error that ended it.
type streamEvent struct {
	Candidates []candidate `json:"candidates"`
	Error      *apiError   `json:"error"`
}

type candidate struct {
	Content      *content `json:"content"`
	FinishReason string   `json:"finishReason"`
}

type apiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// pending is an utterance being assembled from deltas. The tracker is created on the
// first delta so the reported latency covers the wait the caller actually experienced,
// not just the request this provider eventually makes.
type pending struct {
	tracker *tts.Synthesis
	text    strings.Builder
}

// TTS is a Gemini text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	client  *http.Client
	emitter *tts.Emitter

	mu sync.Mutex
	// pending holds utterances still being assembled from deltas.
	pending map[string]*pending
	// inFlight cancels syntheses that are being generated, which is how barge-in works.
	inFlight map[string]context.CancelFunc
	ctx      context.Context
	cancel   context.CancelFunc
	// running counts syntheses in flight so Close can wait for them.
	running  sync.WaitGroup
	started  bool
	shutdown bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("gemini: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultBaseURL
	}
	if !strings.HasPrefix(options.BaseURL, "http://") && !strings.HasPrefix(options.BaseURL, "https://") {
		return nil, fmt.Errorf("gemini: base url must be http:// or https://, got %s", options.BaseURL)
	}
	if options.Timeout == 0 {
		options.Timeout = 60 * time.Second
	}
	client := options.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &TTS{
		options:  options,
		logger:   logger.With("provider", ProviderName, "model", options.Model),
		client:   client,
		emitter:  tts.NewEmitter(64),
		pending:  map[string]*pending{},
		inFlight: map[string]context.CancelFunc{},
	}, nil
}

// Start prepares the session. There is no connection to open, so this only fixes the
// lifetime that syntheses run within.
func (t *TTS) Start(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.started {
		return errors.New("gemini: already started")
	}
	t.started = true
	t.ctx, t.cancel = context.WithCancel(context.WithoutCancel(ctx))

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	return nil
}

// Synthesize buffers text and, once the utterance is final, sends it to Gemini. Audio is
// emitted as the response streams in.
func (t *TTS) Synthesize(request tts.Request) error {
	synthesis, text, ready, err := t.accumulate(request)
	if err != nil {
		return err
	}
	if !ready {
		return nil
	}

	t.emitter.Send(tts.SynthesisStarted{
		SynthesisID: synthesis.ID,
		Provider:    ProviderName,
		Model:       t.options.Model,
		Voice:       t.voiceFor(request),
		At:          time.Now(),
	})

	ctx, cancel := context.WithTimeout(t.ctx, t.options.Timeout)
	t.mu.Lock()
	t.inFlight[synthesis.ID] = cancel
	t.mu.Unlock()

	t.running.Add(1)
	go func() {
		defer t.running.Done()
		defer cancel()
		t.synthesize(ctx, synthesis, text, request)
	}()
	return nil
}

// Interrupt cancels every synthesis in flight, which closes the stream and stops any
// further audio reaching the caller.
func (t *TTS) Interrupt() error {
	t.mu.Lock()
	cancels := make([]context.CancelFunc, 0, len(t.inFlight))
	for _, cancel := range t.inFlight {
		cancels = append(cancels, cancel)
	}
	// Text buffered for utterances that were never finished is no longer wanted.
	clear(t.pending)
	t.mu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close cancels anything in flight and waits for it to report, so no synthesis goes
// unaccounted for.
func (t *TTS) Close() error {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil
	}
	t.shutdown = true
	cancel := t.cancel
	t.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	t.running.Wait()

	t.emitter.Send(tts.Disconnected{
		Provider: ProviderName,
		Model:    t.options.Model,
		Clean:    true,
		At:       time.Now(),
	})
	t.emitter.Close()
	return nil
}

// Provider implements tts.TTS.
func (t *TTS) Provider() string { return ProviderName }

// Model implements tts.TTS.
func (t *TTS) Model() string { return t.options.Model }

// Voice is the session's voice, or empty when Gemini picks one.
func (t *TTS) Voice() string { return t.options.Voice }

// Streaming reports false: the model takes a whole transcript per request, so a caller
// must send complete sentences rather than deltas.
func (t *TTS) Streaming() bool { return false }

// Performs reports false. The 3.8 models act vocal events written in angle brackets,
// not the square-bracketed directions this contract means, which would be read out.
func (t *TTS) Performs() bool { return false }

// Prompt reports nothing: there is no direction this voice is asked to act.
func (t *TTS) Prompt() string { return "" }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return OutputSampleRate }

// Client exposes the HTTP client so callers can call endpoints this provider does not wrap.
func (t *TTS) Client() *http.Client { return t.client }

// accumulate adds a delta to its utterance and reports whether the utterance is ready to
// be synthesised.
func (t *TTS) accumulate(request tts.Request) (*tts.Synthesis, string, bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, "", false, errors.New("gemini: session closed")
	}
	if !t.started {
		return nil, "", false, errors.New("gemini: not started")
	}
	// A partial with no id could not be matched to its continuation, so it is a caller
	// error rather than something to silently drop.
	if !request.Final && request.ID == "" {
		return nil, "", false, errors.New("gemini: a partial request needs an id")
	}

	current := t.pending[request.ID]
	if current == nil {
		current = &pending{tracker: tts.NewSynthesis(request.ID)}
		t.pending[request.ID] = current
	}
	current.text.WriteString(request.Text)

	if !request.Final {
		return current.tracker, "", false, nil
	}

	text := strings.TrimSpace(current.text.String())
	delete(t.pending, request.ID)
	if text == "" {
		return nil, "", false, errors.New("gemini: nothing to say")
	}

	current.tracker.AddText(text)
	return current.tracker, text, true, nil
}

// synthesize performs one request and streams its audio out.
func (t *TTS) synthesize(ctx context.Context, synthesis *tts.Synthesis, text string, request tts.Request) {
	defer func() {
		t.mu.Lock()
		delete(t.inFlight, synthesis.ID)
		t.mu.Unlock()
	}()

	response, err := t.post(ctx, text, request)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, true))
			return
		}
		t.fail(synthesis, err, "request")
		return
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		t.fail(synthesis, fmt.Errorf("http %d: %s", response.StatusCode, strings.TrimSpace(string(body))), "request")
		return
	}

	interrupted := t.stream(synthesis, response.Body)
	t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, interrupted))
}

// stream reads server-sent events from the response and emits their audio, reporting
// whether the utterance was cut short.
func (t *TTS) stream(synthesis *tts.Synthesis, body io.Reader) bool {
	reader := bufio.NewReader(body)
	var carry []byte

	for {
		line, err := reader.ReadBytes('\n')
		if payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:")); ok {
			var event streamEvent
			if decodeErr := json.Unmarshal(bytes.TrimSpace(payload), &event); decodeErr != nil {
				t.logger.Debug("undecodable event", "error", decodeErr)
			} else {
				var failure error
				if carry, failure = t.handleEvent(synthesis, event, carry); failure != nil {
					t.emitter.Send(tts.Error{
						Provider:    ProviderName,
						Model:       t.options.Model,
						SynthesisID: synthesis.ID,
						Err:         failure,
						Context:     "audio",
					})
					return true
				}
			}
		}

		if err != nil {
			if errors.Is(err, io.EOF) {
				return false
			}
			// A cancelled context is barge-in or teardown, not a provider failure.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return true
			}
			t.emitter.Send(tts.Error{
				Provider:    ProviderName,
				Model:       t.options.Model,
				SynthesisID: synthesis.ID,
				Err:         fmt.Errorf("gemini: %w", err),
				Context:     "audio",
			})
			return true
		}
	}
}

// handleEvent emits the audio in one event and returns the odd byte, if any, that has to
// wait for the next one: a chunk is not promised to end on a whole sample. An error, or a
// candidate that stopped for any reason but reaching the end, ends the utterance.
func (t *TTS) handleEvent(synthesis *tts.Synthesis, event streamEvent, carry []byte) ([]byte, error) {
	if event.Error != nil {
		return carry, fmt.Errorf("gemini: %s: %s", event.Error.Status, event.Error.Message)
	}
	for _, candidate := range event.Candidates {
		if candidate.Content != nil {
			for _, piece := range candidate.Content.Parts {
				if piece.InlineData == nil || !strings.HasPrefix(piece.InlineData.MimeType, "audio/") {
					continue
				}
				raw, err := base64.StdEncoding.DecodeString(piece.InlineData.Data)
				if err != nil {
					t.logger.Debug("undecodable audio", "error", err)
					continue
				}
				block := append(carry, raw...)
				carry = nil
				if len(block)%2 != 0 {
					carry = []byte{block[len(block)-1]}
					block = block[:len(block)-1]
				}
				if len(block) > 0 {
					t.emitter.Send(synthesis.Chunk(audio.FromBytes(block, OutputSampleRate, 1)))
				}
			}
		}
		if candidate.FinishReason != "" && candidate.FinishReason != "STOP" {
			return carry, fmt.Errorf("gemini: synthesis stopped: %s", candidate.FinishReason)
		}
	}
	return carry, nil
}

func (t *TTS) post(ctx context.Context, text string, request tts.Request) (*http.Response, error) {
	payload, err := json.Marshal(t.body(text, request))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	endpoint := strings.TrimSuffix(t.options.BaseURL, "/") + "/models/" +
		url.PathEscape(t.options.Model) + ":streamGenerateContent?alt=sse"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpRequest.Header.Set("x-goog-api-key", t.options.APIKey)
	httpRequest.Header.Set("Content-Type", "application/json")

	return t.client.Do(httpRequest)
}

// body is one utterance: the transcript, spoken in raw PCM, in the voice and language
// asked for.
func (t *TTS) body(text string, request tts.Request) generateRequest {
	body := generateRequest{
		Contents: []content{{Role: "user", Parts: []part{{Text: text}}}},
		GenerationConfig: generationConfig{
			ResponseModalities: []string{"AUDIO"},
			ResponseFormat: &responseFormat{Audio: audioResponseFormat{
				MimeType:   audioFormat,
				SampleRate: OutputSampleRate,
			}},
		},
	}
	voice, language := t.voiceFor(request), t.languageFor(request)
	if voice != "" || language != "" {
		body.GenerationConfig.SpeechConfig = &speechConfig{LanguageCode: language}
		if voice != "" {
			body.GenerationConfig.SpeechConfig.VoiceConfig = &voiceConfig{Voice: voice}
		}
	}
	return body
}

// voiceFor lets a request override the session's voice, which Gemini allows because each
// synthesis is its own request.
func (t *TTS) voiceFor(request tts.Request) string {
	if request.Voice != "" {
		return request.Voice
	}
	return t.options.Voice
}

// languageFor lets a request override the session's language, for the same reason.
func (t *TTS) languageFor(request tts.Request) string {
	if request.Language != "" {
		return request.Language
	}
	return t.options.Language
}

func (t *TTS) fail(synthesis *tts.Synthesis, err error, context string) {
	t.emitter.Send(tts.Error{
		Provider:    ProviderName,
		Model:       t.options.Model,
		SynthesisID: synthesis.ID,
		Err:         fmt.Errorf("gemini: %w", err),
		Context:     context,
	})
	t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, true))
}
