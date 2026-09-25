// Package speechify implements the tts.TTS contract against Speechify's hosted API.
//
// Speechify has no socket: POST /v1/audio/stream takes one whole utterance and streams the
// audio back over a chunked response. The provider therefore buffers partial text until the
// utterance is final, then streams the response out chunk by chunk so playback starts
// before synthesis finishes.
package speechify

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "speechify"

// DefaultModel is Simba 3.2, Speechify's streaming-native English model.
const DefaultModel = "simba-3.2"

// DefaultVoiceID is the voice Speechify's own examples use. The API requires a voice.
const DefaultVoiceID = "geffen_32"

// DefaultSampleRate is what the provider asks Speechify to synthesise at. It is also what
// the Simba 3 models produce natively: a workspace pinned before API version 2026-09-30
// answers pcm_16000 with 24 kHz samples, so another rate is only safe on a newer pin.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production endpoint.
const defaultBaseURL = "https://api.speechify.ai"

// caller identifies this integration to Speechify through the Speechify-Caller header.
const caller = "vision-agents"

// chunkBytes is how much audio is emitted at a time: about 85 ms at 24 kHz, small enough
// that playback starts promptly and large enough not to flood the event channel.
const chunkBytes = 4096

// sampleRates are the rates Speechify serves raw PCM at.
var sampleRates = map[int]bool{8_000: true, 16_000: true, 22_050: true, 24_000: true, 44_100: true, 48_000: true}

// Options configures the provider. APIKey falls back to SPEECHIFY_API_KEY.
type Options struct {
	APIKey string
	// VoiceID is a Speechify voice id. Empty uses DefaultVoiceID.
	VoiceID string
	Model   string
	// SampleRate is the rate to synthesise at, one of the pcm_* rates Speechify serves.
	SampleRate int
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// Timeout bounds one synthesis, including reading the audio back.
	Timeout time.Duration
	// HTTPClient overrides the client, for a custom transport.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// streamRequest is the JSON body of POST /v1/audio/stream.
type streamRequest struct {
	Input        string `json:"input"`
	VoiceID      string `json:"voice_id"`
	Model        string `json:"model"`
	OutputFormat string `json:"output_format"`
}

// pending is an utterance being assembled from deltas. The tracker is created on the
// first delta so the reported latency covers the wait the caller actually experienced,
// not just the request this provider eventually makes.
type pending struct {
	tracker *tts.Synthesis
	text    strings.Builder
}

// TTS is a Speechify text-to-speech session.
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
		options.APIKey = os.Getenv("SPEECHIFY_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("speechify: api key is required (set SPEECHIFY_API_KEY)")
	}
	if options.VoiceID == "" {
		options.VoiceID = DefaultVoiceID
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.SampleRate == 0 {
		options.SampleRate = DefaultSampleRate
	}
	if !sampleRates[options.SampleRate] {
		return nil, fmt.Errorf("speechify: no pcm format at %d Hz", options.SampleRate)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
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
		return errors.New("speechify: already started")
	}
	t.started = true
	t.ctx, t.cancel = context.WithCancel(context.WithoutCancel(ctx))

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	return nil
}

// Synthesize buffers text and, once the utterance is final, sends it to Speechify. Audio
// is emitted as the response arrives.
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

// Interrupt cancels every synthesis in flight, which drops the response mid-stream and
// stops any further audio reaching the caller.
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

// Voice is the session's voice.
func (t *TTS) Voice() string { return t.options.VoiceID }

// Streaming reports false: Speechify synthesises a whole utterance per request, so a
// caller must send complete sentences rather than deltas.
func (t *TTS) Streaming() bool { return false }

// Performs reports false: emotion is SSML markup, not a bracketed direction.
func (t *TTS) Performs() bool { return false }

// Prompt reports nothing: there is no direction this voice would act.
func (t *TTS) Prompt() string { return "" }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.options.SampleRate }

// Client exposes the HTTP client so callers can call endpoints this provider does not wrap.
func (t *TTS) Client() *http.Client { return t.client }

// accumulate adds a delta to its utterance and reports whether the utterance is ready to
// be synthesised.
func (t *TTS) accumulate(request tts.Request) (*tts.Synthesis, string, bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, "", false, errors.New("speechify: session closed")
	}
	if !t.started {
		return nil, "", false, errors.New("speechify: not started")
	}
	// A partial with no id could not be matched to its continuation, so it is a caller
	// error rather than something to silently drop.
	if !request.Final && request.ID == "" {
		return nil, "", false, errors.New("speechify: a partial request needs an id")
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
		return nil, "", false, errors.New("speechify: nothing to say")
	}

	current.tracker.AddText(text)
	return current.tracker, text, true, nil
}

// synthesize performs one request and streams its audio out.
func (t *TTS) synthesize(
	ctx context.Context,
	synthesis *tts.Synthesis,
	text string,
	request tts.Request,
) {
	defer func() {
		t.mu.Lock()
		delete(t.inFlight, synthesis.ID)
		t.mu.Unlock()
	}()

	response, err := t.post(ctx, text, request)
	if err != nil {
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

// stream reads PCM from the response and emits it in chunks, reporting whether it was cut
// short. A read can end mid-sample, so a stray odd byte is carried to the next chunk.
// Speechify aborts the transfer when its content policy refuses text already being
// spoken, which arrives here as a read error rather than a status.
func (t *TTS) stream(synthesis *tts.Synthesis, body io.Reader) bool {
	buffer := make([]byte, chunkBytes)
	var carry []byte

	for {
		read, err := body.Read(buffer)
		if read > 0 {
			block := append(carry, buffer[:read]...)
			if odd := len(block) % 2; odd != 0 {
				carry = append([]byte(nil), block[len(block)-1:]...)
				block = block[:len(block)-1]
			} else {
				carry = nil
			}
			if len(block) > 0 {
				t.emitter.Send(synthesis.Chunk(audio.FromBytes(block, t.options.SampleRate, 1)))
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
				Err:         err,
				Context:     "audio",
			})
			return true
		}
	}
}

func (t *TTS) post(ctx context.Context, text string, request tts.Request) (*http.Response, error) {
	payload, err := json.Marshal(streamRequest{
		Input:        text,
		VoiceID:      t.voiceFor(request),
		Model:        t.options.Model,
		OutputFormat: fmt.Sprintf("pcm_%d", t.options.SampleRate),
	})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	url := strings.TrimSuffix(t.options.BaseURL, "/") + "/v1/audio/stream"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpRequest.Header.Set("Authorization", "Bearer "+t.options.APIKey)
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "audio/pcm")
	httpRequest.Header.Set("Speechify-Caller", caller)

	return t.client.Do(httpRequest)
}

// voiceFor lets a request override the session's voice, which Speechify allows because
// each synthesis is its own request.
func (t *TTS) voiceFor(request tts.Request) string {
	if request.Voice != "" {
		return request.Voice
	}
	return t.options.VoiceID
}

func (t *TTS) fail(synthesis *tts.Synthesis, err error, context string) {
	t.emitter.Send(tts.Error{
		Provider:    ProviderName,
		Model:       t.options.Model,
		SynthesisID: synthesis.ID,
		Err:         fmt.Errorf("speechify: %w", err),
		Context:     context,
	})
	t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, true))
}
