// Package fish implements the tts.TTS contract against Fish Audio's hosted API.
//
// Fish is a request/response API rather than a socket: one POST per utterance, with the
// audio streaming back in the response body. The provider therefore buffers partial text
// until the utterance is final, then streams the response out chunk by chunk so playback
// can still start before synthesis finishes.
package fish

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
const ProviderName = "fish"

// DefaultModel is Fish's current flagship, which is also what we self-host as s2pro.
const DefaultModel = "s2-pro"

// DefaultSampleRate is what the provider asks Fish to synthesise at.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production endpoint.
const defaultBaseURL = "https://api.fish.audio"

// chunkBytes is how much audio is emitted at a time: about 85 ms at 24 kHz, small enough
// that playback starts promptly and large enough not to flood the event channel.
const chunkBytes = 4096

// Options configures the provider. APIKey falls back to FISH_API_KEY and Voice to
// FISH_VOICE_ID.
type Options struct {
	APIKey string
	// Voice is a Fish reference id. Empty uses the model's own voice.
	Voice string
	Model string
	// SampleRate is the rate to synthesise at.
	SampleRate int
	// Latency is Fish's quality/latency trade-off. Empty leaves the server default.
	Latency string
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// Timeout bounds one synthesis, including reading the audio back.
	Timeout time.Duration
	// HTTPClient overrides the client, for a custom transport.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// synthesisRequest is the JSON body of POST /v1/tts.
type synthesisRequest struct {
	Text        string `json:"text"`
	ReferenceID string `json:"reference_id,omitempty"`
	Format      string `json:"format"`
	SampleRate  int    `json:"sample_rate"`
	Normalize   bool   `json:"normalize"`
	Latency     string `json:"latency,omitempty"`
}

// pending is an utterance being assembled from deltas. The tracker is created on the
// first delta so the reported latency covers the wait the caller actually experienced,
// not just the request this provider eventually makes.
type pending struct {
	tracker *tts.Synthesis
	text    strings.Builder
}

// TTS is a Fish Audio text-to-speech session.
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
		options.APIKey = os.Getenv("FISH_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("fish: api key is required (set FISH_API_KEY)")
	}
	if options.Voice == "" {
		options.Voice = os.Getenv("FISH_VOICE_ID")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.SampleRate == 0 {
		options.SampleRate = DefaultSampleRate
	}
	if options.SampleRate <= 0 {
		return nil, fmt.Errorf("fish: sample rate must be positive, got %d", options.SampleRate)
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
		return errors.New("fish: already started")
	}
	t.started = true
	t.ctx, t.cancel = context.WithCancel(context.WithoutCancel(ctx))

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	return nil
}

// Synthesize buffers text and, once the utterance is final, sends it to Fish. Audio is
// emitted as the response arrives.
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

// Interrupt cancels every synthesis in flight, which stops the response being read and
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

// Streaming reports false: Fish synthesises a whole utterance per request, so a caller
// must send complete sentences rather than deltas.
func (t *TTS) Streaming() bool { return false }

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
		return nil, "", false, errors.New("fish: session closed")
	}
	if !t.started {
		return nil, "", false, errors.New("fish: not started")
	}
	// A partial with no id could not be matched to its continuation, so it is a caller
	// error rather than something to silently drop.
	if !request.Final && request.ID == "" {
		return nil, "", false, errors.New("fish: a partial request needs an id")
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
		return nil, "", false, errors.New("fish: nothing to say")
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
	payload, err := json.Marshal(synthesisRequest{
		Text:        text,
		ReferenceID: t.voiceFor(request),
		Format:      "pcm",
		SampleRate:  t.options.SampleRate,
		Normalize:   true,
		Latency:     t.options.Latency,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	url := strings.TrimSuffix(t.options.BaseURL, "/") + "/v1/tts"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpRequest.Header.Set("Authorization", "Bearer "+t.options.APIKey)
	httpRequest.Header.Set("Content-Type", "application/json")
	// The model is selected by header rather than in the body.
	httpRequest.Header.Set("model", t.options.Model)

	return t.client.Do(httpRequest)
}

// voiceFor lets a request override the session's voice, which Fish allows because each
// synthesis is its own request.
func (t *TTS) voiceFor(request tts.Request) string {
	if request.Voice != "" {
		return request.Voice
	}
	return t.options.Voice
}

func (t *TTS) fail(synthesis *tts.Synthesis, err error, context string) {
	t.emitter.Send(tts.Error{
		Provider:    ProviderName,
		Model:       t.options.Model,
		SynthesisID: synthesis.ID,
		Err:         fmt.Errorf("fish: %w", err),
		Context:     context,
	})
	t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, true))
}
