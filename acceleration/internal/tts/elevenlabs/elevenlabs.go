// Package elevenlabs implements the tts.TTS contract against ElevenLabs' streaming
// WebSocket.
//
// It uses the multi-stream endpoint, where each utterance is a server-side "context". That
// maps directly onto tts.Request.ID: opening a context starts a synthesis, closing one
// ends it, and closing it early is barge-in. Audio arrives base64-encoded in JSON frames
// as it is generated, so the first sound reaches the listener long before the sentence is
// finished.
package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "elevenlabs"

// DefaultModel is the low-latency model, which is what a live conversation wants.
const DefaultModel = "eleven_flash_v2_5"

// DefaultVoiceID is Arnold from the public voice library, so the provider works without a
// voice having been picked.
const DefaultVoiceID = "VR6AewLTigWG4xSOukaG"

// DefaultSampleRate is the highest rate the PCM output format offers without paying for
// bandwidth nobody hears.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production endpoint.
const defaultBaseURL = "wss://api.elevenlabs.io"

// supportedSampleRates are the rates the pcm_* output formats cover.
var supportedSampleRates = []int{8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000}

// multilingualModels accept a language_code. Passing one to another model is rejected
// upstream, so it is only sent when it applies.
var multilingualModels = []string{"eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"}

// Options configures the provider. APIKey falls back to ELEVENLABS_API_KEY and VoiceID to
// ELEVENLABS_VOICE_ID.
type Options struct {
	APIKey string
	// VoiceID is the speaker. The connection is bound to it, so one session is one voice.
	VoiceID string
	Model   string
	// Language is an ISO code. It is only sent for models that accept one.
	Language string
	// SampleRate is the rate to synthesise at, one of supportedSampleRates.
	SampleRate int
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	// CloseTimeout bounds how long Close waits for the server's closing handshake.
	CloseTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a frame sent to ElevenLabs. Fields are omitted when unset because the
// server distinguishes an absent field from a zero one.
type clientMessage struct {
	Text          string          `json:"text,omitempty"`
	ContextID     string          `json:"context_id,omitempty"`
	VoiceSettings *voiceSettings  `json:"voice_settings,omitempty"`
	Flush         bool            `json:"flush,omitempty"`
	CloseContext  bool            `json:"close_context,omitempty"`
	CloseSocket   bool            `json:"close_socket,omitempty"`
	Generation    *generationConf `json:"generation_config,omitempty"`
}

type voiceSettings struct {
	Stability       float64 `json:"stability"`
	SimilarityBoost float64 `json:"similarity_boost"`
}

// generationConf controls how much text the model waits for before generating. The first
// threshold is deliberately small so the first chunk comes back fast.
type generationConf struct {
	ChunkLengthSchedule []int `json:"chunk_length_schedule"`
}

// serverMessage is a frame sent by ElevenLabs.
type serverMessage struct {
	// Audio is base64-encoded PCM in the requested output format.
	Audio     string `json:"audio"`
	ContextID string `json:"contextId"`
	IsFinal   bool   `json:"isFinal"`
	Error     string `json:"error"`
	Message   string `json:"message"`
}

// utterance is one synthesis in flight.
type utterance struct {
	tracker *tts.Synthesis
	// interrupted stops audio still in flight from being forwarded after barge-in.
	interrupted bool
}

// TTS is a streaming ElevenLabs text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// socketClosed is closed when the server acknowledges close_socket.
	socketClosed     chan struct{}
	socketClosedOnce sync.Once

	mu       sync.Mutex
	active   map[string]*utterance
	started  bool
	shutdown bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("ELEVENLABS_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("elevenlabs: api key is required (set ELEVENLABS_API_KEY)")
	}
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("ELEVENLABS_VOICE_ID")
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
	if !slices.Contains(supportedSampleRates, options.SampleRate) {
		return nil, fmt.Errorf("elevenlabs: sample rate %d is not one of %v",
			options.SampleRate, supportedSampleRates)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 15 * time.Second
	}
	if options.CloseTimeout == 0 {
		options.CloseTimeout = 2 * time.Second
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &TTS{
		options:      options,
		logger:       logger.With("provider", ProviderName, "model", options.Model),
		emitter:      tts.NewEmitter(64),
		socketClosed: make(chan struct{}),
		active:       map[string]*utterance{},
	}, nil
}

// Start dials the WebSocket. There is no handshake to wait for: the connection is ready
// for text as soon as it is open.
func (t *TTS) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return errors.New("elevenlabs: already started")
	}
	t.started = true
	t.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: t.options.HandshakeTimeout}
	header := http.Header{"xi-api-key": []string{t.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, t.url(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("elevenlabs: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("elevenlabs: dial: %w", err)
	}
	t.conn = conn

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	go t.readLoop()
	return nil
}

// Synthesize sends text upstream. Several requests sharing an ID stream one sentence, and
// the one with Final set closes it so the tail of the audio is generated immediately.
func (t *TTS) Synthesize(request tts.Request) error {
	if request.Voice != "" && request.Voice != t.options.VoiceID {
		return fmt.Errorf(
			"elevenlabs: the connection is bound to voice %s, open a new session for %s",
			t.options.VoiceID, request.Voice)
	}

	current, opened, err := t.utteranceFor(request.ID)
	if err != nil {
		return err
	}

	if opened {
		if err := t.openContext(current.tracker.ID); err != nil {
			return err
		}
		t.emitter.Send(tts.SynthesisStarted{
			SynthesisID: current.tracker.ID,
			Provider:    ProviderName,
			Model:       t.options.Model,
			Voice:       t.options.VoiceID,
			At:          time.Now(),
		})
	}

	if request.Text != "" {
		current.tracker.AddText(request.Text)
		// The trailing space is what ElevenLabs uses to tell one word from the next
		// across deltas.
		err := t.send(clientMessage{Text: request.Text + " ", ContextID: current.tracker.ID})
		if err != nil {
			return fmt.Errorf("elevenlabs: send text: %w", err)
		}
	}

	if request.Final {
		// Closing the context flushes the remaining audio and makes the server report
		// the synthesis as final straight after it.
		if err := t.send(clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			return fmt.Errorf("elevenlabs: close context: %w", err)
		}
	}
	return nil
}

// Interrupt closes every context in flight, which stops the server generating and stops
// audio already on the wire from being forwarded.
func (t *TTS) Interrupt() error {
	t.mu.Lock()
	interrupting := make([]*utterance, 0, len(t.active))
	for _, current := range t.active {
		if !current.interrupted {
			current.interrupted = true
			interrupting = append(interrupting, current)
		}
	}
	t.mu.Unlock()

	var failures []error
	for _, current := range interrupting {
		if err := t.send(clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			failures = append(failures, err)
		}
		t.complete(current.tracker.ID)
	}

	if len(failures) > 0 {
		return fmt.Errorf("elevenlabs: interrupt: %w", errors.Join(failures...))
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close asks the server to close, then tears down the connection. Anything still in flight
// is reported as interrupted so no work goes unaccounted for.
func (t *TTS) Close() error {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil
	}
	t.shutdown = true
	conn := t.conn
	t.mu.Unlock()

	t.settleOutstanding()

	if conn != nil {
		// The protocol tears down in two steps: we ask the server to close, then it does.
		// Waiting for its close avoids ending the session on an abnormal closure.
		if err := t.send(clientMessage{CloseSocket: true}); err != nil {
			t.logger.Debug("close_socket not delivered", "error", err)
		} else {
			select {
			case <-t.socketClosed:
			case <-time.After(t.options.CloseTimeout):
				t.logger.Debug("server did not close the socket in time")
			}
		}
		conn.Close()
	}

	t.emitter.Close()
	return nil
}

// Provider implements tts.TTS.
func (t *TTS) Provider() string { return ProviderName }

// Model implements tts.TTS.
func (t *TTS) Model() string { return t.options.Model }

// Streaming reports true: the model generates from partial text.
func (t *TTS) Streaming() bool { return true }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.options.SampleRate }

// Client exposes the underlying WebSocket so callers can use the API directly.
func (t *TTS) Client() *websocket.Conn { return t.conn }

// url builds the multi-stream endpoint for the configured voice and model.
func (t *TTS) url() string {
	query := []string{
		"model_id=" + t.options.Model,
		"output_format=pcm_" + strconv.Itoa(t.options.SampleRate),
		// auto_mode lets the server decide when it has enough text, which is what keeps
		// time-to-first-audio low without the caller tuning anything.
		"auto_mode=true",
	}
	if t.options.Language != "" && slices.Contains(multilingualModels, t.options.Model) {
		query = append(query, "language_code="+strings.ToLower(t.options.Language))
	}

	return fmt.Sprintf("%s/v1/text-to-speech/%s/multi-stream-input?%s",
		strings.TrimSuffix(t.options.BaseURL, "/"), t.options.VoiceID, strings.Join(query, "&"))
}

// utteranceFor returns the tracker for a request, reporting whether it had to be created.
func (t *TTS) utteranceFor(id string) (*utterance, bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, false, errors.New("elevenlabs: session closed")
	}
	if !t.started || t.conn == nil {
		return nil, false, errors.New("elevenlabs: not started")
	}

	if id != "" {
		if existing, ok := t.active[id]; ok {
			return existing, false, nil
		}
	}

	current := &utterance{tracker: tts.NewSynthesis(id)}
	t.active[current.tracker.ID] = current
	return current, true, nil
}

// openContext initialises a server-side context for one utterance.
func (t *TTS) openContext(id string) error {
	// A single space is the documented way to open a context without saying anything yet.
	message := clientMessage{
		Text:          " ",
		ContextID:     id,
		VoiceSettings: &voiceSettings{Stability: 0.5, SimilarityBoost: 0.8},
		// A short first threshold trades a little prosody for audio that starts sooner.
		Generation: &generationConf{ChunkLengthSchedule: []int{50, 120, 160, 290}},
	}
	if err := t.send(message); err != nil {
		return fmt.Errorf("elevenlabs: open context: %w", err)
	}
	return nil
}

func (t *TTS) send(message clientMessage) error {
	payload, err := json.Marshal(message)
	if err != nil {
		return err
	}

	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	if t.conn == nil {
		return errors.New("not connected")
	}
	return t.conn.WriteMessage(websocket.TextMessage, payload)
}

// readLoop translates server frames into events until the connection ends.
func (t *TTS) readLoop() {
	for {
		_, raw, err := t.conn.ReadMessage()
		if err != nil {
			t.socketClosedOnce.Do(func() { close(t.socketClosed) })
			t.handleReadError(err)
			// The connection is gone, so nothing in flight will ever finish on its own.
			t.settleOutstanding()
			return
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			t.logger.Debug("undecodable frame", "error", err)
			continue
		}
		t.handleMessage(message)
	}
}

func (t *TTS) handleReadError(err error) {
	t.mu.Lock()
	shutdown := t.shutdown
	t.mu.Unlock()
	if shutdown {
		return
	}

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		t.emitter.Send(tts.Disconnected{
			Provider: ProviderName,
			Model:    t.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		return
	}
	t.emitter.Send(tts.Error{
		Provider: ProviderName,
		Model:    t.options.Model,
		Err:      err,
		Context:  "read",
		Fatal:    true,
	})
}

func (t *TTS) handleMessage(message serverMessage) {
	if failure := failureOf(message); failure != "" {
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: message.ContextID,
			Err:         errors.New(failure),
			Context:     "server",
			Fatal:       false,
		})
		return
	}

	if message.Audio != "" {
		t.handleAudio(message)
	}
	if message.IsFinal {
		t.complete(message.ContextID)
	}
}

func (t *TTS) handleAudio(message serverMessage) {
	raw, err := base64.StdEncoding.DecodeString(message.Audio)
	if err != nil {
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: message.ContextID,
			Err:         fmt.Errorf("decode audio: %w", err),
			Context:     "audio",
		})
		return
	}

	current, ok := t.lookup(message.ContextID)
	if !ok {
		// The utterance was interrupted or already settled, so this audio is stale.
		return
	}
	t.emitter.Send(current.tracker.Chunk(audio.FromBytes(raw, t.options.SampleRate, 1)))
}

// complete settles an utterance and emits its summary. It is a no-op for one that has
// already been settled, so a late final frame after barge-in is harmless.
func (t *TTS) complete(id string) {
	t.mu.Lock()
	current, ok := t.active[id]
	delete(t.active, id)
	t.mu.Unlock()

	if !ok {
		return
	}
	t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, current.interrupted))
}

// settleOutstanding reports every utterance still in flight as interrupted. A connection
// that dies, or a session that is torn down, must not leave work unaccounted for.
func (t *TTS) settleOutstanding() {
	t.mu.Lock()
	outstanding := make([]*utterance, 0, len(t.active))
	for _, current := range t.active {
		outstanding = append(outstanding, current)
	}
	clear(t.active)
	t.mu.Unlock()

	for _, current := range outstanding {
		t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, true))
	}
}

// lookup returns an utterance that is still accepting audio.
func (t *TTS) lookup(id string) (*utterance, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.active[id]
	if !ok || current.interrupted {
		return nil, false
	}
	return current, true
}

// failureOf returns the error text of a frame, or empty when it is not a failure.
func failureOf(message serverMessage) string {
	if message.Error != "" {
		return message.Error
	}
	// A frame carrying only a message and no audio is how a rejection arrives.
	if message.Message != "" && message.Audio == "" {
		return message.Message
	}
	return ""
}
