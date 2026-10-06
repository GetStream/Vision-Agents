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
	"net"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "elevenlabs"

// DefaultModel is the low-latency model, which is what a live conversation wants.
const DefaultModel = "eleven_flash_v2_5"

// DefaultVoiceID is a voice from the public voice library, so the provider works without a
// voice having been picked. It is deliberately not one of the older premade voices: those
// are legacy, are absent from accounts made since, and the v3 models reject one with
// "Invalid argument received." at the point they try to generate rather than on connect.
const DefaultVoiceID = "NOpBlnGInO9m6vDvFkFC"

// DefaultSampleRate is the highest rate the PCM output format offers without paying for
// bandwidth nobody hears.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production endpoint.
const defaultBaseURL = "wss://api.elevenlabs.io"

// minSpeed and maxSpeed bound voice_settings.speed. A speed outside them is refused rather
// than clamped, since a caller asking for 1.5 and hearing 1.2 cannot tell it was not honoured.
const (
	minSpeed = 0.7
	maxSpeed = 1.2
)

// supportedSampleRates are the rates the pcm_* output formats cover.
var supportedSampleRates = []int{8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000}

// multilingualModels accept a language_code. Passing one to another model is rejected
// upstream, so it is only sent when it applies.
var multilingualModels = []string{"eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"}

// dialogueModelPrefixes mark the v3 and v4 families, the only ones that act audio tags.
// Everything before them reads a bracket out as the word inside it.
var dialogueModelPrefixes = []string{"eleven_v3", "eleven_v4"}

// Performs reports whether a model acts audio tags rather than reading them out.
func Performs(model string) bool {
	for _, prefix := range dialogueModelPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// AudioTagPrompt is what a model writing for a v3 voice is told about directing it. The
// tags named are examples rather than a closed set: the voice takes any bracketed
// direction, so listing a few is enough to show the shape of one.
const AudioTagPrompt = "Your words are spoken by a voice that acts stage directions. " +
	"Write a direction in square brackets just before the words it applies to, and it " +
	"will be performed rather than read out: [laughs], [sighs], [whispers], [excited], " +
	"[sarcastic], [short pause]. Any direction works, not only these. Use them where a " +
	"person would naturally have done that, and no more than once or twice a turn: a " +
	"reply full of directions sounds like a performance rather than a conversation."

// Options configures the provider. APIKey falls back to ELEVENLABS_API_KEY and VoiceID to
// ELEVENLABS_VOICE_ID.
type Options struct {
	APIKey string
	// VoiceID is the speaker. The connection is bound to it, so one session is one voice.
	VoiceID string
	Model   string
	// Language is an ISO code. It is only sent for models that accept one.
	Language string
	// Speed is the rate of delivery, 1 being the voice's own and zero leaving it there.
	// ElevenLabs accepts 0.7 to 1.2, and only on this socket: the dialogue models take none.
	Speed float64
	// SampleRate is the rate to synthesise at, one of supportedSampleRates.
	SampleRate int
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	// CloseTimeout bounds how long Close waits for in-flight events and reader shutdown.
	// If the event consumer remains blocked past this period, Close closes the event
	// stream so sends can unwind; undelivered terminal events may then be dropped.
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
	Speed           float64 `json:"speed,omitempty"`
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
	tracker   *tts.Synthesis
	socket    *socket
	ready     chan struct{}
	opened    chan struct{}
	openOnce  sync.Once
	requestMu sync.Mutex
	eventMu   sync.Mutex
	// interrupted stops audio still in flight from being forwarded after barge-in.
	interrupted bool
}

// socket is one physical WebSocket generation. A replacement connection always gets a
// fresh value so an old reader can never clear or settle work belonging to the new one.
type socket struct {
	conn      *websocket.Conn
	writeMu   sync.Mutex
	ready     chan struct{}
	readyOnce sync.Once
	done      chan struct{}
	closeOnce sync.Once
}

// dialAttempt owns the raw transport only while a WebSocket handshake is in flight.
// Its close callback is bound to this instance, so a delayed cancellation can never
// close a later reconnect's transport.
type dialAttempt struct {
	mu     sync.Mutex
	conn   net.Conn
	closed bool
}

// mergeDialContext lets the caller bound the initial connection while keeping Close
// authoritative over that same attempt, including DNS/TCP work before a raw socket exists.
func mergeDialContext(caller, lifetime context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(caller)
	stopLifetimeCancel := context.AfterFunc(lifetime, cancel)
	return ctx, func() {
		stopLifetimeCancel()
		cancel()
	}
}

func (a *dialAttempt) attach(conn net.Conn) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return false
	}
	a.conn = conn
	return true
}

func (a *dialAttempt) close() {
	a.mu.Lock()
	a.closed = true
	conn := a.conn
	a.conn = nil
	a.mu.Unlock()
	if conn != nil {
		_ = conn.Close()
	}
}

// finish transfers a successful connection to the WebSocket. It returns false when a
// cancellation callback already closed the attempt.
func (a *dialAttempt) finish() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return false
	}
	a.closed = true
	a.conn = nil
	return true
}

// TTS is a streaming ElevenLabs text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	// dialMu serializes socket replacement with registration of work on that socket.
	dialMu sync.Mutex
	conn   *socket
	ctx    context.Context
	cancel context.CancelFunc

	dialConnMu  sync.Mutex
	dialAttempt *dialAttempt
	readers     sync.WaitGroup

	mu sync.Mutex
	// terminalIDs prevents late deltas/finals for a failed or interrupted ID from opening
	// another upstream context after a reconnect.
	terminalIDs map[string]struct{}
	active      map[string]*utterance
	started     bool
	shutdown    bool
}

// normalize fills in the defaults and reports what cannot be defaulted. Both endpoints
// take the same options, so they settle them the same way and differ only in the model
// they fall back to.
func normalize(options Options, fallbackModel string) (Options, *slog.Logger, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("ELEVENLABS_API_KEY")
	}
	if options.APIKey == "" {
		return options, nil, stack.Wrap(errors.New("elevenlabs: api key is required (set ELEVENLABS_API_KEY)"))
	}
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("ELEVENLABS_VOICE_ID")
	}
	if options.VoiceID == "" {
		options.VoiceID = DefaultVoiceID
	}
	if options.Model == "" {
		options.Model = fallbackModel
	}
	if options.SampleRate == 0 {
		options.SampleRate = DefaultSampleRate
	}
	if !slices.Contains(supportedSampleRates, options.SampleRate) {
		return options, nil, stack.Wrap(fmt.Errorf("elevenlabs: sample rate %d is not one of %v",
			options.SampleRate, supportedSampleRates))
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
	return options, logger, nil
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	options, logger, err := normalize(options, DefaultModel)
	if err != nil {
		return nil, err
	}
	// The v3 family is not served here at all, so a caller that asked for one would get a
	// connection the server refuses rather than a voice.
	if Performs(options.Model) {
		return nil, stack.Wrap(fmt.Errorf(
			"elevenlabs: %s is a dialogue model, open it with NewDialogue", options.Model))
	}
	if options.Speed != 0 && (options.Speed < minSpeed || options.Speed > maxSpeed) {
		return nil, stack.Wrap(fmt.Errorf("elevenlabs: speed %g is outside %g to %g", options.Speed, minSpeed, maxSpeed))
	}

	return &TTS{
		options:     options,
		logger:      logger.With("provider", ProviderName, "model", options.Model),
		emitter:     tts.NewEmitter(64),
		terminalIDs: map[string]struct{}{},
		active:      map[string]*utterance{},
	}, nil
}

// Start dials the WebSocket using ctx for the initial handshake. After the handshake,
// reconnects use a session lifetime that ends at Close, so a caller's setup deadline does
// not disable later recovery.
func (t *TTS) Start(ctx context.Context) error {
	if ctx == nil {
		return stack.Wrap(errors.New("elevenlabs: start context is required"))
	}
	t.dialMu.Lock()
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		t.dialMu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	if t.started {
		t.mu.Unlock()
		t.dialMu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: already started"))
	}
	t.started = true
	// The successful Start context bounds only this initial handshake. Reconnects
	// later in the session use a separate lifetime canceled by Close.
	t.ctx, t.cancel = context.WithCancel(context.Background())
	t.mu.Unlock()

	sock, fresh, err := t.connectLocked(ctx)
	if err != nil {
		t.mu.Lock()
		t.started = false
		cancel := t.cancel
		t.cancel = nil
		t.ctx = nil
		t.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		t.dialMu.Unlock()
		return err
	}
	t.dialMu.Unlock()
	if fresh {
		t.announce(sock)
	} else {
		<-sock.ready
	}
	return nil
}

// Synthesize sends text upstream. Several requests sharing an ID stream one sentence, and
// the one with Final set closes it so the tail of the audio is generated immediately.
func (t *TTS) Synthesize(request tts.Request) error {
	if request.Voice != "" && request.Voice != t.options.VoiceID {
		return stack.Wrap(fmt.Errorf(
			"elevenlabs: the connection is bound to voice %s, open a new session for %s",
			t.options.VoiceID, request.Voice))
	}

	// A final/error ID is never reopened. In particular, late deltas after a socket loss
	// must not be replayed into a replacement connection as a new utterance.
	t.dialMu.Lock()
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		t.dialMu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	if !t.started {
		t.mu.Unlock()
		t.dialMu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: not started"))
	}
	if request.ID != "" {
		if _, terminal := t.terminalIDs[request.ID]; terminal {
			t.mu.Unlock()
			t.dialMu.Unlock()
			return nil
		}
	}
	current := t.active[request.ID]
	t.mu.Unlock()

	opened := current == nil
	var sock *socket
	fresh := false
	if opened {
		var err error
		sock, fresh, err = t.connectLocked(nil)
		if err != nil {
			t.mu.Lock()
			stopping := t.shutdown || t.ctx == nil || t.ctx.Err() != nil
			t.mu.Unlock()
			if stopping {
				t.dialMu.Unlock()
				return err
			}
			tracker := tts.NewSynthesis(request.ID)
			if request.Text != "" {
				tracker.AddText(request.Text)
			}
			t.mu.Lock()
			t.terminalIDs[tracker.ID] = struct{}{}
			t.mu.Unlock()
			t.dialMu.Unlock()
			t.emitRejected(tracker, err, "connect")
			return err
		}
		current = &utterance{
			tracker: tts.NewSynthesis(request.ID),
			socket:  sock,
			ready:   make(chan struct{}),
			opened:  make(chan struct{}),
		}
		t.mu.Lock()
		if t.shutdown || t.conn != sock {
			t.mu.Unlock()
			t.dialMu.Unlock()
			return stack.Wrap(errors.New("elevenlabs: session closed"))
		}
		if request.ID != "" {
			if _, terminal := t.terminalIDs[request.ID]; terminal {
				t.mu.Unlock()
				t.dialMu.Unlock()
				return nil
			}
		}
		t.active[current.tracker.ID] = current
		t.mu.Unlock()
	} else {
		sock = current.socket
	}
	t.dialMu.Unlock()
	if opened {
		current.requestMu.Lock()
		defer current.requestMu.Unlock()
		defer current.openOnce.Do(func() { close(current.opened) })
	} else {
		<-current.opened
		current.requestMu.Lock()
		defer current.requestMu.Unlock()
		if !t.isRegistered(sock, current.tracker.ID, current) {
			return nil
		}
	}

	if fresh {
		t.announce(sock)
	} else {
		<-sock.ready
	}
	if !opened {
		<-current.ready
	}
	if opened {
		// Track text before sending so a write failure still produces an accurate terminal
		// synthesis record. The Started event must precede a possible Error/Complete pair.
		t.emitter.Send(tts.SynthesisStarted{
			SynthesisID: current.tracker.ID,
			Provider:    ProviderName,
			Model:       t.options.Model,
			Voice:       t.options.VoiceID,
			At:          time.Now(),
		})
		close(current.ready)
	}
	if request.Text != "" {
		current.tracker.AddText(request.Text)
	}

	if opened {
		if err := t.openContext(sock, current.tracker.ID); err != nil {
			t.connectionEnded(sock, err, "write")
			return stack.Wrap(fmt.Errorf("elevenlabs: open context: %w", err))
		}
	}
	if !t.isRegistered(sock, current.tracker.ID, current) {
		return nil
	}

	if request.Text != "" {
		// The trailing space is what ElevenLabs uses to tell one word from the next
		// across deltas.
		err := t.send(sock, clientMessage{Text: request.Text + " ", ContextID: current.tracker.ID})
		if err != nil {
			t.connectionEnded(sock, err, "write")
			return stack.Wrap(fmt.Errorf("elevenlabs: send text: %w", err))
		}
	}

	if request.Final {
		// Closing the context flushes the remaining audio and makes the server report
		// the synthesis as final straight after it.
		if err := t.send(sock, clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			t.connectionEnded(sock, err, "write")
			return stack.Wrap(fmt.Errorf("elevenlabs: close context: %w", err))
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
		<-current.opened
		current.requestMu.Lock()
		if !t.isPresent(current.socket, current.tracker.ID, current) {
			current.requestMu.Unlock()
			continue
		}
		if err := t.send(current.socket, clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			t.connectionEnded(current.socket, err, "write")
			current.requestMu.Unlock()
			failures = append(failures, err)
			continue
		}
		t.complete(current.socket, current.tracker.ID)
		current.requestMu.Unlock()
	}

	if len(failures) > 0 {
		return stack.Wrap(fmt.Errorf("elevenlabs: interrupt: %w", errors.Join(failures...)))
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close cancels pending dials and tears down the connection. Outstanding work normally
// receives an interrupted completion. If the event consumer stays blocked through
// CloseTimeout, Close closes the event stream to release sends; terminal events that could
// not be delivered may be dropped.
func (t *TTS) Close() error {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil
	}
	t.shutdown = true
	cancel := t.cancel
	t.mu.Unlock()
	// Start the shutdown bound before waiting on dial, event, or reader locks. Closing
	// the emitter cancels any Send blocked by a consumer that stopped draining Events.
	watchdog := time.AfterFunc(t.options.CloseTimeout, func() {
		t.logger.Debug("closing event stream after shutdown timeout")
		t.emitter.Close()
	})
	if cancel != nil {
		// Cancel before waiting for dialMu: DialContext must be able to unwind promptly.
		cancel()
	}
	t.dialConnMu.Lock()
	attempt := t.dialAttempt
	t.dialAttempt = nil
	t.dialConnMu.Unlock()
	if attempt != nil {
		// Gorilla's handshake read is bounded by HandshakeTimeout but does not react to
		// context cancellation after the TCP dial. Closing the in-flight transport makes
		// Close cancel a stalled upgrade instead of waiting for that timeout.
		attempt.close()
	}

	t.dialMu.Lock()
	t.mu.Lock()
	sock := t.conn
	t.conn = nil
	outstanding := make([]*utterance, 0, len(t.active))
	for id, current := range t.active {
		current.interrupted = true
		t.terminalIDs[id] = struct{}{}
		outstanding = append(outstanding, current)
	}
	clear(t.active)
	t.mu.Unlock()
	t.dialMu.Unlock()

	if sock != nil {
		sock.signalReady()
		sock.closeConn()
	}
	for _, current := range outstanding {
		<-current.ready
		current.eventMu.Lock()
		t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, true))
		current.eventMu.Unlock()
	}
	// dialMu is held during every reader Add, so after detaching the connection no
	// new readers can be added and this Wait is safe. The watchdog above releases
	// blocked event sends before this join if the consumer is stalled.
	t.readers.Wait()
	watchdog.Stop()
	t.emitter.Close()
	return nil
}

// Provider implements tts.TTS.
func (t *TTS) Provider() string { return ProviderName }

// Model implements tts.TTS.
func (t *TTS) Model() string { return t.options.Model }

// Voice is the voice id the connection is bound to.
func (t *TTS) Voice() string { return t.options.VoiceID }

// Streaming reports true: the model generates from partial text.
func (t *TTS) Streaming() bool { return true }

// Performs reports false: only the v3 family acts audio tags, and those are served by
// Dialogue rather than here.
func (t *TTS) Performs() bool { return false }

// Prompt reports nothing: asking this model for a direction would have it read the words
// inside the brackets out.
func (t *TTS) Prompt() string { return "" }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.options.SampleRate }

// Client exposes the current underlying WebSocket so callers can use the API directly.
// It returns nil while disconnected.
func (t *TTS) Client() *websocket.Conn {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn == nil {
		return nil
	}
	return t.conn.conn
}

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

// openContext initialises a server-side context for one utterance.
func (t *TTS) openContext(sock *socket, id string) error {
	// A single space is the documented way to open a context without saying anything yet.
	message := clientMessage{
		Text:          " ",
		ContextID:     id,
		VoiceSettings: &voiceSettings{Stability: 0.5, SimilarityBoost: 0.8, Speed: t.options.Speed},
		// A short first threshold trades a little prosody for audio that starts sooner.
		Generation: &generationConf{ChunkLengthSchedule: []int{50, 120, 160, 290}},
	}
	return t.send(sock, message)
}

func (t *TTS) send(sock *socket, message clientMessage) error {
	if sock == nil {
		return stack.Wrap(errors.New("not connected"))
	}
	payload, err := json.Marshal(message)
	if err != nil {
		return stack.Wrap(err)
	}

	sock.writeMu.Lock()
	defer sock.writeMu.Unlock()
	return stack.Wrap(sock.conn.WriteMessage(websocket.TextMessage, payload))
}

// connectLocked returns the current socket or dials a replacement. dialMu must be held.
// An explicit dialContext is used only for Start's initial connection; reconnects
// otherwise use the Close-owned session lifetime.
func (t *TTS) connectLocked(dialContext context.Context) (*socket, bool, error) {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil, false, stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	if !t.started || t.ctx == nil {
		t.mu.Unlock()
		return nil, false, stack.Wrap(errors.New("elevenlabs: not started"))
	}
	if t.conn != nil {
		sock := t.conn
		t.mu.Unlock()
		return sock, false, nil
	}
	sessionCtx := t.ctx
	ctx := sessionCtx
	var cancelDial context.CancelFunc
	if dialContext != nil {
		ctx, cancelDial = mergeDialContext(dialContext, sessionCtx)
	}
	t.mu.Unlock()
	if cancelDial != nil {
		defer cancelDial()
	}

	dialer := &websocket.Dialer{HandshakeTimeout: t.options.HandshakeTimeout}
	attempt := &dialAttempt{}
	dialer.NetDialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := (&net.Dialer{}).DialContext(ctx, network, address)
		if err != nil {
			return nil, err
		}
		t.mu.Lock()
		stopping := t.shutdown || t.ctx == nil || t.ctx.Err() != nil || ctx.Err() != nil
		t.mu.Unlock()
		if stopping || !attempt.attach(conn) {
			_ = conn.Close()
			return nil, context.Canceled
		}
		t.dialConnMu.Lock()
		t.dialAttempt = attempt
		t.dialConnMu.Unlock()
		return conn, nil
	}
	header := http.Header{"xi-api-key": []string{t.options.APIKey}}
	stopDialCancel := context.AfterFunc(ctx, attempt.close)
	conn, response, err := dialer.DialContext(ctx, t.url(), header)
	t.dialConnMu.Lock()
	if t.dialAttempt == attempt {
		t.dialAttempt = nil
	}
	t.dialConnMu.Unlock()
	stopDialCancel()
	if response != nil && response.Body != nil {
		_ = response.Body.Close()
	}
	if err != nil {
		attempt.close()
		if response != nil {
			return nil, false, stack.Wrap(fmt.Errorf("elevenlabs: dial: %w (http %d)", err, response.StatusCode))
		}
		return nil, false, stack.Wrap(fmt.Errorf("elevenlabs: dial: %w", err))
	}
	if err := ctx.Err(); err != nil {
		attempt.close()
		return nil, false, stack.Wrap(fmt.Errorf("elevenlabs: dial: %w", err))
	}
	if !attempt.finish() {
		_ = conn.Close()
		return nil, false, stack.Wrap(errors.New("elevenlabs: dial canceled"))
	}

	sock := &socket{conn: conn, ready: make(chan struct{}), done: make(chan struct{})}
	t.mu.Lock()
	if t.shutdown || t.ctx.Err() != nil || (dialContext != nil && dialContext.Err() != nil) {
		t.mu.Unlock()
		_ = conn.Close()
		return nil, false, stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	t.conn = sock
	t.mu.Unlock()
	// connectLocked is called while dialMu is held. Close takes that same lock before
	// waiting, which prevents WaitGroup.Add from racing the final Wait.
	t.readers.Add(1)
	go t.readLoop(sock)
	return sock, true, nil
}

func (t *TTS) announce(sock *socket) {
	t.mu.Lock()
	announce := !t.shutdown && t.conn == sock
	t.mu.Unlock()
	if announce {
		t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	}
	sock.signalReady()
}

func (s *socket) signalReady() { s.readyOnce.Do(func() { close(s.ready) }) }

func (s *socket) closeConn() {
	s.closeOnce.Do(func() { _ = s.conn.Close() })
}

// readLoop translates server frames for this socket generation until it ends.
func (t *TTS) readLoop(sock *socket) {
	defer t.readers.Done()
	defer close(sock.done)
	<-sock.ready
	for {
		_, raw, err := sock.conn.ReadMessage()
		if err != nil {
			t.connectionEnded(sock, err, "read")
			return
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			t.logger.Debug("undecodable frame", "error", err)
			continue
		}
		t.handleMessage(sock, message)
	}
}

// connectionEnded detaches only the socket that ended. A late reader from an older
// generation cannot alter the replacement or settle its utterances.
func (t *TTS) connectionEnded(sock *socket, err error, failureContext string) {
	t.dialMu.Lock()
	t.mu.Lock()
	if t.conn != sock || t.shutdown {
		t.mu.Unlock()
		t.dialMu.Unlock()
		sock.closeConn()
		return
	}
	t.conn = nil
	failed := make([]*utterance, 0)
	for id, current := range t.active {
		if current.socket != sock {
			continue
		}
		delete(t.active, id)
		t.terminalIDs[id] = struct{}{}
		current.interrupted = true
		failed = append(failed, current)
	}
	t.mu.Unlock()
	t.dialMu.Unlock()
	sock.closeConn()

	for _, current := range failed {
		t.emitTerminal(current, err, failureContext)
	}
	clean := websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway)
	if err != nil {
		t.emitter.Send(tts.Disconnected{
			Provider: ProviderName,
			Model:    t.options.Model,
			Reason:   "upstream WebSocket closed",
			Clean:    clean,
			At:       time.Now(),
		})
	}
}

func (t *TTS) handleMessage(sock *socket, message serverMessage) {
	if failure := failureOf(message); failure != "" {
		if message.ContextID == "" {
			t.connectionEnded(sock, errors.New(failure), "server")
			return
		}
		t.failUtterance(sock, message.ContextID, errors.New(failure), "server")
		return
	}

	if message.Audio != "" {
		t.handleAudio(sock, message)
	}
	if message.IsFinal {
		t.complete(sock, message.ContextID)
	}
}

func (t *TTS) handleAudio(sock *socket, message serverMessage) {
	raw, err := base64.StdEncoding.DecodeString(message.Audio)
	if err != nil {
		t.failUtterance(sock, message.ContextID, fmt.Errorf("decode audio: %w", err), "audio")
		return
	}

	current, ok := t.lookup(sock, message.ContextID)
	if !ok {
		// The utterance was interrupted or already settled, so this audio is stale.
		return
	}
	current.eventMu.Lock()
	if t.isActive(sock, message.ContextID, current) {
		t.emitter.Send(current.tracker.Chunk(audio.FromBytes(raw, t.options.SampleRate, 1)))
	}
	current.eventMu.Unlock()
}

// complete settles an utterance and emits its summary. It is a no-op for one that has
// already been settled, so a late final frame after barge-in is harmless.
func (t *TTS) complete(sock *socket, id string) {
	t.mu.Lock()
	current, ok := t.active[id]
	if !ok || current.socket != sock || t.conn != sock {
		t.mu.Unlock()
		return
	}
	delete(t.active, id)
	t.terminalIDs[id] = struct{}{}
	t.mu.Unlock()

	<-current.ready
	current.eventMu.Lock()
	t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, current.interrupted))
	current.eventMu.Unlock()
}

// lookup returns an utterance that is still accepting audio.
func (t *TTS) lookup(sock *socket, id string) (*utterance, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.active[id]
	if !ok || current.socket != sock || t.conn != sock || current.interrupted || t.shutdown {
		return nil, false
	}
	return current, true
}

func (t *TTS) isActive(sock *socket, id string, current *utterance) bool {
	return t.isRegistered(sock, id, current)
}

func (t *TTS) isRegistered(sock *socket, id string, current *utterance) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return !t.shutdown && t.conn == sock && t.active[id] == current && !current.interrupted
}

func (t *TTS) isPresent(sock *socket, id string, current *utterance) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return !t.shutdown && t.conn == sock && t.active[id] == current
}

func (t *TTS) failUtterance(sock *socket, id string, err error, context string) {
	if id == "" {
		t.logger.Debug("server error without synthesis id", "context", context)
		return
	}
	t.mu.Lock()
	current, ok := t.active[id]
	if !ok || current.socket != sock || t.conn != sock {
		t.mu.Unlock()
		return
	}
	delete(t.active, id)
	t.terminalIDs[id] = struct{}{}
	current.interrupted = true
	t.mu.Unlock()
	t.emitTerminal(current, err, context)
}

func (t *TTS) emitTerminal(current *utterance, err error, context string) {
	<-current.ready
	current.eventMu.Lock()
	if err != nil {
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: current.tracker.ID,
			Err:         err,
			Context:     context,
			Fatal:       false,
		})
	}
	t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, true))
	current.eventMu.Unlock()
}

func (t *TTS) emitRejected(tracker *tts.Synthesis, err error, context string) {
	t.emitter.Send(tts.SynthesisStarted{
		SynthesisID: tracker.ID,
		Provider:    ProviderName,
		Model:       t.options.Model,
		Voice:       t.options.VoiceID,
		At:          time.Now(),
	})
	t.emitter.Send(tts.Error{
		Provider:    ProviderName,
		Model:       t.options.Model,
		SynthesisID: tracker.ID,
		Err:         err,
		Context:     context,
	})
	t.emitter.Send(tracker.Complete(ProviderName, t.options.Model, true))
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
