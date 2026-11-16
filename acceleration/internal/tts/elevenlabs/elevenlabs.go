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
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "elevenlabs"

// DefaultModel is the low-latency model, which is what a live conversation wants.
const DefaultModel = "eleven_flash_v2_5"

// quotaExceeded is the error a frame carries when the key the socket was opened on has no
// quota left. The key rests, so the sessions after this one are opened on another.
const quotaExceeded = "quota_exceeded"

// invalidAPIKey is the error a frame carries, right after the handshake was accepted, when the
// server does not take the key the socket was opened on.
const invalidAPIKey = "invalid_api_key"

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

// defaultStability and defaultSimilarity open a context the caller asked neither for. They
// are what this provider always sent, not ElevenLabs' own defaults (0.5 and 0.75,
// https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input, opened 2026-10-15).
const (
	defaultStability  = 0.5
	defaultSimilarity = 0.8
)

// ErrRefused marks a setting this provider will not take: a value out of range, or one the
// model has no place for. It is the caller's to fix, not a fault of the provider.
var ErrRefused = errors.New("elevenlabs: refused")

// ErrStyle marks a refused style: one outside 0 to 1, or one asked of a model that has no
// style. It is a refusal like any other (errors.Is ErrRefused holds), told apart because a
// style reaches this provider only in the caller's elevenlabs overwrite.
var ErrStyle = fmt.Errorf("%w: style", ErrRefused)

// minVoiceSetting and maxVoiceSetting bound stability, similarity_boost and style: 0 to 1 per
// https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-websocket (opened 2026-10-15).
// Live, 1.5 or -0.2 on the text-to-speech socket closes it with an unexplained 1011. No doc
// gives style a range; live on 2026-10-16, a style of 1.5 or -0.2 on Flash v2.5 was refused
// with "expected to be greater or equal to 0.0 and less or equal to 1.0".
const (
	minVoiceSetting = 0
	maxVoiceSetting = 1
)

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
	// Stability and Similarity are voice_settings, nil for the provider's own default. The
	// dialogue models take stability and no similarity.
	Stability  *float64
	Similarity *float64
	// Style is voice_settings.style, how far to exaggerate the speaker's own style, nil
	// for none. Only the models before v3 take it, see takesStyle.
	Style *float64
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
	Speed           float64 `json:"speed,omitempty"`
	// Style is a pointer because 0 is a style a caller can ask for, and nil sends nothing.
	Style *float64 `json:"style,omitempty"`
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
	announced chan struct{}
	// interrupted stops audio still in flight from being forwarded after barge-in.
	interrupted bool
}

// socket owns one physical WebSocket generation. A reader and every utterance stay
// attached to this value after a later connection replaces it.
type socket struct {
	conn *websocket.Conn
	key  string

	writeMu sync.Mutex
	ended   chan struct{}
	once    sync.Once
}

// dialAttempt lets cancellation close the raw transport while Gorilla is waiting for
// the HTTP upgrade response. Once finish succeeds, cancellation no longer owns it.
type dialAttempt struct {
	mu       sync.Mutex
	raw      net.Conn
	finished bool
	canceled bool
	callback chan struct{}
}

func (a *dialAttempt) setRaw(conn net.Conn) {
	a.mu.Lock()
	if a.finished || a.canceled {
		a.mu.Unlock()
		_ = conn.Close()
		return
	}
	a.raw = conn
	a.mu.Unlock()
}

func (a *dialAttempt) abort() {
	defer close(a.callback)
	a.mu.Lock()
	var conn net.Conn
	if !a.finished {
		a.canceled = true
		conn = a.raw
	}
	a.mu.Unlock()
	if conn != nil {
		_ = conn.Close()
	}
}

func (a *dialAttempt) finish(keep bool) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.canceled {
		return false
	}
	a.finished = true
	conn := a.raw
	a.raw = nil
	if !keep && conn != nil {
		_ = conn.Close()
	}
	return true
}

// TTS is a streaming ElevenLabs text-to-speech session.
type TTS struct {
	options Options
	keys    *keypool.Pool
	logger  *slog.Logger
	emitter *tts.Emitter

	// requestMu keeps each context's open/text sequence together and makes concurrent
	// new requests share one replacement socket after a disconnect.
	requestMu sync.Mutex
	conn      *socket
	session   context.Context
	cancel    context.CancelFunc

	mu           sync.Mutex
	active       map[string]*utterance
	terminalIDs  map[string]struct{}
	started      bool
	sessionReady bool
	shutdown     bool
}

// normalize fills in the defaults and reports what cannot be defaulted. Both endpoints
// take the same options, so they settle them the same way and differ only in the model
// they fall back to.
func normalize(options Options, fallbackModel string) (Options, *keypool.Pool, *slog.Logger, error) {
	keys := keypool.FromEnv(options.APIKey, "ELEVENLABS_API_KEY").Pin()
	if keys.Len() == 0 {
		return options, nil, nil, stack.Wrap(errors.New("elevenlabs: api key is required (set ELEVENLABS_API_KEY)"))
	}
	options.APIKey = keys.Peek()
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
		return options, nil, nil, stack.Wrap(fmt.Errorf("elevenlabs: sample rate %d is not one of %v",
			options.SampleRate, supportedSampleRates))
	}
	if err := inRange("stability", options.Stability); err != nil {
		return options, nil, nil, err
	}
	if err := inRange("similarity", options.Similarity); err != nil {
		return options, nil, nil, err
	}
	if err := styleFor(options.Model, options.Style); err != nil {
		return options, nil, nil, err
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
	return options, keys, logger, nil
}

// inRange refuses a voice setting ElevenLabs would not take rather than clamp it, since a
// caller asking for 1.5 and hearing 1 cannot tell it was not honoured.
func inRange(name string, value *float64) error {
	if !outside(value) {
		return nil
	}
	return stack.Wrap(fmt.Errorf("%w: %s %g is outside %d to %d", ErrRefused, name, *value, minVoiceSetting, maxVoiceSetting))
}

// outside reports whether a voice setting was asked for and is not one ElevenLabs takes.
func outside(value *float64) bool {
	return value != nil && (*value < minVoiceSetting || *value > maxVoiceSetting)
}

// styleFor refuses a style the model would not honour, rather than send one it ignores.
func styleFor(model string, style *float64) error {
	if style == nil {
		return nil
	}
	if outside(style) {
		return stack.Wrap(fmt.Errorf("%w %g is outside %d to %d", ErrStyle, *style, minVoiceSetting, maxVoiceSetting))
	}
	if !takesStyle(model) {
		return stack.Wrap(fmt.Errorf("%w is not taken by %s", ErrStyle, model))
	}
	return nil
}

// takesStyle reports whether a model reads voice_settings.style. The stream-input socket
// says style "is available on V2+ models"
// (https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input,
// opened 2026-10-16), and the v4 page says "Style and Speed sliders are not available in
// Eleven v4" (https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4,
// opened 2026-10-16). No doc says whether v3 reads it, and live on 2026-10-16 the file
// endpoint took a style of 1.5 on eleven_v3 and eleven_v4 without a word while it refused
// one on Flash v2.5, so the v3 and v4 families are taken to ignore it.
func takesStyle(model string) bool { return !Performs(model) }

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	options, keys, logger, err := normalize(options, DefaultModel)
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
		return nil, stack.Wrap(fmt.Errorf("%w: speed %g is outside %g to %g", ErrRefused, options.Speed, minSpeed, maxSpeed))
	}

	return &TTS{
		options:     options,
		keys:        keys,
		logger:      logger.With("provider", ProviderName, "model", options.Model),
		emitter:     tts.NewEmitter(64),
		active:      map[string]*utterance{},
		terminalIDs: map[string]struct{}{},
	}, nil
}

// Start dials the WebSocket. There is no handshake to wait for: the connection is ready
// for text as soon as it is open.
func (t *TTS) Start(ctx context.Context) error {
	t.requestMu.Lock()
	defer t.requestMu.Unlock()

	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: already started"))
	}
	t.started = true
	t.session, t.cancel = context.WithCancel(context.Background())
	session := t.session
	t.mu.Unlock()

	dialCtx, cancel := context.WithCancel(ctx)
	stopSession := context.AfterFunc(session, cancel)
	defer func() {
		stopSession()
		cancel()
	}()

	_, response, err := t.connect(dialCtx)
	if err != nil {
		t.mu.Lock()
		t.session = nil
		cancelSession := t.cancel
		t.cancel = nil
		t.mu.Unlock()
		if cancelSession != nil {
			cancelSession()
		}
		if response != nil {
			return stack.Wrap(fmt.Errorf("elevenlabs: dial: %w (http %d)", err, response.StatusCode))
		}
		return stack.Wrap(fmt.Errorf("elevenlabs: dial: %w", err))
	}
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	t.sessionReady = true
	t.mu.Unlock()
	return nil
}

// Synthesize sends text upstream. Several requests sharing an ID stream one sentence, and
// the one with Final set closes it so the tail of the audio is generated immediately.
func (t *TTS) Synthesize(request tts.Request) error {
	t.requestMu.Lock()
	defer t.requestMu.Unlock()

	if request.Voice != "" && request.Voice != t.options.VoiceID {
		return stack.Wrap(fmt.Errorf(
			"elevenlabs: the connection is bound to voice %s, open a new session for %s",
			t.options.VoiceID, request.Voice))
	}

	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: session closed"))
	}
	if !t.started {
		t.mu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: not started"))
	}
	if !t.sessionReady {
		t.mu.Unlock()
		return stack.Wrap(errors.New("elevenlabs: not started"))
	}
	if request.ID != "" {
		if _, terminal := t.terminalIDs[request.ID]; terminal {
			t.mu.Unlock()
			return nil
		}
	}
	current := t.active[request.ID]
	t.mu.Unlock()

	opened := current == nil
	if opened {
		sock, _, err := t.connect(t.session)
		if err != nil {
			return stack.Wrap(fmt.Errorf("elevenlabs: reconnect: %w", err))
		}
		t.mu.Lock()
		if t.shutdown {
			t.mu.Unlock()
			return stack.Wrap(errors.New("elevenlabs: session closed"))
		}
		if request.ID != "" {
			if _, terminal := t.terminalIDs[request.ID]; terminal {
				t.mu.Unlock()
				return nil
			}
		}
		// The reader may detach the socket between connect returning and registration.
		if t.conn != sock {
			t.mu.Unlock()
			return stack.Wrap(errors.New("elevenlabs: connection ended before synthesis"))
		}
		current = &utterance{
			tracker:   tts.NewSynthesis(request.ID),
			socket:    sock,
			announced: make(chan struct{}),
		}
		t.active[current.tracker.ID] = current
		t.mu.Unlock()
	}

	if opened {
		t.emitter.Send(tts.SynthesisStarted{
			SynthesisID: current.tracker.ID,
			Provider:    ProviderName,
			Model:       t.options.Model,
			Voice:       t.options.VoiceID,
			At:          time.Now(),
		})
		close(current.announced)
		if err := t.openContext(current.socket, current.tracker.ID); err != nil {
			return err
		}
	}

	if request.Text != "" {
		current.tracker.AddText(request.Text)
		// The trailing space is what ElevenLabs uses to tell one word from the next
		// across deltas.
		err := t.send(current.socket, clientMessage{Text: request.Text + " ", ContextID: current.tracker.ID})
		if err != nil {
			return stack.Wrap(fmt.Errorf("elevenlabs: send text: %w", err))
		}
	}

	if request.Final {
		// Closing the context flushes the remaining audio and makes the server report
		// the synthesis as final straight after it.
		if err := t.send(current.socket, clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			return stack.Wrap(fmt.Errorf("elevenlabs: close context: %w", err))
		}
	}
	return nil
}

// Interrupt closes every context in flight, which stops the server generating and stops
// audio already on the wire from being forwarded.
func (t *TTS) Interrupt() error {
	t.mu.Lock()
	if len(t.active) == 0 {
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()

	t.requestMu.Lock()
	defer t.requestMu.Unlock()

	t.mu.Lock()
	interrupting := make([]*utterance, 0, len(t.active))
	for _, current := range t.active {
		if !current.interrupted {
			current.interrupted = true
			t.terminalIDs[current.tracker.ID] = struct{}{}
			interrupting = append(interrupting, current)
		}
	}
	t.mu.Unlock()

	var failures []error
	for _, current := range interrupting {
		if err := t.send(current.socket, clientMessage{ContextID: current.tracker.ID, CloseContext: true}); err != nil {
			failures = append(failures, err)
		}
		t.complete(current.socket, current.tracker.ID)
	}

	if len(failures) > 0 {
		return stack.Wrap(fmt.Errorf("elevenlabs: interrupt: %w", errors.Join(failures...)))
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
	cancel := t.cancel
	t.mu.Unlock()
	if cancel != nil {
		cancel()
	}

	// Cancellation above must happen before waiting for a synthesis that may be in
	// the replacement handshake.
	t.requestMu.Lock()
	defer t.requestMu.Unlock()

	t.mu.Lock()
	sock := t.conn
	t.mu.Unlock()

	t.settleOutstanding()

	if sock != nil {
		// The protocol tears down in two steps: we ask the server to close, then it does.
		// Waiting for its close avoids ending the session on an abnormal closure.
		if err := t.send(sock, clientMessage{CloseSocket: true}); err != nil {
			t.logger.Debug("close_socket not delivered", "error", err)
		} else {
			select {
			case <-sock.ended:
			case <-time.After(t.options.CloseTimeout):
				t.logger.Debug("server did not close the socket in time")
			}
		}
		_ = sock.conn.Close()
		t.mu.Lock()
		if t.conn == sock {
			t.conn = nil
		}
		t.mu.Unlock()
	}

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

// Client exposes the underlying WebSocket so callers can use the API directly.
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
		Text:      " ",
		ContextID: id,
		VoiceSettings: &voiceSettings{
			Stability:       valueOr(t.options.Stability, defaultStability),
			SimilarityBoost: valueOr(t.options.Similarity, defaultSimilarity),
			Speed:           t.options.Speed,
			Style:           t.options.Style,
		},
		// A short first threshold trades a little prosody for audio that starts sooner.
		Generation: &generationConf{ChunkLengthSchedule: []int{50, 120, 160, 290}},
	}
	if err := t.send(sock, message); err != nil {
		return stack.Wrap(fmt.Errorf("elevenlabs: open context: %w", err))
	}
	return nil
}

// valueOr is what was asked for, or the fallback when nothing was.
func valueOr(asked *float64, fallback float64) float64 {
	if asked == nil {
		return fallback
	}
	return *asked
}

func (t *TTS) send(sock *socket, message clientMessage) error {
	payload, err := json.Marshal(message)
	if err != nil {
		return stack.Wrap(err)
	}

	if sock == nil {
		return stack.Wrap(errors.New("not connected"))
	}
	sock.writeMu.Lock()
	err = sock.conn.WriteMessage(websocket.TextMessage, payload)
	sock.writeMu.Unlock()
	if err != nil {
		t.connectionEnded(sock, err)
	}
	return stack.Wrap(err)
}

// connect returns the current socket or installs one replacement. Synthesis calls are
// serialized while connecting, so concurrent new requests share the same successful dial.
func (t *TTS) connect(ctx context.Context) (*socket, *http.Response, error) {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil, nil, errors.New("session closed")
	}
	if !t.started {
		t.mu.Unlock()
		return nil, nil, errors.New("not started")
	}
	if t.conn != nil {
		sock := t.conn
		t.mu.Unlock()
		return sock, nil, nil
	}
	t.mu.Unlock()

	attempt := &dialAttempt{callback: make(chan struct{})}
	stop := context.AfterFunc(ctx, attempt.abort)
	dialer := &websocket.Dialer{
		HandshakeTimeout: t.options.HandshakeTimeout,
		NetDialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			conn, err := (&net.Dialer{}).DialContext(ctx, network, address)
			if err == nil {
				attempt.setRaw(conn)
			}
			return conn, err
		},
	}
	conn, response, key, err := t.keys.Dial(ctx, dialer, func(key string) (string, http.Header) {
		return t.url(), http.Header{"xi-api-key": []string{key}}
	})
	stopped := stop()
	if !stopped {
		<-attempt.callback
	}
	if err != nil {
		attempt.finish(false)
		return nil, response, err
	}
	if ctx.Err() != nil || !attempt.finish(true) {
		_ = conn.Close()
		if ctx.Err() != nil {
			return nil, response, ctx.Err()
		}
		return nil, response, context.Canceled
	}

	sock := &socket{conn: conn, key: key, ended: make(chan struct{})}
	t.mu.Lock()
	if t.shutdown || t.conn != nil {
		shutdown := t.shutdown
		current := t.conn
		t.mu.Unlock()
		_ = conn.Close()
		if shutdown {
			return nil, response, errors.New("session closed")
		}
		return current, response, nil
	}
	t.conn = sock
	t.mu.Unlock()

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	go t.readLoop(sock)
	return sock, response, nil
}

// readLoop translates frames from one physical connection. A replaced reader cannot
// change the state of the newer socket.
func (t *TTS) readLoop(sock *socket) {
	refused := false
	for {
		_, raw, err := sock.conn.ReadMessage()
		if err != nil {
			t.connectionEnded(sock, err)
			return
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			t.logger.Debug("undecodable frame", "error", err)
			continue
		}
		if message.Error == quotaExceeded {
			t.keys.QuotaExceeded(sock.key)
		}
		if message.Error == invalidAPIKey && !refused {
			refused = true
			t.keys.InvalidKey()
		}
		t.handleMessage(sock, message)
	}
}

func (t *TTS) connectionEnded(sock *socket, err error) {
	sock.once.Do(func() {
		defer close(sock.ended)
		t.mu.Lock()
		current := t.conn == sock
		if current {
			t.conn = nil
		}
		shutdown := t.shutdown
		var outstanding []*utterance
		for id, active := range t.active {
			if active.socket == sock {
				delete(t.active, id)
				t.terminalIDs[id] = struct{}{}
				outstanding = append(outstanding, active)
			}
		}
		t.mu.Unlock()

		_ = sock.conn.Close()
		if current && !shutdown {
			t.handleReadError(err)
		}
		for _, active := range outstanding {
			<-active.announced
			t.emitter.Send(active.tracker.Complete(ProviderName, t.options.Model, true))
		}
	})
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

func (t *TTS) handleMessage(sock *socket, message serverMessage) {
	t.mu.Lock()
	currentSocket := t.conn == sock
	t.mu.Unlock()
	if !currentSocket {
		return
	}

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
		t.handleAudio(sock, message)
	}
	if message.IsFinal {
		t.complete(sock, message.ContextID)
	}
}

func (t *TTS) handleAudio(sock *socket, message serverMessage) {
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

	current, ok := t.lookup(sock, message.ContextID)
	if !ok {
		// The utterance was interrupted or already settled, so this audio is stale.
		return
	}
	t.emitter.Send(current.tracker.Chunk(audio.FromBytes(raw, t.options.SampleRate, 1)))
}

// complete settles an utterance and emits its summary. It is a no-op for one that has
// already been settled, so a late final frame after barge-in is harmless.
func (t *TTS) complete(sock *socket, id string) {
	t.mu.Lock()
	current, ok := t.active[id]
	if ok && current.socket == sock {
		delete(t.active, id)
	} else {
		ok = false
	}
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
func (t *TTS) lookup(sock *socket, id string) (*utterance, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.active[id]
	if !ok || current.socket != sock || current.interrupted {
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
