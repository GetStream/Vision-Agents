// Package cartesia implements the tts.TTS contract against Cartesia's streaming WebSocket.
//
// Speech is generated on server-side contexts, which map directly onto tts.Request.ID: the
// first frame naming a context starts a synthesis, frames carrying continue keep it open,
// and the one without it ends it. Cancelling a context is barge-in. Audio arrives
// base64-encoded in JSON frames as it is generated, so the first sound reaches the listener
// long before the sentence is finished.
package cartesia

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
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "cartesia"

// DefaultModel is Sonic 3.6, which is only served under this name while it is in beta.
// Cartesia says it goes generally available later this month, at which point this becomes
// sonic-3.6 and sonic-preview goes back to meaning whatever is next.
const DefaultModel = "sonic-preview"

// DefaultVoiceID is Skylar, Cartesia's own default and one of the voices they recommend for
// agents, so the provider works without a voice having been picked.
const DefaultVoiceID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"

// DefaultSampleRate matches the other providers: the highest rate the raw PCM output offers
// without paying for bandwidth nobody hears.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production endpoint.
const defaultBaseURL = "wss://api.cartesia.ai"

// apiVersion pins the request and response shapes. Cartesia dates its API rather than
// numbering it, and sends the version as a query parameter.
const apiVersion = "2026-08-14"

// reconnectDelay is how long to wait before replacing a socket that has closed.
//
// Cartesia hangs up on a TTS connection it has not been asked to say anything on for five
// minutes, and a caller who spends that long listening or thinking is ordinary. Ping frames
// do not count: they keep their Line sockets open but not these, so the only way to still
// have a voice afterwards is to open another one.
const reconnectDelay = time.Second

// supportedSampleRates are the rates the raw output format offers.
var supportedSampleRates = []int{8_000, 16_000, 22_050, 24_000, 44_100, 48_000}

// Options configures the provider. APIKey falls back to CARTESIA_API_KEY and VoiceID to
// CARTESIA_VOICE_ID.
type Options struct {
	APIKey string
	// VoiceID is the speaker used for utterances that do not name one of their own.
	VoiceID string
	Model   string
	// Language is a base ISO code such as "en". Empty lets the model infer it.
	Language string
	// SampleRate is the rate to synthesise at, one of supportedSampleRates.
	SampleRate int
	// BaseURL overrides the endpoint, for a proxy or a test server.
	BaseURL string
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger

	// reconnect is how long to wait before replacing a socket that closed. Unexported
	// because it exists so a test need not wait out the real one.
	reconnect time.Duration
}

// generation asks for speech on a context. Model, voice and output format go on every
// frame because Cartesia requires them to agree across the whole context.
type generation struct {
	ContextID    string       `json:"context_id"`
	ModelID      string       `json:"model_id"`
	Transcript   string       `json:"transcript"`
	Voice        voice        `json:"voice"`
	OutputFormat outputFormat `json:"output_format"`
	Language     string       `json:"language,omitempty"`
	// Continue says more text is coming. The frame that leaves it out ends the context
	// and makes the server generate the tail immediately.
	Continue bool `json:"continue"`
	// MaxBufferDelayMs is how long the server may wait for more text before generating.
	// Zero means never: the agent already aggregates into sentences, and waiting again
	// on top of that is the mistake Cartesia's own buffering guide warns about.
	MaxBufferDelayMs int `json:"max_buffer_delay_ms"`
}

type voice struct {
	ID string `json:"id"`
}

type outputFormat struct {
	Container  string `json:"container"`
	Encoding   string `json:"encoding"`
	SampleRate int    `json:"sample_rate"`
}

// cancellation stops a context generating anything further.
type cancellation struct {
	ContextID string `json:"context_id"`
	Cancel    bool   `json:"cancel"`
}

// serverMessage is a frame sent by Cartesia. One shape covers every type it sends, since
// only the type field decides which of the others carry anything.
type serverMessage struct {
	Type      string `json:"type"`
	ContextID string `json:"context_id"`
	// Data is base64-encoded PCM in the requested output format.
	Data string `json:"data"`
	Done bool   `json:"done"`
	// Title, Message and ErrorCode describe a failure.
	Title     string `json:"title"`
	Message   string `json:"message"`
	ErrorCode string `json:"error_code"`
}

// The frame types Cartesia sends. Timestamps and phoneme timestamps are not asked for and
// so are not listed.
const (
	typeChunk = "chunk"
	typeDone  = "done"
	typeError = "error"
	// typeFlushDone marks the end of one run of audio within a context. Nothing here asks
	// for a flush, but the server draws the boundary anyway at the end of an utterance.
	typeFlushDone = "flush_done"
)

// utterance is one synthesis in flight.
type utterance struct {
	tracker *tts.Synthesis
	// voice is the speaker this context was opened with. Cartesia binds a context to one
	// voice, so it is remembered rather than read from the options on every frame.
	voice string
	// interrupted stops audio still in flight from being forwarded after barge-in.
	interrupted bool
}

// TTS is a streaming Cartesia text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// done stops a reconnect that is waiting, so a socket is not reopened behind Close.
	done chan struct{}

	mu       sync.Mutex
	active   map[string]*utterance
	started  bool
	shutdown bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("CARTESIA_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("cartesia: api key is required (set CARTESIA_API_KEY)")
	}
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("CARTESIA_VOICE_ID")
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
		return nil, fmt.Errorf("cartesia: sample rate %d is not one of %v",
			options.SampleRate, supportedSampleRates)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 15 * time.Second
	}
	if options.reconnect == 0 {
		options.reconnect = reconnectDelay
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &TTS{
		options: options,
		logger:  logger.With("provider", ProviderName, "model", options.Model),
		emitter: tts.NewEmitter(64),
		active:  map[string]*utterance{},
		done:    make(chan struct{}),
	}, nil
}

// Start dials the WebSocket. There is no handshake to wait for: the connection is ready
// for text as soon as it is open.
func (t *TTS) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return errors.New("cartesia: already started")
	}
	t.started = true
	t.mu.Unlock()

	if err := t.dial(ctx); err != nil {
		return err
	}

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	return nil
}

// dial opens a connection and starts reading it.
//
// The read loop is handed the connection it owns rather than reading the field, so a loop
// left over from a socket that has been replaced cannot consume the new one's frames.
func (t *TTS) dial(ctx context.Context) error {
	dialer := &websocket.Dialer{HandshakeTimeout: t.options.HandshakeTimeout}
	header := http.Header{"X-API-Key": []string{t.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, t.url(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("cartesia: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("cartesia: dial: %w", err)
	}

	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		conn.Close()
		return errors.New("cartesia: closed")
	}
	t.conn = conn
	t.mu.Unlock()

	go t.readLoop(conn)
	return nil
}

// redial opens a replacement for a socket that has gone, so the next thing said does not
// wait on a handshake.
//
// One attempt is enough here. If it fails the connection is opened on demand by the next
// utterance instead, which costs that utterance a handshake but never leaves the agent
// mute, and retrying in here would be a second thing that has to know when to give up.
func (t *TTS) redial() {
	select {
	case <-t.done:
		return
	case <-time.After(t.options.reconnect):
	}

	if err := t.dial(context.Background()); err != nil {
		t.logger.Debug("could not reopen the voice, leaving it to the next utterance",
			"error", err)
		return
	}
	t.logger.Info("reopened the voice after an idle hang-up")
	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
}

// Synthesize sends text upstream. Several requests sharing an ID stream one utterance, and
// the one with Final set ends the context so the tail of the audio is generated at once.
func (t *TTS) Synthesize(request tts.Request) error {
	current, opened, err := t.utteranceFor(request)
	if err != nil {
		return err
	}

	if opened {
		t.emitter.Send(tts.SynthesisStarted{
			SynthesisID: current.tracker.ID,
			Provider:    ProviderName,
			Model:       t.options.Model,
			Voice:       current.voice,
			At:          time.Now(),
		})
	}

	// A final request with no text still has to go: an empty transcript that does not
	// continue is how a context is closed when the last words are already sent.
	if request.Text == "" && !request.Final {
		return nil
	}

	current.tracker.AddText(request.Text)
	message := generation{
		ContextID:  current.tracker.ID,
		ModelID:    t.options.Model,
		Transcript: request.Text,
		Voice:      voice{ID: current.voice},
		OutputFormat: outputFormat{
			Container:  "raw",
			Encoding:   "pcm_s16le",
			SampleRate: t.options.SampleRate,
		},
		Language: language(request.Language, t.options.Language),
		Continue: !request.Final,
	}
	if err := t.send(message); err != nil {
		return fmt.Errorf("cartesia: send text: %w", err)
	}
	return nil
}

// Interrupt cancels every context in flight, which stops the server generating and stops
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
		if err := t.send(cancellation{ContextID: current.tracker.ID, Cancel: true}); err != nil {
			failures = append(failures, err)
		}
		t.complete(current.tracker.ID)
	}

	if len(failures) > 0 {
		return fmt.Errorf("cartesia: interrupt: %w", errors.Join(failures...))
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close tears down the connection. Anything still in flight is reported as interrupted so
// no work goes unaccounted for.
func (t *TTS) Close() error {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil
	}
	t.shutdown = true
	conn := t.conn
	t.mu.Unlock()

	// Stopped before the socket goes, so nothing reopens the connection behind this.
	close(t.done)

	t.settleOutstanding()

	if conn != nil {
		// Cartesia has no command for ending the session, so the close frame is the whole
		// goodbye. A server that does not answer it is not worth waiting on.
		deadline := time.Now().Add(time.Second)
		message := websocket.FormatCloseMessage(websocket.CloseNormalClosure, "")
		if err := conn.WriteControl(websocket.CloseMessage, message, deadline); err != nil {
			t.logger.Debug("close frame not delivered", "error", err)
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

// Performs reports false: Sonic reads a bracketed direction out as words.
func (t *TTS) Performs() bool { return false }

// Prompt reports nothing: there is no direction this voice would act.
func (t *TTS) Prompt() string { return "" }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.options.SampleRate }

// Client exposes the underlying WebSocket so callers can use the API directly.
func (t *TTS) Client() *websocket.Conn { return t.conn }

// url builds the endpoint. The API version is dated rather than numbered and is required.
func (t *TTS) url() string {
	return fmt.Sprintf("%s/tts/websocket?cartesia_version=%s",
		strings.TrimSuffix(t.options.BaseURL, "/"), apiVersion)
}

// utteranceFor returns the tracker for a request, reporting whether it had to be created.
func (t *TTS) utteranceFor(request tts.Request) (*utterance, bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, false, errors.New("cartesia: session closed")
	}
	if !t.started || t.conn == nil {
		return nil, false, errors.New("cartesia: not started")
	}

	if request.ID != "" {
		if existing, ok := t.active[request.ID]; ok {
			// Cartesia binds a context to the voice it was opened with, so changing it
			// halfway would be one utterance said in two voices.
			if request.Voice != "" && request.Voice != existing.voice {
				return nil, false, fmt.Errorf(
					"cartesia: utterance %s is being said in voice %s, not %s",
					request.ID, existing.voice, request.Voice)
			}
			return existing, false, nil
		}
	}

	speaker := request.Voice
	if speaker == "" {
		speaker = t.options.VoiceID
	}
	current := &utterance{tracker: tts.NewSynthesis(request.ID), voice: speaker}
	t.active[current.tracker.ID] = current
	return current, true, nil
}

func (t *TTS) send(message any) error {
	payload, err := json.Marshal(message)
	if err != nil {
		return err
	}

	t.writeMu.Lock()
	defer t.writeMu.Unlock()

	t.mu.Lock()
	conn, shutdown := t.conn, t.shutdown
	t.mu.Unlock()
	if shutdown {
		return errors.New("not connected")
	}
	if conn != nil {
		if err := conn.WriteMessage(websocket.TextMessage, payload); err == nil {
			return nil
		}
	}

	// The keepalive should mean the socket is still up, but a connection can be lost for
	// reasons no ping prevents, and the caller is mid-sentence. Redialling costs a
	// handshake once; not redialling costs the agent its voice for the rest of the call.
	t.logger.Debug("reconnecting to speak")
	if err := t.dial(context.Background()); err != nil {
		return err
	}

	t.mu.Lock()
	conn = t.conn
	t.mu.Unlock()
	if conn == nil {
		return errors.New("not connected")
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

// readLoop translates server frames into events until the connection ends.
func (t *TTS) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			t.handleReadError(conn, err)
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

func (t *TTS) handleReadError(conn *websocket.Conn, err error) {
	t.mu.Lock()
	shutdown, superseded := t.shutdown, t.conn != conn
	t.mu.Unlock()
	if shutdown || superseded {
		// A socket that has already been replaced has nothing left to report: whatever
		// was in flight on it has moved to the connection that took its place.
		return
	}

	// The connection is gone, so nothing in flight will ever finish on its own.
	t.settleOutstanding()

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		t.emitter.Send(tts.Disconnected{
			Provider: ProviderName,
			Model:    t.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		// A clean close mid-call is Cartesia's idle timeout, not the end of the
		// session: the agent still has a caller and will need a voice for them.
		go t.redial()
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
	switch message.Type {
	case typeChunk:
		t.handleAudio(message)

	case typeDone:
		t.complete(message.ContextID)

	case typeError:
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: message.ContextID,
			Err:         errors.New(failureOf(message)),
			Context:     "server",
			Fatal:       false,
		})
		// An error ends the context, so the utterance it names will never finish on its
		// own and would otherwise be left in flight forever.
		t.complete(message.ContextID)

	case typeFlushDone:
		// The audio either side of a flush is one utterance here, and the done frame is
		// what settles it.

	default:
		t.logger.Debug("unhandled frame", "type", message.Type)
	}
}

func (t *TTS) handleAudio(message serverMessage) {
	raw, err := base64.StdEncoding.DecodeString(message.Data)
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
// already been settled, so a late done frame after barge-in is harmless.
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

// language picks the code to synthesise in, preferring the one on the request.
func language(request, session string) string {
	if request != "" {
		return strings.ToLower(request)
	}
	return strings.ToLower(session)
}

// failureOf describes a failure frame in one line. Cartesia sends a short title and a
// longer message, and either may be missing.
func failureOf(message serverMessage) string {
	parts := make([]string, 0, 3)
	for _, part := range []string{message.Title, message.Message, message.ErrorCode} {
		if part != "" {
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		return "cartesia rejected the request"
	}
	return strings.Join(parts, ": ")
}
