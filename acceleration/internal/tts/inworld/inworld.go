// Package inworld implements the tts.TTS contract against Inworld's bidirectional
// WebSocket.
//
// One utterance is one server-side context, which maps onto tts.Request.ID: create opens
// it, send_text streams the words, flush_context ends the text so the tail generates, and
// close_context is barge-in. Audio arrives base64-encoded PCM as it is generated. Frames
// for a context that has already been closed keep draining onto the shared socket, so they
// are dropped rather than spoken over the next sentence.
package inworld

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "inworld"

// DefaultModel is Realtime TTS-2 Flash, the low-latency member of the TTS-2 family.
const DefaultModel = "inworld-tts-2-flash"

// DefaultVoiceID is Sarah, Inworld's own default and the voice the Python plugin uses
// when nobody has picked one.
const DefaultVoiceID = "Sarah"

// DefaultSampleRate matches the other providers: the highest rate the raw PCM output
// offers without paying for bandwidth nobody hears.
const DefaultSampleRate = 24_000

// defaultBaseURL is the production host. The path is appended in url().
const defaultBaseURL = "wss://api.inworld.ai"

// streamPath is the bidirectional socket the Python plugin already speaks.
const streamPath = "/tts/v1/voice:streamBidirectional"

// reconnectDelay is how long to wait before replacing a socket that has closed.
const reconnectDelay = time.Second

// keepAlivePeriod is how often a ping is sent so an idle call does not lose its voice.
const keepAlivePeriod = 30 * time.Second

// Options configures the provider. APIKey falls back to INWORLD_API_KEY and VoiceID to
// INWORLD_VOICE_ID.
type Options struct {
	APIKey string
	// VoiceID is the speaker used for utterances that do not name one of their own.
	VoiceID string
	Model   string
	// SampleRate is the rate to synthesise at.
	SampleRate int
	// BaseURL overrides the host, for a proxy or a test server.
	BaseURL string
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger

	// reconnect is how long to wait before replacing a socket that closed. Unexported
	// because it exists so a test need not wait out the real one.
	reconnect time.Duration
}

// createConfig is the body of the frame that opens a context.
type createConfig struct {
	VoiceID       string      `json:"voiceId"`
	ModelID       string      `json:"modelId"`
	AudioConfig   audioConfig `json:"audioConfig"`
	Temperature   float64     `json:"temperature"`
	AutoMode      bool        `json:"autoMode"`
	TimestampType string      `json:"timestampType"`
}

type audioConfig struct {
	AudioEncoding   string `json:"audioEncoding"`
	SampleRateHertz int    `json:"sampleRateHertz"`
}

type sendText struct {
	Text string `json:"text"`
}

// emptyAction is a JSON object with no fields. Inworld's flush_context and close_context
// are that shape, and encoding/json omits an empty map under omitempty, so a pointer to
// an empty struct is what actually appears on the wire.
type emptyAction struct{}

// clientFrame is one message sent upstream. Only one of the action fields is set.
type clientFrame struct {
	ContextID    string        `json:"contextId"`
	Create       *createConfig `json:"create,omitempty"`
	SendText     *sendText     `json:"send_text,omitempty"`
	FlushContext *emptyAction  `json:"flush_context,omitempty"`
	CloseContext *emptyAction  `json:"close_context,omitempty"`
}

// serverFrame is a message from Inworld. Audio and control live under result; a top-level
// error is a session-wide failure.
type serverFrame struct {
	Result *serverResult `json:"result"`
	Error  any           `json:"error"`
}

type serverResult struct {
	ContextID      string          `json:"contextId"`
	AudioChunk     *serverAudio    `json:"audioChunk"`
	FlushCompleted json.RawMessage `json:"flushCompleted"`
	ContextClosed  json.RawMessage `json:"contextClosed"`
	Status         *serverStatus   `json:"status"`
}

type serverAudio struct {
	AudioContent string `json:"audioContent"`
}

type serverStatus struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// utterance is one synthesis in flight.
type utterance struct {
	tracker *tts.Synthesis
	voice   string
	// interrupted stops audio still in flight from being forwarded after barge-in.
	interrupted bool
	// closed means close_context has already gone out. A second one provokes an error
	// frame that derails the next utterance.
	closed bool
	// settled means the utterance has been reported complete. It is kept here rather than
	// forgotten because the server goes on sending this context's own tail afterwards, and
	// that tail is the end of a sentence the caller is still waiting to hear.
	settled bool
}

// TTS is a streaming Inworld text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// done stops a reconnect that is waiting, so a socket is not reopened behind Close.
	done chan struct{}

	mu sync.Mutex
	// active is every utterance the session still has a use for, which is not the same as
	// every utterance still being spoken: one stays here after it has been reported complete
	// so the tail the server is still sending is forwarded rather than dropped. lookup wants
	// both, utteranceFor and complete only the unsettled ones.
	active   map[string]*utterance
	started  bool
	shutdown bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("INWORLD_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("inworld: api key is required (set INWORLD_API_KEY)")
	}
	if options.VoiceID == "" {
		options.VoiceID = os.Getenv("INWORLD_VOICE_ID")
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
	if options.SampleRate <= 0 {
		return nil, fmt.Errorf("inworld: sample rate must be positive, got %d", options.SampleRate)
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
// for a create frame as soon as it is open.
func (t *TTS) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return errors.New("inworld: already started")
	}
	t.started = true
	t.mu.Unlock()

	if err := t.dial(ctx); err != nil {
		return err
	}

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	go t.keepAlive()
	return nil
}

func (t *TTS) dial(ctx context.Context) error {
	dialer := &websocket.Dialer{HandshakeTimeout: t.options.HandshakeTimeout}
	header := http.Header{
		"Authorization": []string{"Basic " + t.options.APIKey},
		"X-Request-Id":  []string{uuid.NewString()},
	}

	conn, response, err := dialer.DialContext(ctx, t.url(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("inworld: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("inworld: dial: %w", err)
	}

	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		conn.Close()
		return errors.New("inworld: closed")
	}
	t.conn = conn
	t.mu.Unlock()

	go t.readLoop(conn)
	return nil
}

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
// the one with Final set flushes the context so the tail of the audio is generated at once.
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
		if err := t.send(clientFrame{
			ContextID: current.tracker.ID,
			Create: &createConfig{
				VoiceID: current.voice,
				ModelID: t.options.Model,
				AudioConfig: audioConfig{
					AudioEncoding:   "PCM",
					SampleRateHertz: t.options.SampleRate,
				},
				Temperature: 1.1,
				// The agent already flushes at sentence boundaries. Letting the server
				// decide as well splits a streamed utterance into several billed ones.
				AutoMode:      false,
				TimestampType: "TIMESTAMP_TYPE_UNSPECIFIED",
			},
		}); err != nil {
			return fmt.Errorf("inworld: create context: %w", err)
		}
	}

	if request.Text == "" && !request.Final {
		return nil
	}

	if request.Text != "" {
		current.tracker.AddText(request.Text)
		if err := t.send(clientFrame{
			ContextID: current.tracker.ID,
			SendText:  &sendText{Text: request.Text},
		}); err != nil {
			return fmt.Errorf("inworld: send text: %w", err)
		}
	}

	if request.Final {
		if err := t.send(clientFrame{
			ContextID:    current.tracker.ID,
			FlushContext: &emptyAction{},
		}); err != nil {
			return fmt.Errorf("inworld: flush: %w", err)
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
		if err := t.closeContext(current); err != nil {
			failures = append(failures, err)
		}
		t.complete(current.tracker.ID)
	}

	if len(failures) > 0 {
		return fmt.Errorf("inworld: interrupt: %w", errors.Join(failures...))
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

	close(t.done)

	t.settleOutstanding()

	if conn != nil {
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

func (t *TTS) Provider() string { return ProviderName }

func (t *TTS) Model() string { return t.options.Model }

func (t *TTS) Streaming() bool { return true }

func (t *TTS) Performs() bool { return false }

func (t *TTS) Prompt() string { return "" }

func (t *TTS) SampleRate() int { return t.options.SampleRate }

func (t *TTS) Client() *websocket.Conn { return t.conn }

func (t *TTS) url() string {
	return strings.TrimSuffix(t.options.BaseURL, "/") + streamPath
}

func (t *TTS) utteranceFor(request tts.Request) (*utterance, bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, false, errors.New("inworld: session closed")
	}
	if !t.started || t.conn == nil {
		return nil, false, errors.New("inworld: not started")
	}

	if request.ID != "" {
		if existing, ok := t.active[request.ID]; ok && !existing.settled {
			if request.Voice != "" && request.Voice != existing.voice {
				return nil, false, fmt.Errorf(
					"inworld: utterance %s is being said in voice %s, not %s",
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

func (t *TTS) send(message clientFrame) error {
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

func (t *TTS) keepAlive() {
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()

	for {
		select {
		case <-t.done:
			return
		case <-ticker.C:
			t.mu.Lock()
			conn := t.conn
			t.mu.Unlock()
			if conn == nil {
				continue
			}
			deadline := time.Now().Add(time.Second)
			if err := conn.WriteControl(websocket.PingMessage, nil, deadline); err != nil {
				t.logger.Debug("keepalive not delivered", "error", err)
			}
		}
	}
}

func (t *TTS) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			t.handleReadError(conn, err)
			return
		}

		var message serverFrame
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
		return
	}

	t.settleOutstanding()

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		t.emitter.Send(tts.Disconnected{
			Provider: ProviderName,
			Model:    t.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
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

func (t *TTS) handleMessage(message serverFrame) {
	if message.Error != nil {
		t.emitter.Send(tts.Error{
			Provider: ProviderName,
			Model:    t.options.Model,
			Err:      fmt.Errorf("inworld: %v", message.Error),
			Context:  "server",
			Fatal:    false,
		})
		return
	}
	if message.Result == nil {
		return
	}
	result := message.Result

	if result.Status != nil && result.Status.Code != 0 {
		t.handleStatus(result)
		return
	}

	if result.ContextID != "" {
		if _, ok := t.lookup(result.ContextID); !ok && result.AudioChunk != nil {
			// Audio for a context the agent has left behind: one barge-in abandoned, or
			// one the server has already closed. It keeps draining onto the shared socket
			// afterwards, and speaking it would cut into whatever came next.
			return
		}
	}

	if result.AudioChunk != nil && result.AudioChunk.AudioContent != "" {
		t.handleAudio(result)
	}
	if len(result.FlushCompleted) > 0 || len(result.ContextClosed) > 0 {
		if result.ContextID == "" {
			return
		}
		// close_context once: a second one provokes an error frame that derails the
		// next utterance. The server sending contextClosed is the close already done.
		if len(result.ContextClosed) > 0 {
			// Marking it before forgetting it looks redundant, and is not: Interrupt takes
			// the utterance out of the map and closes it without the lock, so a barge-in
			// landing here already holds a pointer that outlives the map entry, and the flag
			// is what stops it sending a second close.
			t.markClosed(result.ContextID)
			t.complete(result.ContextID)
			// The server has closed the context, so nothing more of it is coming.
			t.forget(result.ContextID)
			return
		}
		// A flush says the text is in, not that the audio for it has all been sent. The
		// utterance is reported finished here so a caller is never left waiting on a
		// context the server declines to acknowledge, but it is kept so the rest of its
		// own audio is still spoken rather than dropped as somebody else's.
		t.closeByID(result.ContextID)
		t.complete(result.ContextID)
	}
}

func (t *TTS) handleStatus(result *serverResult) {
	err := errors.New(result.Status.Message)
	if result.Status.Message == "" {
		err = fmt.Errorf("inworld rejected the request (status %d)", result.Status.Code)
	}
	t.emitter.Send(tts.Error{
		Provider:    ProviderName,
		Model:       t.options.Model,
		SynthesisID: result.ContextID,
		Err:         err,
		Context:     "server",
		Fatal:       false,
	})
	if result.ContextID != "" {
		t.complete(result.ContextID)
	}
	if strings.Contains(strings.ToLower(result.Status.Message), "max contexts limit reached") {
		t.resetConnection()
	}
}

func (t *TTS) handleAudio(result *serverResult) {
	raw, err := base64.StdEncoding.DecodeString(result.AudioChunk.AudioContent)
	if err != nil {
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: result.ContextID,
			Err:         fmt.Errorf("decode audio: %w", err),
			Context:     "audio",
		})
		return
	}

	current, ok := t.lookup(result.ContextID)
	if !ok {
		return
	}
	t.emitter.Send(current.tracker.Chunk(audio.FromBytes(raw, t.options.SampleRate, 1)))
}

// complete reports an utterance finished, once. The utterance is left in the session rather
// than forgotten: flushing a context only says the text is in, and the audio for it goes on
// arriving afterwards. forget is what removes it, once the server has closed the context.
func (t *TTS) complete(id string) {
	t.mu.Lock()
	current, ok := t.active[id]
	if !ok || current.settled {
		t.mu.Unlock()
		return
	}
	current.settled = true
	interrupted := current.interrupted
	t.mu.Unlock()

	t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, interrupted))
}

func (t *TTS) settleOutstanding() {
	t.mu.Lock()
	outstanding := make([]*utterance, 0, len(t.active))
	for _, current := range t.active {
		if !current.settled {
			outstanding = append(outstanding, current)
		}
	}
	clear(t.active)
	t.mu.Unlock()

	for _, current := range outstanding {
		t.emitter.Send(current.tracker.Complete(ProviderName, t.options.Model, true))
	}
}

func (t *TTS) lookup(id string) (*utterance, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.active[id]
	if !ok || current.interrupted {
		return nil, false
	}
	return current, true
}

// forget takes an utterance out of the session for good, once the server has acknowledged
// the close and there is no more of it to come.
func (t *TTS) forget(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.active, id)
}

func (t *TTS) markClosed(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current, ok := t.active[id]; ok {
		current.closed = true
	}
}

func (t *TTS) closeByID(id string) {
	t.mu.Lock()
	current, ok := t.active[id]
	t.mu.Unlock()
	if !ok {
		return
	}
	_ = t.closeContext(current)
}

func (t *TTS) closeContext(current *utterance) error {
	t.mu.Lock()
	if current.closed {
		t.mu.Unlock()
		return nil
	}
	current.closed = true
	id := current.tracker.ID
	t.mu.Unlock()

	return t.send(clientFrame{ContextID: id, CloseContext: &emptyAction{}})
}

func (t *TTS) resetConnection() {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return
	}
	conn := t.conn
	t.conn = nil
	t.mu.Unlock()

	if conn != nil {
		conn.Close()
	}
	t.settleOutstanding()

	if err := t.dial(context.Background()); err != nil {
		t.logger.Debug("could not reopen after a context limit", "error", err)
		return
	}
	t.logger.Info("reopened the voice after a context limit")
	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
}
