// Package s2pro implements the tts.TTS contract against the streaming S2 Pro deployment
// in acceleration/deploy/s2-pro.
//
// We own both ends, so the wire protocol is the mirror image of the Parakeet one: a JSON
// handshake, then text frames in and binary PCM16 frames out. The deployment accumulates
// deltas per synthesis id and generates on flush, so several utterances can be in flight
// at once and barge-in is a single cancel frame.
package s2pro

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "s2pro"

// DefaultModel is the model the deployment serves.
const DefaultModel = "s2-pro"

// DefaultSampleRate is the rate the deployment generates at: S2 Pro reconstructs through
// the Descript codec, which runs at 44.1 kHz.
const DefaultSampleRate = 44_100

// Frame types, mirroring acceleration/deploy/s2-pro/model/model.py.
const (
	messageReady  = "ready"
	messageFinal  = "final"
	messageError  = "error"
	controlText   = "text"
	controlFlush  = "flush"
	controlCancel = "cancel"
	controlClose  = "close"
)

// Options configures the provider. URL falls back to S2PRO_WS_URL and APIKey to
// BASETEN_API_KEY.
type Options struct {
	URL    string
	APIKey string
	Model  string
	// Voice names a voice the deployment knows. Empty uses the model's own.
	Voice string
	// ReferenceAudio clones a voice from a clip, with ReferenceText as its transcript.
	ReferenceAudio string
	ReferenceText  string
	// SampleRate must match what the deployment generates; the handshake rejects a
	// mismatch rather than letting audio play back at the wrong speed.
	SampleRate int
	// HandshakeTimeout bounds the initial connect. Baseten cold starts can be slow.
	HandshakeTimeout time.Duration
	// DrainTimeout bounds how long Close waits for utterances still being generated.
	DrainTimeout time.Duration
	Logger       *slog.Logger
}

// clientMetadata is the opening frame that describes the audio stream.
type clientMetadata struct {
	SampleRate int    `json:"sample_rate"`
	Encoding   string `json:"encoding"`
}

// controlFrame is a text frame sent to the deployment.
type controlFrame struct {
	Type           string `json:"type"`
	ID             string `json:"id,omitempty"`
	Text           string `json:"text,omitempty"`
	Voice          string `json:"voice,omitempty"`
	ReferenceAudio string `json:"reference_audio,omitempty"`
	ReferenceText  string `json:"reference_text,omitempty"`
}

// serverMessage is a text frame sent by the deployment.
type serverMessage struct {
	Type             string  `json:"type"`
	ID               string  `json:"id"`
	Error            string  `json:"error"`
	Empty            bool    `json:"empty"`
	Cancelled        bool    `json:"cancelled"`
	SampleRate       int     `json:"sample_rate"`
	AudioDurationMs  float64 `json:"audio_duration_ms"`
	ProcessingTimeMs float64 `json:"processing_time_ms"`
}

// TTS is a streaming S2 Pro text-to-speech session.
type TTS struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	mu sync.Mutex
	// speaking tracks utterances the deployment has been asked to generate. Audio frames
	// carry no id, so the order the deployment finals in is what attributes them.
	speaking map[string]*tts.Synthesis
	// order is the ids in the order they were flushed, so audio goes to the oldest.
	order    []string
	started  bool
	shutdown bool

	// drained is closed once nothing is in flight, so Close can wait for the tail.
	drained     chan struct{}
	drainedOnce sync.Once
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.URL == "" {
		options.URL = os.Getenv("S2PRO_WS_URL")
	}
	if options.URL == "" {
		return nil, errors.New("s2pro: websocket url is required (set S2PRO_WS_URL)")
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("s2pro: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv("BASETEN_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("s2pro: api key is required (set BASETEN_API_KEY)")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.SampleRate == 0 {
		options.SampleRate = DefaultSampleRate
	}
	if options.ReferenceText != "" && options.ReferenceAudio == "" {
		return nil, errors.New("s2pro: reference text needs reference audio")
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 60 * time.Second
	}
	if options.DrainTimeout == 0 {
		options.DrainTimeout = 30 * time.Second
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &TTS{
		options:  options,
		logger:   logger.With("provider", ProviderName, "model", options.Model),
		emitter:  tts.NewEmitter(64),
		speaking: map[string]*tts.Synthesis{},
		drained:  make(chan struct{}),
	}, nil
}

// Start dials the deployment and completes the metadata handshake. It returns once the
// server reports it is ready for text.
func (t *TTS) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return errors.New("s2pro: already started")
	}
	t.started = true
	t.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: t.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Api-Key " + t.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, t.options.URL, header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("s2pro: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("s2pro: dial: %w", err)
	}
	t.conn = conn

	if err := t.handshake(); err != nil {
		conn.Close()
		return err
	}

	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.options.Model, At: time.Now()})
	go t.readLoop()
	return nil
}

// Synthesize sends a text delta, and on a final request asks the deployment to say it.
func (t *TTS) Synthesize(request tts.Request) error {
	synthesis, err := t.track(request)
	if err != nil {
		return err
	}

	if request.Text != "" {
		frame := controlFrame{Type: controlText, ID: synthesis.ID, Text: request.Text}
		if err := t.writeFrame(frame); err != nil {
			return fmt.Errorf("s2pro: send text: %w", err)
		}
		synthesis.AddText(request.Text)
	}
	if !request.Final {
		return nil
	}

	t.flush(synthesis.ID)
	if err := t.writeFrame(t.flushFrame(synthesis.ID, request)); err != nil {
		t.forget(synthesis.ID)
		return fmt.Errorf("s2pro: flush: %w", err)
	}

	t.emitter.Send(tts.SynthesisStarted{
		SynthesisID: synthesis.ID,
		Provider:    ProviderName,
		Model:       t.options.Model,
		Voice:       t.voiceFor(request),
		At:          time.Now(),
	})
	return nil
}

// Interrupt cancels every utterance in flight. The deployment stops generating and
// finals each one as cancelled, so the caller still gets a complete accounting.
func (t *TTS) Interrupt() error {
	t.mu.Lock()
	ids := append([]string(nil), t.order...)
	t.mu.Unlock()

	for _, id := range ids {
		if err := t.writeFrame(controlFrame{Type: controlCancel, ID: id}); err != nil {
			return fmt.Errorf("s2pro: cancel: %w", err)
		}
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close waits for utterances still being generated, then tears down the connection.
func (t *TTS) Close() error {
	t.mu.Lock()
	if t.shutdown {
		t.mu.Unlock()
		return nil
	}
	t.shutdown = true
	conn := t.conn
	pending := len(t.order)
	t.mu.Unlock()

	if conn != nil {
		if pending > 0 {
			select {
			case <-t.drained:
			case <-time.After(t.options.DrainTimeout):
				t.logger.Debug("timed out waiting for audio in flight")
			}
		}
		// A dead connection must not stop teardown, so the close frame is best effort.
		if err := t.writeFrame(controlFrame{Type: controlClose}); err != nil {
			t.logger.Debug("close frame not delivered", "error", err)
		}
		conn.Close()
	}

	// Whatever the deployment never finalled is still work the caller asked for, so it is
	// settled as interrupted rather than left unaccounted for.
	t.settleRemaining()

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

// Streaming reports true: the deployment accumulates deltas and generates on flush.
func (t *TTS) Streaming() bool { return true }

// Performs reports false: the deployment reads a bracketed direction out as words.
func (t *TTS) Performs() bool { return false }

// Prompt reports nothing: there is no direction this voice would act.
func (t *TTS) Prompt() string { return "" }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.options.SampleRate }

// Client exposes the underlying WebSocket so callers can use the deployment directly.
func (t *TTS) Client() *websocket.Conn { return t.conn }

// handshake sends the audio metadata and waits for the server to acknowledge it.
func (t *TTS) handshake() error {
	payload, err := json.Marshal(clientMetadata{SampleRate: t.options.SampleRate, Encoding: "linear16"})
	if err != nil {
		return fmt.Errorf("s2pro: encode metadata: %w", err)
	}
	if err := t.write(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("s2pro: send metadata: %w", err)
	}

	_, raw, err := t.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("s2pro: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("s2pro: decode handshake: %w", err)
	}
	if message.Type == messageError {
		return fmt.Errorf("s2pro: handshake rejected: %s", message.Error)
	}
	if message.Type != messageReady {
		return fmt.Errorf("s2pro: expected %q, got %q", messageReady, message.Type)
	}
	if message.SampleRate != 0 && message.SampleRate != t.options.SampleRate {
		return fmt.Errorf(
			"s2pro: deployment generates at %d Hz, session wants %d",
			message.SampleRate, t.options.SampleRate,
		)
	}
	return nil
}

// track returns the tracker for a request's utterance, creating it on first sight.
func (t *TTS) track(request tts.Request) (*tts.Synthesis, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return nil, errors.New("s2pro: session closed")
	}
	if !t.started || t.conn == nil {
		return nil, errors.New("s2pro: not started")
	}
	// A partial with no id could not be matched to its continuation, so it is a caller
	// error rather than something to silently drop.
	if !request.Final && request.ID == "" {
		return nil, errors.New("s2pro: a partial request needs an id")
	}

	synthesis := t.speaking[request.ID]
	if synthesis == nil {
		synthesis = tts.NewSynthesis(request.ID)
		t.speaking[synthesis.ID] = synthesis
	}
	return synthesis, nil
}

// flush moves an utterance into the queue that incoming audio is attributed to.
func (t *TTS) flush(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.order = append(t.order, id)
}

// forget drops an utterance that never made it to the deployment.
func (t *TTS) forget(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	delete(t.speaking, id)
	for i, queued := range t.order {
		if queued == id {
			t.order = append(t.order[:i], t.order[i+1:]...)
			break
		}
	}
}

func (t *TTS) flushFrame(id string, request tts.Request) controlFrame {
	return controlFrame{
		Type:           controlFlush,
		ID:             id,
		Voice:          t.voiceFor(request),
		ReferenceAudio: t.options.ReferenceAudio,
		ReferenceText:  t.options.ReferenceText,
	}
}

// voiceFor lets a request override the session's voice, which the deployment allows
// because the voice is chosen per flush.
func (t *TTS) voiceFor(request tts.Request) string {
	if request.Voice != "" {
		return request.Voice
	}
	return t.options.Voice
}

func (t *TTS) writeFrame(frame controlFrame) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	return t.write(websocket.TextMessage, payload)
}

func (t *TTS) write(messageType int, payload []byte) error {
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	if t.conn == nil {
		return errors.New("not connected")
	}
	return t.conn.WriteMessage(messageType, payload)
}

// readLoop turns server frames into events until the connection ends.
func (t *TTS) readLoop() {
	for {
		messageType, raw, err := t.conn.ReadMessage()
		if err != nil {
			t.handleReadError(err)
			return
		}

		if messageType == websocket.BinaryMessage {
			t.handleAudio(raw)
			continue
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			t.logger.Debug("undecodable frame", "error", err, "payload", string(raw))
			continue
		}
		t.handleMessage(message)
	}
}

func (t *TTS) handleReadError(err error) {
	t.mu.Lock()
	closed := t.shutdown
	t.mu.Unlock()
	if closed {
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

// handleAudio attributes a binary frame to the oldest utterance in flight. The deployment
// generates one utterance at a time in flush order, so the head of the queue owns it.
func (t *TTS) handleAudio(raw []byte) {
	t.mu.Lock()
	var synthesis *tts.Synthesis
	if len(t.order) > 0 {
		synthesis = t.speaking[t.order[0]]
	}
	t.mu.Unlock()

	if synthesis == nil {
		t.logger.Debug("audio arrived with nothing in flight", "bytes", len(raw))
		return
	}
	t.emitter.Send(synthesis.Chunk(audio.FromBytes(raw, t.options.SampleRate, 1)))
}

func (t *TTS) handleMessage(message serverMessage) {
	switch message.Type {
	case messageFinal:
		t.complete(message)
	case messageError:
		t.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       t.options.Model,
			SynthesisID: message.ID,
			Err:         errors.New(message.Error),
			Context:     "server",
			Fatal:       message.ID == "",
		})
		if message.ID != "" {
			t.complete(serverMessage{ID: message.ID, Cancelled: true})
		}
	default:
		t.logger.Debug("unhandled frame", "type", message.Type)
	}
}

// settleRemaining reports every utterance the deployment never finalled as interrupted.
func (t *TTS) settleRemaining() {
	t.mu.Lock()
	remaining := make([]*tts.Synthesis, 0, len(t.speaking))
	for _, synthesis := range t.speaking {
		remaining = append(remaining, synthesis)
	}
	clear(t.speaking)
	t.order = nil
	t.mu.Unlock()

	for _, synthesis := range remaining {
		t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, true))
	}
}

// complete settles an utterance and signals Close once nothing is left in flight.
func (t *TTS) complete(message serverMessage) {
	t.mu.Lock()
	synthesis := t.speaking[message.ID]
	delete(t.speaking, message.ID)
	for i, queued := range t.order {
		if queued == message.ID {
			t.order = append(t.order[:i], t.order[i+1:]...)
			break
		}
	}
	remaining := len(t.order)
	shutdown := t.shutdown
	t.mu.Unlock()

	if synthesis != nil {
		t.emitter.Send(synthesis.Complete(ProviderName, t.options.Model, message.Cancelled))
	}
	if remaining == 0 && shutdown {
		t.drainedOnce.Do(func() { close(t.drained) })
	}
}
