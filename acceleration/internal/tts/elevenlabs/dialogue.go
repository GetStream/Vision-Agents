package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// DefaultDialogueModel is the v3 model meant for conversation. The other one, eleven_v3,
// is for scripted dialogue between several speakers.
const DefaultDialogueModel = "eleven_v3_conversational"

// dialoguePath is the endpoint that serves the v3 models. It is a different protocol from
// the one above, not another model on the same socket.
const dialoguePath = "/v1/text-to-dialogue/stream-input"

// dialogueKeepAlive is how often to remind the server the session is still wanted. It
// hangs up after twenty seconds of hearing nothing, and a caller who spends that long
// thinking is ordinary.
const dialogueKeepAlive = 12 * time.Second

// dialogueStability is how much the voice may vary between generations. The v3 models
// take no other setting, and a lower value is what gives an audio tag room to be acted.
const dialogueStability = 0.5

// dialogueInput is one speaker's line.
type dialogueInput struct {
	Text    string `json:"text"`
	VoiceID string `json:"voice_id"`
	// NewTurn finishes the segment before this one, which is what stops two utterances
	// running into each other as one breath.
	NewTurn bool `json:"new_turn,omitempty"`
}

// dialogueVoiceSettings is the only setting the v3 models accept, and only on the first
// message.
type dialogueVoiceSettings struct {
	Stability float64 `json:"stability"`
}

// dialogueClientMessage is a frame sent upstream.
type dialogueClientMessage struct {
	Inputs        []dialogueInput        `json:"inputs,omitempty"`
	Voices        []string               `json:"voices,omitempty"`
	VoiceSettings *dialogueVoiceSettings `json:"voice_settings,omitempty"`
	Flush         bool                   `json:"flush,omitempty"`
	CloseSocket   bool                   `json:"close_socket,omitempty"`
	KeepAlive     bool                   `json:"keep_alive,omitempty"`
}

// dialogueServerMessage is a frame sent back. The fields are snake_case here, unlike the
// text-to-speech socket, and none of them name the utterance they belong to.
type dialogueServerMessage struct {
	Audio               string `json:"audio"`
	IsFinalAudioForTurn bool   `json:"is_final_audio_for_turn"`
	IsFinal             bool   `json:"is_final"`
	Message             string `json:"message"`
	Error               string `json:"error"`
	Code                int    `json:"code"`
}

// Dialogue speaks through the text-to-dialogue socket, which is what the v3 models are
// served on.
//
// It exists because the v3 models are the only ones that act audio tags, and they are not
// available on the socket the rest of the provider uses. The protocol is a good deal
// plainer: there are no server-side contexts, so one connection carries one conversation
// and audio is attributed by the order turns were started in, the way s2pro does it.
// Barge-in has no cancel frame either, so it replaces the socket.
type Dialogue struct {
	options Options
	logger  *slog.Logger
	emitter *tts.Emitter

	// done stops the keepalive when the session ends.
	done     chan struct{}
	doneOnce sync.Once
	// drained closes once nothing is in flight, so Close can wait for the tail.
	drained     chan struct{}
	drainedOnce sync.Once

	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	mu   sync.Mutex
	conn *websocket.Conn
	// speaking holds the utterances the server has been given text for.
	speaking map[string]*tts.Synthesis
	// order is those utterances in the order they were started. Audio carries no id, so
	// the oldest unfinished one owns it, and the end of a turn moves the queue on.
	order []string
	// turns counts the turns opened since the socket was, so the first one does not ask
	// to end a turn that was never started.
	turns    int
	started  bool
	shutdown bool
}

// NewDialogue validates the options and returns an unstarted dialogue provider.
func NewDialogue(options Options) (*Dialogue, error) {
	options, logger, err := normalize(options, DefaultDialogueModel)
	if err != nil {
		return nil, err
	}
	if !Performs(options.Model) {
		return nil, fmt.Errorf(
			"elevenlabs: %s is not a dialogue model, open it with New", options.Model)
	}

	return &Dialogue{
		options:  options,
		logger:   logger.With("provider", ProviderName, "model", options.Model),
		emitter:  tts.NewEmitter(64),
		done:     make(chan struct{}),
		drained:  make(chan struct{}),
		speaking: map[string]*tts.Synthesis{},
	}, nil
}

// Start dials the socket and registers the session's voice.
func (d *Dialogue) Start(ctx context.Context) error {
	d.mu.Lock()
	if d.started {
		d.mu.Unlock()
		return errors.New("elevenlabs: already started")
	}
	d.started = true
	d.mu.Unlock()

	if err := d.dial(ctx); err != nil {
		return err
	}

	d.emitter.Send(tts.Connected{Provider: ProviderName, Model: d.options.Model, At: time.Now()})
	go d.keepAlive()
	return nil
}

// Synthesize sends text upstream. Several requests sharing an ID stream one utterance, and
// the one with Final set flushes it so the tail is generated at once.
func (d *Dialogue) Synthesize(request tts.Request) error {
	if request.Voice != "" && request.Voice != d.options.VoiceID {
		return fmt.Errorf(
			"elevenlabs: the connection is bound to voice %s, open a new session for %s",
			d.options.VoiceID, request.Voice)
	}

	synthesis, opened, newTurn, err := d.track(request)
	if err != nil {
		return err
	}

	if opened {
		d.emitter.Send(tts.SynthesisStarted{
			SynthesisID: synthesis.ID,
			Provider:    ProviderName,
			Model:       d.options.Model,
			Voice:       d.options.VoiceID,
			At:          time.Now(),
		})
	}

	if request.Text != "" {
		input := dialogueInput{
			Text:    request.Text + " ",
			VoiceID: d.options.VoiceID,
			NewTurn: newTurn,
		}
		if err := d.send(dialogueClientMessage{Inputs: []dialogueInput{input}}); err != nil {
			d.forget(synthesis.ID)
			return fmt.Errorf("elevenlabs: send text: %w", err)
		}
		synthesis.AddText(request.Text)
	}
	if !request.Final {
		return nil
	}

	// The server buffers until it has enough words to say something well, so an utterance
	// that has ended has to ask for the rest of it.
	if err := d.send(dialogueClientMessage{Flush: true}); err != nil {
		return fmt.Errorf("elevenlabs: flush: %w", err)
	}
	return nil
}

// Interrupt drops what is being said. There is no cancel frame on this protocol, so the
// socket is replaced: the alternative is to keep receiving audio nobody will hear, and to
// go on paying for it while the caller is already talking.
func (d *Dialogue) Interrupt() error {
	d.mu.Lock()
	if d.shutdown || !d.started {
		d.mu.Unlock()
		return nil
	}
	conn := d.conn
	d.conn = nil
	abandoned := d.takeRemaining()
	d.turns = 0
	d.mu.Unlock()

	for _, synthesis := range abandoned {
		d.emitter.Send(synthesis.Complete(ProviderName, d.options.Model, true))
	}
	if conn != nil {
		conn.Close()
	}

	// The call is still going, so the voice is opened again rather than left for whoever
	// next tries to speak through it.
	if err := d.dial(context.Background()); err != nil {
		return fmt.Errorf("elevenlabs: reopen after interrupt: %w", err)
	}
	return nil
}

// Events returns audio and synthesis boundaries.
func (d *Dialogue) Events() <-chan tts.Event { return d.emitter.Events() }

// Close asks the server for the audio it still owes, then tears down the connection.
func (d *Dialogue) Close() error {
	d.mu.Lock()
	if d.shutdown {
		d.mu.Unlock()
		return nil
	}
	d.shutdown = true
	conn := d.conn
	pending := len(d.order)
	d.mu.Unlock()

	d.doneOnce.Do(func() { close(d.done) })

	if conn != nil {
		// A dead connection must not stop teardown, so this is best effort.
		if err := d.send(dialogueClientMessage{CloseSocket: true}); err != nil {
			d.logger.Debug("close frame not delivered", "error", err)
		} else if pending > 0 {
			select {
			case <-d.drained:
			case <-time.After(d.options.CloseTimeout):
				d.logger.Debug("timed out waiting for audio in flight")
			}
		}
		conn.Close()
	}

	// Whatever the server never finished is still work the caller asked for, so it is
	// settled as interrupted rather than left unaccounted for.
	for _, synthesis := range d.remaining() {
		d.emitter.Send(synthesis.Complete(ProviderName, d.options.Model, true))
	}

	d.emitter.Send(tts.Disconnected{
		Provider: ProviderName,
		Model:    d.options.Model,
		Clean:    true,
		At:       time.Now(),
	})
	d.emitter.Close()
	return nil
}

// Provider implements tts.TTS.
func (d *Dialogue) Provider() string { return ProviderName }

// Model implements tts.TTS.
func (d *Dialogue) Model() string { return d.options.Model }

// Streaming reports true: the model generates from partial text.
func (d *Dialogue) Streaming() bool { return true }

// Performs reports true: acting audio tags is what this endpoint is for.
func (d *Dialogue) Performs() bool { return true }

// Prompt tells the model writing the lines that it may direct the delivery.
func (d *Dialogue) Prompt() string { return AudioTagPrompt }

// SampleRate is the rate the audio comes back at.
func (d *Dialogue) SampleRate() int { return d.options.SampleRate }

// Client exposes the underlying WebSocket so callers can use the socket directly.
func (d *Dialogue) Client() *websocket.Conn {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.conn
}

// url is the endpoint the session connects to.
func (d *Dialogue) url() string {
	query := url.Values{}
	query.Set("model_id", d.options.Model)
	query.Set("output_format", "pcm_"+strconv.Itoa(d.options.SampleRate))
	if d.options.Language != "" {
		query.Set("language_code", strings.ToLower(d.options.Language))
	}
	return d.options.BaseURL + dialoguePath + "?" + query.Encode()
}

// dial opens a connection, registers the voice and starts reading.
func (d *Dialogue) dial(ctx context.Context) error {
	dialer := &websocket.Dialer{HandshakeTimeout: d.options.HandshakeTimeout}
	header := http.Header{"xi-api-key": []string{d.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, d.url(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("elevenlabs: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("elevenlabs: dial: %w", err)
	}

	d.mu.Lock()
	d.conn = conn
	d.mu.Unlock()

	// The session is bound to one voice, which the first frame has to register before any
	// line can be sent.
	opening := dialogueClientMessage{
		Voices:        []string{d.options.VoiceID},
		VoiceSettings: &dialogueVoiceSettings{Stability: dialogueStability},
	}
	if err := d.send(opening); err != nil {
		d.mu.Lock()
		d.conn = nil
		d.mu.Unlock()
		conn.Close()
		return fmt.Errorf("elevenlabs: register voice: %w", err)
	}

	go d.readLoop(conn)
	return nil
}

// keepAlive stops the server hanging up on a caller who has not spoken for a while.
func (d *Dialogue) keepAlive() {
	ticker := time.NewTicker(dialogueKeepAlive)
	defer ticker.Stop()

	for {
		select {
		case <-d.done:
			return
		case <-ticker.C:
			if err := d.send(dialogueClientMessage{KeepAlive: true}); err != nil {
				d.logger.Debug("keepalive not delivered", "error", err)
			}
		}
	}
}

// track returns the tracker for a request's utterance, whether this is the first text of
// it, and whether sending that text should end the turn before it. Only the first line of
// an utterance ends the previous one: asking again mid-utterance would cut it into a turn
// per delta, and the first utterance on a connection has none to end.
func (d *Dialogue) track(request tts.Request) (*tts.Synthesis, bool, bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.shutdown {
		return nil, false, false, errors.New("elevenlabs: session closed")
	}
	if !d.started || d.conn == nil {
		return nil, false, false, errors.New("elevenlabs: not started")
	}
	// A partial with no id could not be matched to its continuation, so it is a caller
	// error rather than something to silently drop.
	if !request.Final && request.ID == "" {
		return nil, false, false, errors.New("elevenlabs: a partial request needs an id")
	}

	if request.ID != "" {
		if existing, ok := d.speaking[request.ID]; ok {
			return existing, false, false, nil
		}
	}

	synthesis := tts.NewSynthesis(request.ID)
	d.speaking[synthesis.ID] = synthesis
	d.order = append(d.order, synthesis.ID)
	d.turns++
	return synthesis, true, d.turns > 1, nil
}

// forget drops an utterance whose text never reached the server.
func (d *Dialogue) forget(id string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	delete(d.speaking, id)
	for i, queued := range d.order {
		if queued == id {
			d.order = append(d.order[:i], d.order[i+1:]...)
			break
		}
	}
}

// remaining takes every utterance still in flight, under the lock.
func (d *Dialogue) remaining() []*tts.Synthesis {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.takeRemaining()
}

// takeRemaining is remaining for a caller that already holds the lock.
func (d *Dialogue) takeRemaining() []*tts.Synthesis {
	abandoned := make([]*tts.Synthesis, 0, len(d.order))
	for _, id := range d.order {
		if synthesis, ok := d.speaking[id]; ok {
			abandoned = append(abandoned, synthesis)
		}
	}
	clear(d.speaking)
	d.order = nil
	return abandoned
}

func (d *Dialogue) send(message dialogueClientMessage) error {
	payload, err := json.Marshal(message)
	if err != nil {
		return err
	}

	d.writeMu.Lock()
	defer d.writeMu.Unlock()

	d.mu.Lock()
	conn := d.conn
	d.mu.Unlock()
	if conn == nil {
		return errors.New("not connected")
	}
	return conn.WriteMessage(websocket.TextMessage, payload)
}

// readLoop turns server frames into events until this connection ends. It takes the
// connection rather than reading the field, so a loop left over from a socket that
// barge-in replaced cannot report the new one's audio.
func (d *Dialogue) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			if !d.current(conn) {
				return
			}
			d.handleReadError(err)
			// The connection is gone, so nothing in flight will ever finish on its own.
			for _, synthesis := range d.remaining() {
				d.emitter.Send(synthesis.Complete(ProviderName, d.options.Model, true))
			}
			return
		}

		var message dialogueServerMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			d.logger.Debug("undecodable frame", "error", err)
			continue
		}
		if d.current(conn) {
			d.handleMessage(message)
		}
	}
}

// current reports whether a connection is still the session's own.
func (d *Dialogue) current(conn *websocket.Conn) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.conn == conn
}

func (d *Dialogue) handleReadError(err error) {
	d.mu.Lock()
	shutdown := d.shutdown
	d.mu.Unlock()
	if shutdown {
		return
	}

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		d.emitter.Send(tts.Disconnected{
			Provider: ProviderName,
			Model:    d.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		return
	}
	d.emitter.Send(tts.Error{
		Provider: ProviderName,
		Model:    d.options.Model,
		Err:      err,
		Context:  "read",
		Fatal:    true,
	})
}

func (d *Dialogue) handleMessage(message dialogueServerMessage) {
	if message.Error != "" || message.Message != "" {
		d.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       d.options.Model,
			SynthesisID: d.oldest(),
			Err:         errors.New(failureText(message)),
			Context:     "server",
			// The server states the close code it is about to send, so an error that
			// carries one is the end of the connection rather than of one utterance.
			Fatal: message.Code != 0,
		})
		return
	}

	if message.Audio != "" {
		d.handleAudio(message.Audio)
	}
	if message.IsFinalAudioForTurn {
		d.completeOldest()
	}
	// The closing flush is over, so anything the server still owed is not coming.
	if message.IsFinal {
		d.drainedOnce.Do(func() { close(d.drained) })
	}
}

// handleAudio attributes a frame to the oldest utterance in flight. The server works
// through turns in the order they were started, so the head of the queue owns it.
func (d *Dialogue) handleAudio(encoded string) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		d.emitter.Send(tts.Error{
			Provider:    ProviderName,
			Model:       d.options.Model,
			SynthesisID: d.oldest(),
			Err:         fmt.Errorf("decode audio: %w", err),
			Context:     "audio",
		})
		return
	}

	d.mu.Lock()
	var synthesis *tts.Synthesis
	if len(d.order) > 0 {
		synthesis = d.speaking[d.order[0]]
	}
	d.mu.Unlock()

	if synthesis == nil {
		d.logger.Debug("audio arrived with nothing in flight", "bytes", len(raw))
		return
	}
	d.emitter.Send(synthesis.Chunk(audio.FromBytes(raw, d.options.SampleRate, 1)))
}

// oldest names the utterance a frame that carries no id belongs to.
func (d *Dialogue) oldest() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.order) == 0 {
		return ""
	}
	return d.order[0]
}

// completeOldest settles the utterance the server has finished, and signals Close once
// nothing is left in flight.
func (d *Dialogue) completeOldest() {
	d.mu.Lock()
	if len(d.order) == 0 {
		d.mu.Unlock()
		return
	}
	id := d.order[0]
	synthesis := d.speaking[id]
	delete(d.speaking, id)
	d.order = d.order[1:]
	remaining := len(d.order)
	shutdown := d.shutdown
	d.mu.Unlock()

	if synthesis != nil {
		d.emitter.Send(synthesis.Complete(ProviderName, d.options.Model, false))
	}
	if remaining == 0 && shutdown {
		d.drainedOnce.Do(func() { close(d.drained) })
	}
}

// failureText is what the server said went wrong, preferring the readable half.
func failureText(message dialogueServerMessage) string {
	if message.Message != "" {
		return message.Message
	}
	return message.Error
}
