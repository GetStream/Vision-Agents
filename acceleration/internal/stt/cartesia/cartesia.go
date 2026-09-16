// Package cartesia implements the stt.STT contract on top of Cartesia's Ink 2 over the
// realtime socket that detects turns for itself.
//
// Cartesia serve two realtime endpoints for the same model. This is the one with turn
// detection built in, which is what a call wants: the alternative leaves the boundary to
// the caller and settles only what it is explicitly told to finalise, and the router has
// no voice activity detector of its own to tell it with.
//
// The session is configured entirely by the query string, so there is no setup frame and
// nothing to wait for before sending audio. Audio goes up as raw bytes. What comes back is
// a turn lifecycle rather than a stream of guesses: turn.start when the caller begins,
// turn.update repeatedly with the turn's text so far, and turn.end with the text the turn
// settles on. The transcript is cumulative within a turn and the model never revises what
// it has already emitted, so each update is a replacement rather than a delta.
//
// turn.eager_end is the one event that needs care. It fires when the model thinks the
// caller may be finished, and turn.resume follows when they were not, so it cannot settle
// a turn: reporting it as final is how the first half of a question reaches an agent as
// the whole of it. It goes out as a replacement, which is what it is.
package cartesia

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "cartesia"

// DefaultModel is Ink 2, Cartesia's streaming model. It is English-only.
const DefaultModel = "ink-2"

// DefaultURL is the realtime socket that detects turns for itself.
const DefaultURL = "wss://api.cartesia.ai/stt/turns/websocket"

// APIVersion is the only version this endpoint accepts. Cartesia date their API and
// refuse a request that does not name one.
const APIVersion = "2026-03-01"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "CARTESIA_API_KEY"

// apiKeyHeader carries the credentials. The query string takes a short-lived token
// instead, which is for a browser rather than for the router.
const apiKeyHeader = "X-API-Key"

// encodingPCM16 matches stt.SampleRate and the samples stt.PcmData holds.
const encodingPCM16 = "pcm_s16le"

// maxKeytermChars is the total budget across all the terms, which the endpoint enforces
// separately from the count.
const maxKeytermChars = 1200

// Server event types.
const (
	eventConnected    = "connected"
	eventTurnStart    = "turn.start"
	eventTurnUpdate   = "turn.update"
	eventTurnEagerEnd = "turn.eager_end"
	eventTurnResume   = "turn.resume"
	eventTurnEnd      = "turn.end"
	eventError        = "error"
)

// controlTypeClose tells the server the audio has stopped, which makes it transcribe what
// it is still holding, emit the remaining turn events and hang up.
const controlTypeClose = "close"

// Options configures the provider. APIKey falls back to CARTESIA_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong. The endpoint takes up
	// to a hundred of them, totalling maxKeytermChars characters.
	Keyterms []string
	// TurnEndTimeoutMs is how long the model waits after the caller stops speaking before
	// ending the turn. Zero leaves the server's own default.
	TurnEndTimeoutMs int
	// TurnStartThreshold, TurnEagerEndThreshold and TurnEndThreshold are how sure the
	// model has to be that a turn has begun, may have ended, and has ended. Zero leaves
	// each to the server. They are ordered: start above eager end above end.
	TurnStartThreshold    float64
	TurnEagerEndThreshold float64
	TurnEndThreshold      float64
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// serverMessage is an event sent by the server.
type serverMessage struct {
	Type string `json:"type"`
	// Transcript is the turn's text so far, on the three events that carry text. It
	// restates the turn from its beginning rather than carrying only what is new.
	Transcript string `json:"transcript"`
	RequestID  string `json:"request_id"`
	// The remaining fields are only on an error.
	ErrorCode  string `json:"error_code"`
	StatusCode int    `json:"status_code"`
	Title      string `json:"title"`
	Message    string `json:"message"`
}

// failure is what the server said went wrong, as a sentence rather than a struct.
func (m serverMessage) failure() string {
	described := strings.TrimSpace(m.Title + ": " + m.Message)
	switch {
	case described == ":":
		return "unknown error"
	case m.ErrorCode != "":
		return described + " (" + m.ErrorCode + ")"
	default:
		return described
	}
}

// STT is one Cartesia Ink 2 session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// finished is closed when the server has answered the close command and hung up,
	// which is how Close knows the tail has been transcribed.
	finished     chan struct{}
	finishedOnce sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// utterance counts the turns the server has announced, so a caller can tell the model
	// going back over words from the caller saying them again.
	utterance int64
	started   bool
	closed    bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("cartesia: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("cartesia: url must be ws:// or wss://, got %s", options.URL)
	}
	options.Keyterms = stt.CleanKeyterms(options.Keyterms)
	if len(options.Keyterms) > stt.MaxKeyterms {
		return nil, fmt.Errorf("cartesia: at most %d keyterms, got %d", stt.MaxKeyterms, len(options.Keyterms))
	}
	if spelt := keytermChars(options.Keyterms); spelt > maxKeytermChars {
		return nil, fmt.Errorf("cartesia: keyterms total at most %d characters, got %d",
			maxKeytermChars, spelt)
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 15 * time.Second
	}
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 10 * time.Second
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STT{
		options:  options,
		logger:   logger.With("provider", ProviderName, "model", options.Model),
		emitter:  stt.NewEmitter(64),
		finished: make(chan struct{}),
	}, nil
}

// Start dials the socket. The session is configured by the query string, so there is no
// setup frame and no acknowledgement to wait for before sending audio.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("cartesia: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{apiKeyHeader: []string{s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("cartesia: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("cartesia: dial: %w", err)
	}
	s.conn = conn

	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop()
	return nil
}

// ProcessAudio streams one chunk of audio. The participant labels any transcript that
// results from it.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("cartesia: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("cartesia: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("cartesia: not started")
	}

	if err := s.write(websocket.BinaryMessage, pcm.Bytes()); err != nil {
		return fmt.Errorf("cartesia: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close asks the server to transcribe the audio it has buffered, waits for it to finish
// and hang up, then tears the connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()

	if conn != nil {
		s.flush()
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// Client exposes the underlying WebSocket so callers can use the session directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// endpoint is the socket with the session's configuration attached, which is the only
// place this API takes it.
func (s *STT) endpoint() string {
	query := url.Values{}
	query.Set("model", s.options.Model)
	query.Set("encoding", encodingPCM16)
	query.Set("sample_rate", strconv.Itoa(stt.SampleRate))
	query.Set("cartesia_version", APIVersion)
	if s.options.TurnEndTimeoutMs > 0 {
		query.Set("turn_end_timeout_ms", strconv.Itoa(s.options.TurnEndTimeoutMs))
	}
	if s.options.TurnStartThreshold > 0 {
		query.Set("turn_start_threshold", threshold(s.options.TurnStartThreshold))
	}
	if s.options.TurnEagerEndThreshold > 0 {
		query.Set("turn_eager_end_threshold", threshold(s.options.TurnEagerEndThreshold))
	}
	if s.options.TurnEndThreshold > 0 {
		query.Set("turn_end_threshold", threshold(s.options.TurnEndThreshold))
	}
	// Repeated rather than joined: the parameter is singular and one term per copy is how
	// this endpoint reads a list.
	for _, term := range s.options.Keyterms {
		query.Add("keyterm", term)
	}

	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + query.Encode()
}

// flush tells the server the audio has stopped and waits for it to hang up, which it does
// once it has emitted the events for everything it was still holding. A dead connection
// must not stop teardown.
func (s *STT) flush() {
	if err := s.write(websocket.TextMessage, []byte(`{"type":"`+controlTypeClose+`"}`)); err != nil {
		s.logger.Debug("close command not delivered", "error", err)
		return
	}

	select {
	case <-s.finished:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the last words")
	}
}

func (s *STT) write(messageType int, payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(messageType, payload)
}

// readLoop translates server events into stt events until the connection ends.
func (s *STT) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			s.finishedOnce.Do(func() { close(s.finished) })
			s.handleReadError(err)
			return
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			s.logger.Debug("undecodable frame", "error", err, "payload", string(raw))
			continue
		}
		s.handleMessage(message)
	}
}

func (s *STT) handleReadError(err error) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return
	}

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(stt.Disconnected{
			Provider: ProviderName,
			Model:    s.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		return
	}
	s.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    s.options.Model,
		Err:      err,
		Context:  "read",
		Fatal:    true,
	})
}

func (s *STT) handleMessage(message serverMessage) {
	switch message.Type {
	case eventTurnStart:
		s.startTurn()
	case eventTurnUpdate:
		s.sendTranscript(message.Transcript, stt.ModeReplacement)
	case eventTurnEagerEnd:
		// A guess that the caller may be done, which turn.resume takes back when they
		// were not. The text is worth having early; the turn is not over.
		s.sendTranscript(message.Transcript, stt.ModeReplacement)
	case eventTurnResume:
		// The turn the eager end guessed at carries on, and the updates that follow
		// restate it, so there is nothing to undo here.
	case eventTurnEnd:
		s.sendTranscript(message.Transcript, stt.ModeFinal)
	case eventError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  "server",
			Fatal:    true,
		})
	case eventConnected:
		// Audio may be sent before this arrives, so there is nothing to do with it.
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

func (s *STT) sendTranscript(transcript string, mode stt.Mode) {
	text := strings.TrimSpace(transcript)
	if text == "" {
		return
	}

	participant, utterance, latencyMs := s.snapshot()
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Utterance:        utterance,
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// startTurn numbers the run of speech the server has just announced.
func (s *STT) startTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.utterance++
}

// snapshot returns the current speaker, the turn being spoken and how long ago audio was
// last sent.
//
// The turn is numbered here as well as on turn.start, because the audio buffered when the
// session is closed is transcribed into a turn whose start the server never announced.
func (s *STT) snapshot() (stt.Participant, int64, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 {
		s.utterance = 1
	}
	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	return s.participant, s.utterance, latencyMs
}

// keytermChars is how much of the character budget a list of terms spends.
func keytermChars(terms []string) int {
	spelt := 0
	for _, term := range terms {
		spelt += len(term)
	}
	return spelt
}

// threshold renders a turn detection threshold the way the query string takes it, without
// the exponent or the trailing zeroes a default format would add.
func threshold(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
