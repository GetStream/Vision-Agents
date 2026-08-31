// Package grok implements the stt.STT contract on top of xAI's streaming Speech to Text.
//
// The session is configured by the query string it is opened with rather than by a setup
// frame. Audio goes up as raw binary and transcripts come back as JSON.
//
// What comes back arrives in three shapes, and only the last of them settles a turn. An
// interim may still change. A chunk final locks the words behind it while the speaker
// carries on, so it is text that will not change rather than a turn that is over. An
// utterance final says the speaker stopped, and is what the rest of the call is built on.
//
// All three restate the utterance from its beginning rather than carrying only what is
// new, so the first two are replacements. That matters more than it looks: the server
// revises words it has already reported, turning "young Mia" into "young Mira" and back
// again as more of the sentence arrives, so a transcript assembled by appending would
// keep every guess it ever made.
package grok

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
const ProviderName = "grok"

// DefaultModel is the only speech-to-text model xAI serves. It is named here because the
// endpoint does not take a model parameter and stats still have to say what was listening.
const DefaultModel = "grok-stt"

// DefaultURL is the streaming Speech to Text socket.
const DefaultURL = "wss://api.x.ai/v1/stt"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "XAI_API_KEY"

// Server event types.
const (
	eventCreated = "transcript.created"
	eventPartial = "transcript.partial"
	eventDone    = "transcript.done"
	eventError   = "error"
)

// controlTypeAudioDone tells the server the audio has stopped, which makes it transcribe
// what it is still holding and report it as transcript.done.
const controlTypeAudioDone = "audio.done"

// Options configures the provider. APIKey falls back to XAI_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong, sent as the terms to
	// bias recognition toward. The API takes up to a hundred, which is stt.MaxKeyterms.
	Keyterms []string
	// Language is a code such as "en" or "de". It does not constrain recognition: it
	// turns on the written form of numbers, currencies and units. Empty leaves them as
	// they were spoken.
	Language string
	// EndpointingMs is the silence that ends an utterance. Zero leaves the server's own
	// 400ms in place.
	EndpointingMs int
	// SmartTurn is the confidence, between 0 and 1, at which a silence is taken for the
	// end of a thought rather than a pause. Zero leaves it off, and endpointing alone
	// decides, which cuts off a caller who pauses mid-sentence.
	SmartTurn float64
	// SmartTurnTimeoutMs forces the turn to end after this much silence even when Smart
	// Turn is still predicting the caller has more to say. Only used with SmartTurn.
	SmartTurnTimeoutMs int
	// HandshakeTimeout bounds the initial connect and the wait for the server to be ready.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// serverMessage is a frame sent by the server.
type serverMessage struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// IsFinal says the text will not change. SpeechFinal says the speaker stopped, which
	// is the only one of the two that ends a turn.
	IsFinal     bool `json:"is_final"`
	SpeechFinal bool `json:"speech_final"`
	// Duration is the audio this transcript covers, in seconds.
	Duration float64 `json:"duration"`
	// Language is the language the server recognised, whatever it was asked to expect.
	Language string `json:"language"`
	// EndOfTurnConfidence is only sent when Smart Turn is on.
	EndOfTurnConfidence float64 `json:"end_of_turn_confidence"`
	Message             string  `json:"message"`
}

// STT is an xAI streaming speech-to-text session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// finished is closed when the server reports the transcript of the last of the audio.
	finished     chan struct{}
	finishedOnce sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// utterance counts the runs of speech seen so far, and ended marks that the current
	// one is over so the next transcript starts a new one.
	utterance int64
	ended     bool
	started   bool
	closed    bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("grok: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("grok: url must be ws:// or wss://, got %s", options.URL)
	}
	if len(options.Keyterms) > stt.MaxKeyterms {
		return nil, fmt.Errorf("grok: at most %d keyterms, got %d", stt.MaxKeyterms, len(options.Keyterms))
	}
	if options.SmartTurn < 0 || options.SmartTurn > 1 {
		return nil, fmt.Errorf("grok: smart turn threshold must be between 0 and 1, got %v", options.SmartTurn)
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 30 * time.Second
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

// Start dials the socket and waits for the server to report that it is ready for audio.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("grok: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Bearer " + s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("grok: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("grok: dial: %w", err)
	}
	s.conn = conn

	if err := s.handshake(); err != nil {
		conn.Close()
		return err
	}

	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop()
	return nil
}

// ProcessAudio streams one chunk of audio. The participant labels any transcript that
// results from it.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("grok: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("grok: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("grok: not started")
	}

	if err := s.write(websocket.BinaryMessage, pcm.Bytes()); err != nil {
		return fmt.Errorf("grok: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close ends the audio stream, waits for what is still being transcribed, then tears the
// connection down.
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
	query.Set("sample_rate", strconv.Itoa(stt.SampleRate))
	query.Set("encoding", "pcm")
	// Interims are off by default, and a call that only hears the settled turn looks
	// unanswered for as long as the caller was talking.
	query.Set("interim_results", "true")
	if s.options.Language != "" {
		query.Set("language", s.options.Language)
	}
	if s.options.EndpointingMs > 0 {
		query.Set("endpointing", strconv.Itoa(s.options.EndpointingMs))
	}
	if s.options.SmartTurn > 0 {
		query.Set("smart_turn", strconv.FormatFloat(s.options.SmartTurn, 'f', -1, 64))
		if s.options.SmartTurnTimeoutMs > 0 {
			query.Set("smart_turn_timeout", strconv.Itoa(s.options.SmartTurnTimeoutMs))
		}
	}
	for _, term := range stt.CleanKeyterms(s.options.Keyterms) {
		query.Add("keyterm", term)
	}

	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + query.Encode()
}

// handshake waits for the server to say its recogniser is up. Audio sent before that is
// audio the server is not yet listening to.
func (s *STT) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("grok: read handshake: %w", err)
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("grok: read handshake: %w", err)
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return fmt.Errorf("grok: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("grok: decode handshake: %w", err)
	}
	if message.Type == eventError {
		return fmt.Errorf("grok: handshake rejected: %s", message.Message)
	}
	if message.Type != eventCreated {
		return fmt.Errorf("grok: expected %q, got %q", eventCreated, message.Type)
	}
	return nil
}

// flush tells the server the audio has stopped and waits for the transcript of whatever
// it was still holding, so the tail of a call is not lost. A dead connection must not
// stop teardown.
func (s *STT) flush() {
	if err := s.write(websocket.TextMessage, []byte(`{"type":"`+controlTypeAudioDone+`"}`)); err != nil {
		s.logger.Debug("audio.done not delivered", "error", err)
		return
	}

	select {
	case <-s.finished:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the final transcript")
	}
}

func (s *STT) write(messageType int, payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(messageType, payload)
}

// readLoop translates server frames into events until the connection ends.
func (s *STT) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
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
	case eventPartial:
		s.handlePartial(message)
	case eventDone:
		s.handleDone(message)
	case eventError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.Message),
			Context:  "server",
			Fatal:    true,
		})
	case eventCreated:
		// The handshake already waited for this one, and the server sends it once.
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

// handlePartial reports what has been heard. Of the three shapes a frame comes in, only
// the speaker having stopped settles the turn.
func (s *STT) handlePartial(message serverMessage) {
	text := strings.TrimSpace(message.Text)

	if message.IsFinal && message.SpeechFinal {
		s.sendTranscript(message, stt.ModeFinal, text)
		s.endUtterance()
		return
	}
	// A chunk final locks the words behind it rather than ending the turn: the caller is
	// still talking. It restates the utterance just as an interim does, so both supersede
	// what came before rather than adding to it.
	s.sendTranscript(message, stt.ModeReplacement, text)
}

// handleDone reports the transcript of the last of the audio and releases a waiting Close.
//
// The turn has normally settled by now and this frame says the same words over again,
// which would have whoever is listening answer twice. It only carries news when the caller
// was cut off mid-sentence, which is exactly when the turn has not settled.
func (s *STT) handleDone(message serverMessage) {
	if !s.settledTheLastUtterance() {
		s.sendTranscript(message, stt.ModeFinal, strings.TrimSpace(message.Text))
		s.endUtterance()
	}
	s.finishedOnce.Do(func() { close(s.finished) })
}

func (s *STT) sendTranscript(message serverMessage, mode stt.Mode, text string) {
	if text == "" {
		return
	}
	participant, latencyMs := s.snapshot()

	s.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        mode,
		Utterance:   s.utteranceID(),
		Text:        text,
		// Only sent when Smart Turn is on, and then only at a silence boundary.
		Confidence:       message.EndOfTurnConfidence,
		Language:         message.Language,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
		AudioDurationMs:  message.Duration * 1000,
	})
}

// settledTheLastUtterance reports whether the run of speech that was in progress has
// already been reported as over.
func (s *STT) settledTheLastUtterance() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.ended
}

// endUtterance marks the current run of speech as over.
//
// The count moves on the next transcript rather than here, so a final and the hypothesis
// that opens the next turn are one boundary between utterances rather than two.
func (s *STT) endUtterance() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ended = true
}

// utteranceID numbers the run of speech the next transcript belongs to.
func (s *STT) utteranceID() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 || s.ended {
		s.utterance++
		s.ended = false
	}
	return s.utterance
}

// snapshot returns the current speaker and how long ago audio was last sent.
func (s *STT) snapshot() (stt.Participant, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	return s.participant, latencyMs
}
