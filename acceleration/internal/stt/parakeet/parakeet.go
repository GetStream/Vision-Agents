// Package parakeet implements the stt.STT contract against the streaming Parakeet
// deployment in acceleration/deploy/parakeet.
//
// Parakeet TDT is a chunk model, so the server re-decodes the current utterance on a
// timer and sends the whole hypothesis each time. Partials are therefore replacements,
// not deltas. Turn boundaries come from energy-based silence detection on the server.
package parakeet

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "parakeet"

// DefaultModel is the model the deployment serves.
const DefaultModel = "parakeet-tdt-0.6b-v3"

// Server message types, mirroring acceleration/deploy/parakeet/model/model.py.
const (
	messageReady        = "ready"
	messageStartOfTurn  = "start_of_turn"
	messagePartial      = "partial"
	messageFinal        = "final"
	messageError        = "error"
	statusFinished      = "finished"
	controlTypeEndAudio = "end_audio"
)

// Options configures the provider. URL falls back to PARAKEET_WS_URL and APIKey to
// BASETEN_API_KEY.
type Options struct {
	URL    string
	APIKey string
	Model  string
	// HandshakeTimeout bounds the initial connect. Baseten cold starts can be slow.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the server to finish transcribing
	// whatever audio is still buffered.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMetadata is the opening frame that describes the audio stream.
type clientMetadata struct {
	SampleRate int    `json:"sample_rate"`
	Encoding   string `json:"encoding"`
}

// serverMessage is a frame sent by the deployment.
type serverMessage struct {
	Type             string  `json:"type"`
	Status           string  `json:"status"`
	Text             string  `json:"text"`
	Error            string  `json:"error"`
	AudioDurationMs  float64 `json:"audio_duration_ms"`
	ProcessingTimeMs float64 `json:"processing_time_ms"`
}

// STT is a streaming Parakeet speech-to-text session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// finished is closed when the server acknowledges an end_audio flush.
	finished     chan struct{}
	finishedOnce sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	started     bool
	closed      bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.URL == "" {
		options.URL = os.Getenv("PARAKEET_WS_URL")
	}
	if options.URL == "" {
		return nil, errors.New("parakeet: websocket url is required (set PARAKEET_WS_URL)")
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("parakeet: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv("BASETEN_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("parakeet: api key is required (set BASETEN_API_KEY)")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 60 * time.Second
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

// Start dials the deployment and completes the metadata handshake. It returns once the
// server reports it is ready for audio.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("parakeet: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Api-Key " + s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.options.URL, header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("parakeet: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("parakeet: dial: %w", err)
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
		return fmt.Errorf("parakeet: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("parakeet: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("parakeet: not started")
	}

	if err := s.write(websocket.BinaryMessage, pcm.Bytes()); err != nil {
		return fmt.Errorf("parakeet: write audio: %w", err)
	}
	return nil
}

// Events returns transcripts and turn boundaries.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close asks the server to flush any pending audio, then tears down the connection.
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

// flush asks the server to transcribe any buffered audio and waits for it to confirm,
// so the tail of a call is not lost. A dead connection must not stop teardown.
func (s *STT) flush() {
	if err := s.write(websocket.TextMessage, []byte(`{"type":"`+controlTypeEndAudio+`"}`)); err != nil {
		s.logger.Debug("end_audio not delivered", "error", err)
		return
	}

	select {
	case <-s.finished:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the final transcript")
	}
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// TurnDetection reports true: the deployment ends turns on trailing silence.
func (s *STT) TurnDetection() bool { return true }

// Client exposes the underlying WebSocket so callers can use the deployment directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// handshake sends the audio metadata and waits for the server to acknowledge it.
func (s *STT) handshake() error {
	payload, err := json.Marshal(clientMetadata{SampleRate: stt.SampleRate, Encoding: "linear16"})
	if err != nil {
		return fmt.Errorf("parakeet: encode metadata: %w", err)
	}
	if err := s.write(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("parakeet: send metadata: %w", err)
	}

	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("parakeet: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("parakeet: decode handshake: %w", err)
	}
	if message.Type == messageError {
		return fmt.Errorf("parakeet: handshake rejected: %s", message.Error)
	}
	if message.Type != messageReady {
		return fmt.Errorf("parakeet: expected %q, got %q", messageReady, message.Type)
	}
	return nil
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
	participant, latencyMs := s.snapshot()

	switch message.Type {
	case messageStartOfTurn:
		s.emitter.Send(stt.TurnStarted{Participant: participant})
	case messagePartial:
		s.sendTranscript(participant, message, stt.ModeReplacement, latencyMs)
	case messageFinal:
		s.sendTranscript(participant, message, stt.ModeFinal, latencyMs)
		s.emitter.Send(stt.TurnEnded{Participant: participant, DurationMs: message.AudioDurationMs})
	case messageError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.Error),
			Context:  "server",
			Fatal:    true,
		})
	case "":
		if message.Status == statusFinished {
			s.finishedOnce.Do(func() { close(s.finished) })
			return
		}
		s.logger.Debug("unhandled frame", "status", message.Status)
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

func (s *STT) sendTranscript(
	participant stt.Participant,
	message serverMessage,
	mode stt.Mode,
	latencyMs float64,
) {
	text := strings.TrimSpace(message.Text)
	if text == "" {
		return
	}

	// The server measures its own decode time; fall back to the round trip we observed.
	processingMs := message.ProcessingTimeMs
	if processingMs == 0 {
		processingMs = latencyMs
	}

	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: processingMs,
		AudioDurationMs:  message.AudioDurationMs,
	})
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
