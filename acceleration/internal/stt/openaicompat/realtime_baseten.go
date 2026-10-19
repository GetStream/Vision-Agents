package openaicompat

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type basetenMessage struct {
	Type              string `json:"type"`
	IsFinal           bool   `json:"is_final"`
	IsEndOfAudioFlush bool   `json:"is_end_of_audio_flush"`
	Language          string `json:"language_code"`
	Message           string `json:"message"`
	Segments          []struct {
		Text  string  `json:"text"`
		Start float64 `json:"start_time"`
		End   float64 `json:"end_time"`
	} `json:"segments"`
}

// basetenRealtime keeps the server's VAD boundaries and replacement transcripts.
type basetenRealtime struct {
	options Options
	emitter *stt.Emitter
	writeMu sync.Mutex
	mu      sync.Mutex
	conn    *websocket.Conn
	closed  bool
	started bool
	stop    func() bool

	participant stt.Participant
	utterance   int64
	hasWords    bool
	finished    chan struct{}
	finishOnce  sync.Once
}

// Baseten's Qwen wire format has server VAD and replacement segments rather than
// OpenAI's item IDs and deltas. Keep that lifecycle separate from the OpenAI session.
func newBasetenRealtime(options Options) (*basetenRealtime, error) {
	endpoint, err := parseBaseURL(options.BaseURL)
	if err != nil {
		return nil, err
	}
	if endpoint.Scheme == "https" {
		endpoint.Scheme = "wss"
	} else {
		endpoint.Scheme = "ws"
	}
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + "/websocket"
	options.BaseURL = endpoint.String()
	if options.Settings.Prompt != "" || len(options.Settings.Keywords) > 0 || len(options.Settings.Languages) > 0 || len(options.Languages) > 1 || options.Settings.Delay != "" || len(options.Settings.Include) > 0 {
		return nil, errors.New("openai-compatible: Baseten streaming does not support prompt, keywords, multiple languages, delay or include settings")
	}
	return &basetenRealtime{options: options, emitter: stt.NewEmitter(64), utterance: 1, finished: make(chan struct{})}, nil
}

func (s *basetenRealtime) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started || s.closed {
		s.mu.Unlock()
		return errors.New("openai-compatible: baseten: session already started or closed")
	}
	s.started = true
	s.mu.Unlock()

	header := make(http.Header)
	if s.options.APIKey != "" {
		header.Set("Authorization", "Api-Key "+s.options.APIKey)
	}
	conn, _, err := (&websocket.Dialer{HandshakeTimeout: realtimeHandshakeTimeout}).DialContext(ctx, s.options.BaseURL, header)
	if err != nil {
		return fmt.Errorf("openai-compatible: baseten: connect: %w", err)
	}
	// This protocol has no session.created acknowledgement. Setup precedes all audio.
	setup := map[string]any{
		"whisper_params":   map[string]any{"audio_language": "auto"},
		"streaming_params": map[string]any{"enable_partial_transcripts": true},
	}
	if s.options.Settings.Language != "" {
		setup["whisper_params"].(map[string]any)["audio_language"] = s.options.Settings.Language
	}
	if err := conn.WriteJSON(setup); err != nil {
		_ = conn.Close()
		return fmt.Errorf("openai-compatible: baseten: configure: %w", err)
	}
	s.mu.Lock()
	if s.closed || ctx.Err() != nil {
		s.mu.Unlock()
		_ = conn.Close()
		return errors.New("openai-compatible: baseten: session closed during start")
	}
	s.conn = conn
	s.stop = context.AfterFunc(ctx, func() { _ = conn.Close() })
	s.mu.Unlock()
	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.Model(), At: time.Now()})
	go s.readLoop(conn)
	return nil
}

func (s *basetenRealtime) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("openai-compatible: baseten: %w", err)
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	s.mu.Lock()
	conn, closed := s.conn, s.closed
	s.participant = participant
	s.mu.Unlock()
	if conn == nil || closed {
		return errors.New("openai-compatible: baseten: session is not open")
	}
	_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return conn.WriteJSON(map[string]string{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(pcm.Bytes())})
}

func (s *basetenRealtime) Events() <-chan stt.Event { return s.emitter.Events() }
func (s *basetenRealtime) Provider() string         { return ProviderName }
func (s *basetenRealtime) Model() string            { return s.options.Model }
func (s *basetenRealtime) Client() *websocket.Conn  { return s.conn }

func (s *basetenRealtime) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn, stop := s.conn, s.stop
	s.mu.Unlock()
	if stop != nil {
		stop()
	}
	if conn != nil {
		s.writeMu.Lock()
		_ = conn.SetWriteDeadline(time.Now().Add(realtimeCloseTimeout))
		err := conn.WriteJSON(map[string]string{"type": "input_audio_buffer.commit"})
		s.writeMu.Unlock()
		if err == nil {
			select {
			case <-s.finished:
			case <-time.After(realtimeCloseTimeout):
			}
		}
		_ = conn.Close()
	}
	s.emitter.Close()
	return nil
}

func (s *basetenRealtime) readLoop(conn *websocket.Conn) {
	defer s.finishOnce.Do(func() { close(s.finished) })
	for {
		var message basetenMessage
		if err := conn.ReadJSON(&message); err != nil {
			s.mu.Lock()
			closed := s.closed
			s.mu.Unlock()
			if !closed {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
					s.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: s.Model(), Clean: true, At: time.Now()})
				} else {
					s.emitter.Send(stt.Error{Provider: ProviderName, Model: s.Model(), Err: err, Context: "read", Fatal: true})
				}
			}
			return
		}
		s.handleMessage(message)
		if message.Type == "error" {
			return
		}
	}
}

func (s *basetenRealtime) handleMessage(message basetenMessage) {
	if message.Type == "error" {
		s.emitter.Send(stt.Error{Provider: ProviderName, Model: s.Model(), Err: errors.New(message.Message), Context: "server", Fatal: true})
		return
	}
	if message.Type != "transcription" {
		return
	}
	var parts []string
	for _, segment := range message.Segments {
		if text := strings.TrimSpace(segment.Text); text != "" {
			parts = append(parts, text)
		}
	}
	s.mu.Lock()
	participant, utterance, hadWords := s.participant, s.utterance, s.hasWords
	if message.IsFinal {
		s.utterance++
		s.hasWords = false
	} else if len(parts) > 0 {
		s.hasWords = true
	}
	s.mu.Unlock()
	if len(parts) > 0 || message.IsFinal && hadWords {
		mode := stt.ModeReplacement
		if message.IsFinal {
			mode = stt.ModeFinal
		}
		var durationMs float64
		if len(message.Segments) > 0 {
			durationMs = max(0, message.Segments[len(message.Segments)-1].End-message.Segments[0].Start) * 1000
		}
		s.emitter.Send(stt.Transcript{Participant: participant, Mode: mode, Utterance: utterance, Text: strings.Join(parts, " "), Language: message.Language, AudioDurationMs: durationMs, Provider: ProviderName, Model: s.Model()})
	}
	// A VAD final only ends a turn. Close waits for the explicit commit's flush final.
	if message.IsFinal && message.IsEndOfAudioFlush {
		s.finishOnce.Do(func() { close(s.finished) })
	}
}
