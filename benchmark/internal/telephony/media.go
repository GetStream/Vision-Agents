package telephony

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

// Frame is one 20 ms chunk from the far end.
type Frame struct {
	PCM []int16
}

// Media is a bidirectional 8 kHz PCM pipe.
type Media interface {
	Send(pcm []int16) error
	Recv() <-chan Frame
	Close() error
}

type wsFrame struct {
	Event string        `json:"event"`
	Media *mediaPayload `json:"media,omitempty"`
}

type mediaPayload struct {
	Track   string `json:"track"`
	Payload string `json:"payload"`
}

// Stream is a Telnyx bidirectional media websocket from the harness side.
type Stream struct {
	conn   *websocket.Conn
	recv   chan Frame
	mu     sync.Mutex
	closed bool
	logger *slog.Logger
}

func newStream(conn *websocket.Conn, logger *slog.Logger) *Stream {
	if logger == nil {
		logger = slog.Default()
	}
	return &Stream{
		conn:   conn,
		recv:   make(chan Frame, 64),
		logger: logger,
	}
}

// WaitStart blocks until Telnyx sends the start event.
func (s *Stream) WaitStart(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		_ = s.conn.SetReadDeadline(time.Now().Add(time.Until(deadline)))
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("telephony: read start: %w", err)
		}
		var frame wsFrame
		if err := json.Unmarshal(raw, &frame); err != nil {
			continue
		}
		switch frame.Event {
		case "connected":
			continue
		case "start":
			go s.readLoop()
			return nil
		}
	}
	return fmt.Errorf("telephony: timed out waiting for media start")
}

func (s *Stream) readLoop() {
	defer close(s.recv)
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			return
		}
		var frame wsFrame
		if err := json.Unmarshal(raw, &frame); err != nil {
			continue
		}
		switch frame.Event {
		case "media":
			if frame.Media == nil || frame.Media.Payload == "" {
				continue
			}
			payload, err := decodePayload(frame.Media.Payload)
			if err != nil {
				continue
			}
			select {
			case s.recv <- Frame{PCM: payload}:
			default:
			}
		case "stop", "error":
			return
		}
	}
}

// Send writes 8 kHz PCM as mu-law media frames.
func (s *Stream) Send(pcm []int16) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("telephony: stream closed")
	}
	for i := 0; i < len(pcm); i += audio.FrameSamples {
		end := i + audio.FrameSamples
		chunk := pcm[i:min(end, len(pcm))]
		if len(chunk) < audio.FrameSamples {
			chunk = audio.PadRight(chunk, audio.FrameSamples)
		}
		encoded := audio.EncodeUlaw(chunk)
		msg := map[string]any{
			"event": "media",
			"media": map[string]any{
				"payload": encodePayload(encoded),
			},
		}
		if err := s.conn.WriteJSON(msg); err != nil {
			return err
		}
	}
	return nil
}

// Recv is inbound agent audio.
func (s *Stream) Recv() <-chan Frame { return s.recv }

// Close hangs up the websocket.
func (s *Stream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	_ = s.conn.WriteJSON(map[string]string{"event": "stop"})
	return s.conn.Close()
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Server accepts one Telnyx media websocket at /media/{token}.
type Server struct {
	token  string
	ready  chan *Stream
	logger *slog.Logger
	http   *http.Server
}

// StartMediaServer listens for the inbound Telnyx websocket.
func StartMediaServer(addr, token string, logger *slog.Logger) (*Server, error) {
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{token: token, ready: make(chan *Stream, 1), logger: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("/media/", s.handle)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	s.http = &http.Server{Handler: mux}
	go func() {
		_ = s.http.Serve(listener)
	}()
	return s, nil
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	token := r.URL.Path[len("/media/"):]
	if token != s.token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Error("media upgrade", "err", err)
		return
	}
	stream := newStream(conn, s.logger)
	select {
	case s.ready <- stream:
	default:
		conn.Close()
	}
}

// Accept waits for Telnyx to connect.
func (s *Server) Accept(timeout time.Duration) (*Stream, error) {
	select {
	case stream := <-s.ready:
		return stream, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("telephony: no media websocket within %s", timeout)
	}
}

// Close shuts the HTTP listener.
func (s *Server) Close() error {
	if s.http == nil {
		return nil
	}
	return s.http.Close()
}
