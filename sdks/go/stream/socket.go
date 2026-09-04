package stream

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

// ErrSocketClosed is returned by a send on a socket that is no longer open.
var ErrSocketClosed = errors.New("stream: the socket is not open")

// Frame is one JSON message on a session socket. Every frame carries a "type" and the
// fields of that event.
type Frame map[string]any

// Type is the frame's kind, or the empty string if it has none.
func (f Frame) Type() string {
	kind, _ := f["type"].(string)
	return kind
}

// String reads a string field, or the empty string if it is absent or another type.
func (f Frame) String(key string) string {
	value, _ := f[key].(string)
	return value
}

// Bool reads a boolean field, false if it is absent or another type.
func (f Frame) Bool(key string) bool {
	value, _ := f[key].(bool)
	return value
}

// Int reads a numeric field, zero if it is absent or another type. JSON numbers decode as
// float64, which is why this is not a plain assertion.
func (f Frame) Int(key string) int {
	value, _ := f[key].(float64)
	return int(value)
}

// Frame reads a nested object, nil if it is absent or another type.
func (f Frame) Frame(key string) Frame {
	value, _ := f[key].(map[string]any)
	return Frame(value)
}

// Socket is one WebSocket to the router.
//
// The rest of the client is generated from the OpenAPI spec, but OpenAPI stops at the
// upgrade, so the sockets are written by hand. This is the whole of it: JSON one way, JSON
// and audio the other.
type Socket struct {
	url     string
	headers http.Header
	dialer  *websocket.Dialer
	logger  *slog.Logger

	// writes serialises sending, because a WebSocket allows one writer at a time and tool
	// results are answered from a goroutine per call.
	writes sync.Mutex

	mu         sync.Mutex
	connection *websocket.Conn
}

// NewSocket describes a socket without opening it.
func NewSocket(url string, headers http.Header, client *http.Client, logger *slog.Logger) *Socket {
	dialer := *websocket.DefaultDialer
	if client != nil {
		dialer.Proxy = http.ProxyFromEnvironment
		dialer.Jar = client.Jar
		if client.Timeout > 0 {
			dialer.HandshakeTimeout = client.Timeout
		}
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Socket{url: url, headers: headers, dialer: &dialer, logger: logger}
}

// Open dials the router, returning what it said if it refuses the upgrade.
func (s *Socket) Open(ctx context.Context) error {
	connection, response, err := s.dialer.DialContext(ctx, s.url, s.headers)
	if err != nil {
		if response != nil {
			return fmt.Errorf("stream: the router refused the socket with %s: %w", response.Status, err)
		}
		return fmt.Errorf("stream: dialling %s: %w", s.url, err)
	}

	s.mu.Lock()
	s.connection = connection
	s.mu.Unlock()
	return nil
}

// IsOpen reports whether the socket can still carry a message.
func (s *Socket) IsOpen() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connection != nil
}

// Send writes one JSON frame.
func (s *Socket) Send(frame Frame) error {
	s.mu.Lock()
	connection := s.connection
	s.mu.Unlock()
	if connection == nil {
		return ErrSocketClosed
	}

	s.writes.Lock()
	defer s.writes.Unlock()
	return connection.WriteJSON(frame)
}

// SendAudio writes one binary frame.
func (s *Socket) SendAudio(payload []byte) error {
	s.mu.Lock()
	connection := s.connection
	s.mu.Unlock()
	if connection == nil {
		return ErrSocketClosed
	}

	s.writes.Lock()
	defer s.writes.Unlock()
	return connection.WriteMessage(websocket.BinaryMessage, payload)
}

// Read returns the next frame, or an error once the socket closes.
//
// A text frame that is not JSON is dropped rather than returned: it can only be a bug on
// the far side, and ending the stream over it would lose everything said after. Binary
// frames come back as they are.
func (s *Socket) Read() (Frame, []byte, error) {
	s.mu.Lock()
	connection := s.connection
	s.mu.Unlock()
	if connection == nil {
		return nil, nil, ErrSocketClosed
	}

	for {
		kind, payload, err := connection.ReadMessage()
		if err != nil {
			return nil, nil, err
		}
		if kind == websocket.BinaryMessage {
			return nil, payload, nil
		}

		var frame Frame
		if err := json.Unmarshal(payload, &frame); err != nil {
			s.logger.Warn("dropping a frame that is not JSON", "error", err)
			continue
		}
		return frame, nil, nil
	}
}

// Close shuts the socket. Safe to call twice.
func (s *Socket) Close() error {
	s.mu.Lock()
	connection := s.connection
	s.connection = nil
	s.mu.Unlock()
	if connection == nil {
		return nil
	}
	return connection.Close()
}
