//go:build integration

package api

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// socketEnd reads a socket to its end: every JSON frame left on it, and how it ended.
func (s *RouterSuite) socketEnd(connection *websocket.Conn) ([]frame, error) {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	var read []frame
	for {
		kind, payload, err := connection.ReadMessage()
		if err != nil {
			return read, err
		}
		var decoded frame
		if kind == websocket.TextMessage && json.Unmarshal(payload, &decoded) == nil {
			read = append(read, decoded)
		}
	}
}

// closedWith checks a socket ended with a close frame of the code given.
func (s *RouterSuite) closedWith(ended error, code int) *websocket.CloseError {
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed, "the socket ended without a close frame")
	s.Equal(code, closed.Code)
	return closed
}

// SocketHangUpSuite covers a caller hanging up on /v1/agents/socket while the agent speaks.
// It is a suite of its own for the router's log, and for a voice no other test has closed:
// the harness has one, and the first session to end closes it.
type SocketHangUpSuite struct {
	RouterSuite
	logged *lockedLog
}

func TestSocketHangUpSuite(t *testing.T) {
	runSuite(t, new(SocketHangUpSuite))
}

func (s *SocketHangUpSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *SocketHangUpSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *SocketHangUpSuite) TestACallerHangingUpMidGreetingIsNoFailure() {
	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "session": map[string]any{"greeting": map[string]any{"text": "hello there"}},
	}))
	// Ten seconds of greeting, far more than the socket's 400 ms of queue, so the agent is
	// still publishing when the caller goes.
	speaking := make(chan struct{})
	go func() {
		defer close(speaking)
		chunk := audio.PcmData{Samples: make([]int16, 1600), SampleRate: 16_000, Channels: 1}
		for range 100 {
			s.voice.emitter.Send(tts.AudioChunk{SynthesisID: "greeting", Audio: chunk})
		}
	}()
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		kind, _, err := connection.ReadMessage()
		s.Require().NoError(err, "the greeting never reached the socket")
		if kind == websocket.BinaryMessage {
			break
		}
	}

	s.Require().NoError(connection.Close())

	select {
	case <-speaking:
	case <-time.After(settleFor):
		s.FailNow("the agent never took the rest of the greeting")
	}
	s.Require().Eventually(func() bool {
		return strings.Contains(s.logged.String(), "has been left")
	}, settleFor, 20*time.Millisecond, "the agent never published into the call that was left")
	var failures []string
	for line := range strings.Lines(s.logged.String()) {
		if strings.Contains(line, "level=ERROR") {
			failures = append(failures, line)
		}
	}
	s.Empty(failures, "a caller hanging up is logged as a failure")
}
