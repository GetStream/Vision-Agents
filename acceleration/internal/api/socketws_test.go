//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// SocketSessionSuite covers the socket a voice conversation is held over when there is no
// call: the caller's audio in and the agent's speech out, on the connection that holds the
// session open.
type SocketSessionSuite struct {
	RouterSuite
}

func TestSocketSessionSuite(t *testing.T) {
	runSuite(t, new(SocketSessionSuite))
}

func (s *SocketSessionSuite) SetupTest() {
	s.useFixture("standard")
}

// starts opens a socket session that greets with the given words and returns it with the
// session it was answered with.
func (s *SocketSessionSuite) starts(greeting string) (*websocket.Conn, map[string]any) {
	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "sample_rate": 24_000,
		"session": map[string]any{"greeting": greeting},
	}))
	answered := s.readFrame(connection)
	s.Require().Equal("session", answered["type"], "the socket answered %v", answered)
	return connection, answered
}

func (s *SocketSessionSuite) readFrame(connection *websocket.Conn) map[string]any {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		kind, payload, err := connection.ReadMessage()
		s.Require().NoError(err, "the socket closed before a frame arrived")
		if kind != websocket.TextMessage {
			continue
		}
		var received map[string]any
		s.Require().NoError(json.Unmarshal(payload, &received))
		return received
	}
}

func (s *SocketSessionSuite) TestTheSocketHoldsAVoiceSessionThatGreetsTheCaller() {
	greeting := s.utils.uuid()

	_, answered := s.starts(greeting)

	session := answered["session"].(map[string]any)
	s.Equal(string(Live), session["state"])
	s.InDelta(24_000, answered["sample_rate"], 0)
	s.Require().Eventually(func() bool { return slices.Contains(s.voice.spoken(), greeting) },
		settleFor, 20*time.Millisecond, "the agent never spoke on the socket's call")
}

func (s *SocketSessionSuite) TestTheCallersAudioIsTakenWhileTheSessionLasts() {
	connection, answered := s.starts(s.utils.uuid())
	id := answered["session"].(map[string]any)["id"].(string)

	for range 10 {
		s.Require().NoError(connection.WriteMessage(websocket.BinaryMessage, make([]byte, 24_000*20/1000*2)))
	}

	s.Equal(Live, s.serverClient.getSession(id).State)
}

func (s *SocketSessionSuite) TestClosingTheSocketEndsTheSession() {
	connection, answered := s.starts(s.utils.uuid())
	id := answered["session"].(map[string]any)["id"].(string)

	s.Require().NoError(connection.Close())

	s.Require().Eventually(func() bool {
		var read Session
		status := s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &read)
		return status == http.StatusNotFound || read.State == Ended
	}, settleFor, 50*time.Millisecond, "the session outlived the socket holding it")
}

func (s *SocketSessionSuite) TestASocketThatDoesNotStartWithAStartFrameIsRefused() {
	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "hello"}))

	refused := s.readFrame(connection)

	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "start frame")
}

func (s *SocketSessionSuite) TestATextSessionIsRefused() {
	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "session": map[string]any{"text": true},
	}))

	refused := s.readFrame(connection)

	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "text session")
}

func (s *SocketSessionSuite) TestACallerWithNoCredentialsIsRefused() {
	_, status := s.unauthenticatedClient.watch("/v1/agents/socket")

	s.Equal(http.StatusUnauthorized, status)
}
