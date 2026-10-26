//go:build integration

package api

import (
	"maps"
	"time"

	"github.com/gorilla/websocket"
)

// An approval naming a call the session never made is answered on the socket, naming the call,
// and the socket goes on: its close still ends the session normally.
func (s *SessionEventsSuite) TestAnApprovalNamingNoCallIsAnsweredAndTheSocketStaysOpen() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_approval", "tool_call_id": "call-nobody-made", "allowed": true,
	}))

	s.Equal(frame{"type": "error", "context": "tool", "tool_call_id": "call-nobody-made",
		"error": errNoApprovalWaits.Error()}, frame(s.await(watching, "error")))
	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "close"}))
	_, ended := s.socketEnd(watching)
	s.Equal("the session ended", s.closedWith(ended, websocket.CloseNormalClosure).Text)
}

// A call already answered waits for nothing, so an approval coming after its result is answered.
func (s *SessionEventsSuite) TestAnApprovalOfACallAlreadyAnsweredIsAnswered() {
	opened := s.withATool()
	watching := s.watch(opened.Id)
	s.respond(opened.Id)
	asked := s.await(watching, "tool_call")
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))
	s.await(watching, "tool_ran")

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_approval", "tool_call_id": asked["id"], "allowed": true,
	}))

	refused := s.await(watching, "error")
	s.Equal(asked["id"], refused["tool_call_id"])
	s.Equal(errNoApprovalWaits.Error(), refused["error"])
}

// The control: an approval sent before the result of a call outside a durable command, as the
// JS SDK's decide sends one, is answered by nothing but the call's own tool_ran, as on base
// 8e132bd3 (probe: no error frame before tool_ran, author-1.md «Base probe»).
func (s *SessionEventsSuite) TestAnApprovalBeforeTheResultOfAClientCallIsNotAnswered() {
	opened := s.withATool()
	watching := s.watch(opened.Id)
	s.respond(opened.Id)
	asked := s.await(watching, "tool_call")

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_approval", "tool_call_id": asked["id"], "allowed": true,
	}))
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))

	s.Equal("it ships tomorrow", s.awaitWithoutError(watching, "tool_ran")["result"])
}

// A waiting call outside any command is not the call of the command an approval names, so that
// approval reached nothing and is answered.
func (s *SessionEventsSuite) TestAnApprovalNamingACommandOfNoCallIsAnswered() {
	opened := s.withATool()
	watching := s.watch(opened.Id)
	s.respond(opened.Id)
	asked := s.await(watching, "tool_call")

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_approval", "tool_call_id": asked["id"], "request_id": s.utils.uuid(), "allowed": true,
	}))

	refused := s.await(watching, "error")
	s.Equal(asked["id"], refused["tool_call_id"])
	s.Equal(errNoApprovalWaits.Error(), refused["error"])
}

// An approval without allowed says nothing, and is answered naming its call.
func (s *SessionEventsSuite) TestAnApprovalWithoutAllowedIsAnsweredNamingItsCall() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": "call-1"}))

	s.Equal(frame{"type": "error", "context": "tool", "tool_call_id": "call-1",
		"error": "tool_approval needs allowed"}, frame(s.await(watching, "error")))
}

// allowed null is allowed left out.
func (s *SessionEventsSuite) TestAnApprovalWhoseAllowedIsNullIsAnsweredAsOneWithout() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteMessage(websocket.TextMessage,
		[]byte(`{"type":"tool_approval","tool_call_id":"call-1","allowed":null}`)))

	s.Equal("tool_approval needs allowed", s.await(watching, "error")["error"])
}

// An approval naming no call at all is answered with no tool_call_id: the frame names a call
// only when there is one.
func (s *SessionEventsSuite) TestAnApprovalWithNoCallIDIsAnsweredWithoutOne() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_approval", "allowed": false}))

	s.Equal(frame{"type": "error", "context": "tool", "error": errNoApprovalWaits.Error()},
		frame(s.await(watching, "error")))
}

// The control: an allowed that is not a boolean cannot be read as a command, and is refused as
// any such frame is, as on base 8e132bd3.
func (s *SessionEventsSuite) TestAnApprovalWhoseAllowedIsNotABooleanIsRefusedWithPolicyViolation() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteMessage(websocket.TextMessage,
		[]byte(`{"type":"tool_approval","tool_call_id":"call-1","allowed":"yes"}`)))

	s.refusedAsUnreadable(watching)
}

// A socket on a node that is not running the session is answered too: the answer is the
// session's, and crosses the relay as its other frames do.
func (s *RelayedSessionSuite) TestAnApprovalNamingNoCallIsAnsweredOnTheOtherNode() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_approval", "tool_call_id": "call-nobody-made", "allowed": true,
	}))

	refused := s.await(watching, "error")
	s.Equal("call-nobody-made", refused["tool_call_id"])
	s.Equal(errNoApprovalWaits.Error(), refused["error"])
}

// The control: on a durable command, an approval repeating the call's request_id and turn_id is
// recorded on its step and answered by nothing but the call's own tool_ran, as on base 8e132bd3
// (probe: no error frame before tool_ran, author-1.md «Base probe»).
func (s *SessionEventsSuite) TestAnApprovalOfACommandsCallIsNotAnswered() {
	user := s.data.createUser()
	opened := user.createSession(s.askingForATool())
	watching := user.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "respond", "text": "where is my order", "request_id": s.utils.uuid(),
	}))
	asked := s.await(watching, "tool_call")
	s.Require().NotEmpty(asked["request_id"])
	s.Require().NotEmpty(asked["turn_id"])
	ids := map[string]any{"tool_call_id": asked["id"], "request_id": asked["request_id"], "turn_id": asked["turn_id"]}

	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_approval", "allowed": true})))
	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_result", "output": "it ships tomorrow"})))

	s.Equal("it ships tomorrow", s.awaitWithoutError(watching, "tool_ran")["result"])
}

// awaitWithoutError is await for a socket that must send no error frame before the one wanted.
func (s *RouterSuite) awaitWithoutError(connection *websocket.Conn, wanted string) map[string]any {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var received map[string]any
		s.Require().NoError(connection.ReadJSON(&received), "the socket closed before "+wanted+" arrived")
		s.Require().NotEqual("error", received["type"], "answered with an error: %v", received)
		if received["type"] == wanted {
			return received
		}
	}
}

func merged(frames ...map[string]any) map[string]any {
	all := map[string]any{}
	for _, one := range frames {
		maps.Copy(all, one)
	}
	return all
}
