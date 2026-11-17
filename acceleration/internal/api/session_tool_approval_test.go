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

	s.Equal(frame{"type": "tool_approval_ignored", "tool_call_id": "call-nobody-made",
		"reason": errNoApprovalWaits.Error()}, frame(s.await(watching, "tool_approval_ignored")))
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

	s.Equal(frame{"type": "tool_approval_ignored", "tool_call_id": asked["id"], "reason": errNoApprovalWaits.Error()},
		frame(s.await(watching, "tool_approval_ignored")))
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

	refused := s.await(watching, "tool_approval_ignored")
	s.Equal(asked["id"], refused["tool_call_id"])
	s.Equal(errNoApprovalWaits.Error(), refused["reason"])
}

// The control: an approval without allowed says nothing, and is refused with the error frame
// base 8e132bd3 sent, naming no call.
func (s *SessionEventsSuite) TestAnApprovalWithoutAllowedIsRefusedAsOnBase() {
	watching := s.watch(s.inWriting(CreateSessionRequest{}).Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": "call-1"}))

	s.Equal(frame{"type": "error", "context": "tool", "error": "tool_approval needs allowed"},
		frame(s.await(watching, "error")))
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

	s.Equal(frame{"type": "tool_approval_ignored", "reason": errNoApprovalWaits.Error()},
		frame(s.await(watching, "tool_approval_ignored")))
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

	refused := s.await(watching, "tool_approval_ignored")
	s.Equal("call-nobody-made", refused["tool_call_id"])
	s.Equal(errNoApprovalWaits.Error(), refused["reason"])
}

// The control: on a durable command, an approval repeating the call's request_id and turn_id is
// recorded on its step and answered by nothing but the call's own tool_ran, as on base 8e132bd3
// (probe: no error frame before tool_ran, author-1.md «Base probe»).
func (s *SessionEventsSuite) TestAnApprovalOfACommandsCallIsNotAnswered() {
	watching, ids := s.commandsCallAsked()

	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_approval", "allowed": true})))
	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_result", "output": "it ships tomorrow"})))

	s.Equal("it ships tomorrow", s.awaitWithoutError(watching, "tool_ran")["result"])
}

// A command's call answered already waits for nothing, so an approval repeating its ids after
// its result reaches nothing and is answered.
func (s *SessionEventsSuite) TestASecondApprovalOfACommandsCallIsAnswered() {
	watching, ids := s.commandsCallAsked()
	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_result", "output": "it ships tomorrow"})))
	s.await(watching, "tool_ran")

	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_approval", "allowed": false})))

	s.Equal(frame(merged(ids, map[string]any{"type": "tool_approval_ignored", "reason": errNoApprovalWaits.Error()})),
		frame(s.await(watching, "tool_approval_ignored")))
}

// The command and turn an approval names being right does not make up a call: one naming a
// call nobody made is answered.
func (s *SessionEventsSuite) TestAnApprovalNamingNoCallOfTheCommandsTurnIsAnswered() {
	watching, ids := s.commandsCallAsked()

	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_approval",
		"tool_call_id": "call-nobody-made", "allowed": true})))

	s.Equal(frame(merged(ids, map[string]any{"type": "tool_approval_ignored", "tool_call_id": "call-nobody-made",
		"reason": errNoApprovalWaits.Error()})), frame(s.await(watching, "tool_approval_ignored")))
}

// A command's call answered with its request_id and no turn_id is answered naming the missing
// turn_id, as the Go SDK's AnswerApproval sends one, and goes on waiting for its result.
func (s *SessionEventsSuite) TestAnApprovalOfACommandsCallWithoutTheTurnIsAnsweredNamingIt() {
	watching, ids := s.commandsCallAsked()

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_approval",
		"tool_call_id": ids["tool_call_id"], "request_id": ids["request_id"], "allowed": true}))

	s.Equal(frame{"type": "tool_approval_ignored", "tool_call_id": ids["tool_call_id"], "request_id": ids["request_id"],
		"reason": errApprovalNeedsTurn.Error()}, frame(s.await(watching, "tool_approval_ignored")))
	s.Require().NoError(watching.WriteJSON(merged(ids, map[string]any{"type": "tool_result", "output": "it ships tomorrow"})))
	s.Equal("it ships tomorrow", s.await(watching, "tool_ran")["result"])
}

// commandsCallAsked opens a session asked by a durable command, and is the socket with the ids
// of the client call the command's turn made.
func (s *SessionEventsSuite) commandsCallAsked() (*websocket.Conn, map[string]any) {
	user := s.data.createUser()
	opened := user.createSession(s.askingForATool())
	watching := user.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "respond", "text": "where is my order", "request_id": s.utils.uuid(),
	}))
	asked := s.await(watching, "tool_call")
	s.Require().NotEmpty(asked["request_id"])
	s.Require().NotEmpty(asked["turn_id"])
	return watching, map[string]any{"tool_call_id": asked["id"], "request_id": asked["request_id"], "turn_id": asked["turn_id"]}
}

// awaitWithoutError is await for a socket that must send no error or tool_approval_ignored
// frame before the one wanted.
func (s *RouterSuite) awaitWithoutError(connection *websocket.Conn, wanted string) map[string]any {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var received map[string]any
		s.Require().NoError(connection.ReadJSON(&received), "the socket closed before "+wanted+" arrived")
		s.Require().NotContains([]string{"error", "tool_approval_ignored"}, received["type"], "answered: %v", received)
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
