//go:build integration

package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// RelayedSessionSuite covers the events socket when it does not land on the node running
// the conversation, which is every socket of a deployment with more than one node.
type RelayedSessionSuite struct {
	RouterSuite

	// other is the node that is running nothing: every session a test here opens is
	// opened on the suite's own node and watched from this one.
	other *httptest.Server
}

func TestRelayedSessionSuite(t *testing.T) {
	runSuite(t, new(RelayedSessionSuite))
}

func (s *RelayedSessionSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.other = s.otherNode()
}

func (s *RelayedSessionSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *RelayedSessionSuite) TestASessionIsWatchedFromANodeThatIsNotRunningIt() {
	answer := s.utils.uuid()
	opened := s.inWriting(CreateSessionRequest{
		Llm: pointerTo("echo/echo-model"), ConfigId: s.data.instructedAgent(answer),
	})
	watching := s.watchFromTheOtherNode(opened.Id)

	s.respond(opened.Id, "hello")

	answered := s.await(watching, "responded")
	s.Contains(answered["text"], answer, "the conversation's own answer did not cross the relay")
	s.NotEmpty(answered["turn_id"])
}

func (s *RelayedSessionSuite) TestAToolResultSentToTheOtherNodeReachesTheModel() {
	opened := s.inWriting(CreateSessionRequest{
		Llm: pointerTo("tooling/tool-model"),
		Tools: &[]SessionTool{{
			Name: lookupOrder, Description: "find an order by its number",
			Parameters: &map[string]any{"type": "object"},
		}},
	})
	watching := s.watchFromTheOtherNode(opened.Id)
	s.respond(opened.Id, "where is my order")

	asked := s.await(watching, "tool_call")
	s.Equal(lookupOrder, asked["name"])
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))

	ran := s.await(watching, "tool_ran")
	s.Equal("it ships tomorrow", ran["result"],
		"the answer did not get back to the node holding the conversation")
	s.Empty(ran["error"])
}

func (s *RelayedSessionSuite) TestAWatcherOnTheOtherNodeRunsOnlyWhatItSaidItRuns() {
	opened := s.inWriting(CreateSessionRequest{
		Llm: pointerTo("tooling/tool-model"),
		Tools: &[]SessionTool{{
			Name: lookupOrder, Description: "find an order by its number",
			Parameters: &map[string]any{"type": "object"},
		}},
	})
	elsewhere := s.serverClient.on(s.other).opens("/v1/agents/sessions/" + opened.Id + "/events?tools=")
	holder := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.respond(opened.Id, "where is my order")

	asked := s.await(elsewhere, "tool_call")
	s.Equal(false, asked["run"], "what it said it runs did not cross the relay")
	s.Require().NoError(elsewhere.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "error": "nothing is registered as lookup_order",
	}))
	s.Equal(true, s.await(holder, "tool_call")["run"])
	s.Require().NoError(holder.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))
	s.Equal("it ships tomorrow", s.await(holder, "tool_ran")["result"])
}

func (s *RelayedSessionSuite) TestClosingTheSocketOnTheOtherNodeEndsTheConversation() {
	opened := s.inWriting(CreateSessionRequest{})
	watching := s.watchFromTheOtherNode(opened.Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "close"}))

	s.Require().Eventually(func() bool {
		status, _ := s.serverClient.call(http.MethodGet, "/v1/agents/sessions/"+opened.Id, nil)
		return status == http.StatusNotFound
	}, settleFor, 10*time.Millisecond, "the conversation outlived the socket that closed it")
}

func (s *RelayedSessionSuite) TestTheSocketClosesWhenTheSessionEndsOnTheOtherNode() {
	opened := s.inWriting(CreateSessionRequest{})
	watching := s.watchFromTheOtherNode(opened.Id)

	s.serverClient.stopSession(opened.Id)

	s.Require().NoError(watching.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		if _, _, err := watching.ReadMessage(); err != nil {
			s.True(websocket.IsCloseError(err, websocket.CloseNormalClosure),
				"a watcher is told the session ended rather than left on a dead socket: %v", err)
			return
		}
	}
}

func (s *RelayedSessionSuite) TestTheOtherNodeWaitsForTheAnswerToItsClose() {
	opened := s.inWriting(CreateSessionRequest{})
	watching := s.watchFromTheOtherNode(opened.Id)

	s.serverClient.stopSession(opened.Id)

	closed, held := s.unanswered(watching)
	s.Equal(websocket.CloseNormalClosure, closed.Code)
	s.Equal("the session ended", closed.Text)
	s.heldUntilAnswered(held)
}

// A frame the socket cannot read ends it on either node, whichever is running the session.
func (s *RelayedSessionSuite) TestAFrameThatIsNotJSONIsRefusedWithPolicyViolation() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)
	s.Require().NoError(watching.WriteMessage(websocket.TextMessage, []byte("not json")))

	s.refusedAsUnreadable(watching)
}

func (s *RelayedSessionSuite) TestJSONThatIsNotACommandIsRefusedWithPolicyViolation() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)
	s.Require().NoError(watching.WriteMessage(websocket.TextMessage, []byte("[1]")))

	s.refusedAsUnreadable(watching)
}

func (s *RelayedSessionSuite) TestABinaryFrameThatIsNotJSONIsRefusedWithPolicyViolation() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)
	s.Require().NoError(watching.WriteMessage(websocket.BinaryMessage, []byte("not json")))

	s.refusedAsUnreadable(watching)
}

// The control: a command of a type the router does not know is reported and the socket
// stays open, as on base 8705bbaf9.
func (s *RelayedSessionSuite) TestACommandOfAnUnknownTypeIsReportedAndTheSocketStaysOpen() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)

	s.unknownCommandLeavesTheSocketOpen(watching, websocket.TextMessage, `{"type":"nope"}`, `unknown command "nope"`)
}

// A command in a binary frame is read as one in a text frame.
func (s *RelayedSessionSuite) TestACommandInABinaryFrameIsReportedAndTheSocketStaysOpen() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)

	s.unknownCommandLeavesTheSocketOpen(watching, websocket.BinaryMessage, `{"type":"nope"}`, `unknown command "nope"`)
}

// An object with no type is a command of no type: reported, not refused.
func (s *RelayedSessionSuite) TestAnObjectWithNoTypeIsReportedAndTheSocketStaysOpen() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)

	s.unknownCommandLeavesTheSocketOpen(watching, websocket.TextMessage, `{}`, `unknown command ""`)
}

// An object whose type is not a string cannot be read as a command and is refused.
func (s *RelayedSessionSuite) TestACommandWhoseTypeIsNotAStringIsRefusedWithPolicyViolation() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)
	s.Require().NoError(watching.WriteMessage(websocket.TextMessage, []byte(`{"type":1}`)))

	s.refusedAsUnreadable(watching)
}

// The control: a caller that closes first is answered with its own close, as on base
// 8705bbaf9.
func (s *RelayedSessionSuite) TestACallerThatClosesTheSocketIsAnsweredWithItsOwnClose() {
	watching := s.watchFromTheOtherNode(s.inWriting(CreateSessionRequest{}).Id)
	s.Require().NoError(watching.WriteMessage(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseGoingAway, "")))

	read, ended := s.socketEnd(watching)

	s.Empty(read)
	s.closedWith(ended, websocket.CloseGoingAway)
}

func (s *RelayedSessionSuite) TestASessionNoNodeIsRunningIsNotFound() {
	_, status := s.serverClient.on(s.other).watch(
		"/v1/agents/sessions/" + s.utils.uuid() + "/events")

	s.Equal(http.StatusNotFound, status)
}

// Who may watch is decided by the node holding the session, so this is the test that the
// node holding the socket is not the one trusted to have asked.
func (s *RelayedSessionSuite) TestAnotherAppCannotWatchASessionFromAnotherNode() {
	opened := s.inWriting(CreateSessionRequest{})

	_, status := s.data.backendOfAnotherApp().on(s.other).watch(
		"/v1/agents/sessions/" + opened.Id + "/events")

	s.Equal(http.StatusNotFound, status, "a stranger is not told the id is real")
}

// inWriting opens a conversation on the suite's own node that keeps no transcript, which
// is the cheapest session with an events socket.
func (s *RelayedSessionSuite) inWriting(request CreateSessionRequest) Session {
	request.Incognito = pointerTo(true)
	if request.Llm == nil {
		request.Llm = pointerTo("llm-flow")
	}
	return s.serverClient.createSession(request)
}

// watchFromTheOtherNode opens the events socket on the node that is running nothing.
func (s *RelayedSessionSuite) watchFromTheOtherNode(id string) *websocket.Conn {
	return s.serverClient.on(s.other).opens("/v1/agents/sessions/" + id + "/events")
}

func (s *RelayedSessionSuite) respond(id, text string) {
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond", SayRequest{Text: text}, nil))
}
