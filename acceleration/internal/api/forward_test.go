//go:build integration

package api

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// ForwardedSessionSuite covers what is asked of a session over HTTP when the request does
// not land on the node running the conversation.
//
// Every session a test here opens is opened on the other node and then asked about on the
// suite's own, which is what a load balancer with no reason to prefer either produces
// about half the time.
type ForwardedSessionSuite struct {
	RouterSuite

	// other is the node running every session a test here opens.
	other *httptest.Server
}

func TestForwardedSessionSuite(t *testing.T) {
	runSuite(t, new(ForwardedSessionSuite))
}

func (s *ForwardedSessionSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.other = s.otherNode()
}

func (s *ForwardedSessionSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ForwardedSessionSuite) TestASessionIsReadBackFromANodeThatIsNotRunningIt() {
	opened := s.onTheOtherNode(CreateSessionRequest{})

	read := s.serverClient.getSession(opened.Id)

	// The session keeps no row, so there is nothing here to answer from: this is the
	// running conversation or it is a 404.
	s.Equal(opened.Id, read.Id)
	s.Equal(opened.Llm, read.Llm)
}

func (s *ForwardedSessionSuite) TestASessionIsToldToAnswerFromANodeThatIsNotRunningIt() {
	answer := s.utils.uuid()
	opened := s.onTheOtherNode(CreateSessionRequest{
		Llm: pointerTo("echo/echo-model"), ConfigId: s.data.instructedAgent(answer),
	})
	// Watched from here too, which the relay serves. What is under test is that the
	// request to answer reached the conversation, and the answer is where that shows.
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", SayRequest{Text: "hello"}, nil))

	answered := s.await(watching, "responded")
	s.Contains(answered["text"], answer, "the conversation never heard what it was asked")
}

// Deleting is the operation that used to do damage from the wrong node: the row says the
// session is the caller's, so it was authorized here, and then the rows, the turns and
// what the session remembered were deleted while the node running it carried on against
// them. So this is recorded rather than incognito, which is what gives it rows to lose.
func (s *ForwardedSessionSuite) TestDeletingASessionFromAnotherNodeEndsItWhereItIsRunning() {
	opened := s.serverClient.on(s.other).createSession(CreateSessionRequest{
		Llm: pointerTo("llm-flow"),
	})

	s.serverClient.deleteSession(opened.Id)

	status, _ := s.serverClient.on(s.other).call(
		http.MethodGet, "/v1/agents/sessions/"+opened.Id, nil)
	s.Equal(http.StatusNotFound, status, "the conversation outlived the rows it was writing")
}

func (s *ForwardedSessionSuite) TestASessionNoNodeIsRunningIsAnsweredHere() {
	status, message := s.serverClient.failure(
		http.MethodGet, "/v1/agents/sessions/"+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, status)
	s.Equal(errUnknownSession.Message, message)
}

// Who may touch a session is decided by the node running it, so this is the test that the
// node taking the request is not the one trusted to have asked.
func (s *ForwardedSessionSuite) TestAnotherAppCannotReachASessionOnAnotherNode() {
	opened := s.onTheOtherNode(CreateSessionRequest{})

	status, _ := s.data.backendOfAnotherApp().call(
		http.MethodGet, "/v1/agents/sessions/"+opened.Id, nil)

	s.Equal(http.StatusNotFound, status, "a stranger is not told the id is real")
}

// A message arriving where the conversation is not used to find nothing running and be
// handed to a worker, which is a second agent writing into a conversation the first is
// already answering in.
func (s *ForwardedSessionSuite) TestAMessageArrivingOnTheWrongNodeIsAnsweredByTheOneRunningTheConversation() {
	channel, written := "chat-"+s.utils.uuid(), "is anybody there?"
	// A message in writing is answered off the call rather than through a turn, so where
	// it shows is the model being asked rather than anything a watcher is sent.
	s.onTheOtherNode(CreateSessionRequest{
		AgentId: &channel, Llm: pointerTo("vision/vision-model"),
	})
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK,
		s.signedly("/v1/chat/hooks/stream", writtenTo(channel, written)))

	s.Require().Eventually(func() bool { return s.wasAsked(written) },
		settleFor, 10*time.Millisecond,
		"the message never reached the conversation on the node running it")
	// Whether to hand this to a worker was settled before the delivery was answered, so
	// there is nothing left to wait for.
	select {
	case message := <-worker.Messages():
		s.Failf("a second agent was started", "%s was given to a worker as well", message.ChannelID)
	default:
	}
}

// wasAsked reports whether the model was given something to answer.
func (s *ForwardedSessionSuite) wasAsked(text string) bool {
	for _, request := range s.vision.requests() {
		for _, message := range request.Input {
			if message.Content == text {
				return true
			}
		}
	}
	return false
}

// onTheOtherNode opens an incognito conversation in writing on the node the suite's
// clients do not send to.
//
// Incognito because a session with no row is one only the node running it can answer for,
// which is what makes a 200 here proof that the request was carried there.
func (s *ForwardedSessionSuite) onTheOtherNode(request CreateSessionRequest) Session {
	request.Incognito = pointerTo(true)
	if request.Llm == nil {
		request.Llm = pointerTo("llm-flow")
	}
	return s.serverClient.on(s.other).createSession(request)
}

// writtenTo is a message.new for a channel, as Stream would deliver it.
func writtenTo(channel, text string) string {
	return fmt.Sprintf(`{
  "type": "message.new",
  "cid": "agent:%s",
  "channel_id": "%s",
  "channel_type": "agent",
  "channel_custom": {},
  "message": {
    "id": "message-1",
    "text": "%s",
    "user": {"id": "sam", "name": "Sam"}
  }
}`, channel, channel, text)
}
