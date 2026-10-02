//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DispatchTextSuite covers an agent that leaves what end users write to the customer's own
// dispatch worker: the worker is handed it, and the model answers only the server.
type DispatchTextSuite struct {
	RouterSuite
}

func TestDispatchTextSuite(t *testing.T) {
	runSuite(t, new(DispatchTextSuite))
}

func (s *DispatchTextSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *DispatchTextSuite) TestWhatAnEndUserWritesIsHandedToTheWorkerWithItsSession() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 4})
	defer release()
	opened, command := s.openedByEndUser(), s.utils.uuid()

	s.writes(opened.Id, frame{"type": "respond", "command_id": command, "text": "Where is my order?"})

	handed := s.handed(worker)
	s.Equal(opened.Id, handed.SessionID, "the worker answers on the session it was written to")
	s.Equal(command, handed.CommandID)
	s.Equal("Where is my order?", handed.Text)
	s.Equal(s.client.userID, handed.UserID)
	s.Equal("thinking", s.receipt(opened.Id, command).State, "the end user is shown it being answered")
}

func (s *DispatchTextSuite) TestTheModelAnswersOnlyOnceTheServerAsksItTo() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 4})
	defer release()
	opened, command := s.openedByEndUser(), s.utils.uuid()
	s.writes(opened.Id, frame{"type": "respond", "command_id": command, "text": "Where is my order?"})
	handed := s.handed(worker)

	s.Never(func() bool {
		return s.receipt(opened.Id, command).State == "completed"
	}, 300*time.Millisecond, 20*time.Millisecond, "the model answered before the server asked it to")

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.actingFor(s.client).do(
		http.MethodPost, "/v1/agents/sessions/"+handed.SessionID+"/responses",
		CreateResponseRequest{CommandId: &handed.CommandID, Text: handed.Text}, &answering))
	s.NotEmpty(answering.Id, "the server's response is the turn that answers the command")
	s.Require().Eventually(func() bool {
		return s.receipt(opened.Id, command).State == "completed"
	}, settleFor, 10*time.Millisecond)
}

func (s *DispatchTextSuite) TestAMessageNoWorkerCanTakeIsWithdrawn() {
	opened, command := s.openedByEndUser(), s.utils.uuid()

	s.writes(opened.Id, frame{"type": "respond", "command_id": command, "text": "Where is my order?"})

	s.Require().Eventually(func() bool {
		var receipt CommandReceipt
		status := s.serverClient.actingFor(s.client).do(http.MethodGet,
			"/v1/agents/sessions/"+opened.Id+"/commands/"+url.PathEscape(command), nil, &receipt)
		return status == http.StatusOK && receipt.State == "cancelled"
	}, settleFor, 10*time.Millisecond,
		"a command nobody will answer must not be left looking like it is being answered")
}

func (s *DispatchTextSuite) TestWhatTheServerWritesIsAnsweredByTheModel() {
	// The server is how a worker answers, so handing its text to a worker would hand the
	// worker its own reply.
	opened, command := s.openedByEndUser(), s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{CommandId: &command, Text: "Where is my order?"}, nil))

	s.Require().Eventually(func() bool {
		return s.receipt(opened.Id, command).State == "completed"
	}, settleFor, 10*time.Millisecond)
}

func (s *DispatchTextSuite) TestAWorkerIsHandedTextOnlySoAnImageIsRefused() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 4})
	defer release()
	opened := s.openedByEndUser()

	watching := s.writes(opened.Id, frame{"type": "respond", "text": "What is this?",
		"images": []map[string]string{{"url": "https://example.com/a.png"}}})

	s.Require().NoError(watching.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var received map[string]any
		s.Require().NoError(watching.ReadJSON(&received))
		if received["type"] == "error" {
			s.Contains(received["error"], "text only")
			break
		}
	}
	s.Empty(worker.Messages(), "half a message is not handed on")
}

// openedByEndUser is a conversation a signed-in end user holds with an agent that leaves
// what they write to dispatch.
func (s *DispatchTextSuite) openedByEndUser() Session {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "en-low-latency", DispatchText: true,
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return s.client.createSession(CreateSessionRequest{Agent: &config.Name, Text: pointerTo(true)})
}

// writes sends a frame over the end user's own socket, which is how a device writes to a
// conversation, and returns the socket for what comes back.
func (s *DispatchTextSuite) writes(sessionID string, sent frame) *websocket.Conn {
	watching := s.client.opens("/v1/agents/sessions/" + sessionID + "/events")
	s.T().Cleanup(func() { _ = watching.Close() })
	s.Require().NoError(watching.WriteJSON(sent))
	return watching
}

// receipt reads a command back as the backend acting for the end user, since reading a
// command is server-side only.
func (s *DispatchTextSuite) receipt(id, command string) CommandReceipt {
	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodGet,
		"/v1/agents/sessions/"+id+"/commands/"+url.PathEscape(command), nil, &receipt))
	return receipt
}

// handed is the next message a worker was given.
func (s *DispatchTextSuite) handed(worker *dispatch.Worker) dispatch.Message {
	select {
	case message := <-worker.Messages():
		return message
	case <-time.After(settleFor):
		s.FailNow("the worker was handed nothing")
		return dispatch.Message{}
	}
}
