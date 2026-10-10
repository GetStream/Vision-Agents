//go:build integration

package api

import (
	"net/http"
	"net/url"
	"testing"
	"time"
)

// SessionCommandsSuite covers the durable command ledger: a caller naming what it asked
// for so the answer can be found again, and stopped by name while it is being written.
type SessionCommandsSuite struct {
	RouterSuite
}

func TestSessionCommandsSuite(t *testing.T) {
	runSuite(t, new(SessionCommandsSuite))
}

func (s *SessionCommandsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionCommandsSuite) TestAResponseCanBeAskedForAsANamedCommand() {
	opened, command := s.writing(), s.utils.uuid()

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: "First question"}, nil))

	s.Equal("thinking", s.receipt(opened.Id, command).State,
		"the response is the command, so its receipt says it is being answered")
}

func (s *SessionCommandsSuite) TestTheSameCommandAskedTwiceWhileItRunsIsAConflict() {
	opened, command := s.writing(), s.utils.uuid()
	s.submit(opened.Id, command, "First question")

	status, _ := s.serverClient.call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: "Another question"})

	s.Equal(http.StatusConflict, status)
}

func (s *SessionCommandsSuite) TestACommandCarriesTextAndNothingElse() {
	opened, command := s.writing(), s.utils.uuid()

	status, _ := s.serverClient.call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses", CreateResponseRequest{
			RequestId: &command, Text: "What is this?",
			Images: &[]ImageSource{{Url: "https://example.com/a.png"}},
		})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SessionCommandsSuite) TestASessionThatKeepsNothingAnswersAQuestionCarryingARequestID() {
	// Every SDK sends a request id with every question, whatever the session keeps.
	request := textSession(nil)
	request.Incognito = pointerTo(true)
	opened, command := s.serverClient.createSession(request), s.utils.uuid()

	s.Equal(http.StatusAccepted, s.serverClient.do(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: "First question"}, nil))
}

func (s *SessionCommandsSuite) TestACommandLeftAloneFinishesOnItsOwn() {
	opened, command := s.writing(), s.utils.uuid()
	s.submit(opened.Id, command, "First question")

	s.Require().Eventually(func() bool {
		return s.receipt(opened.Id, command).State == "completed"
	}, settleFor, 10*time.Millisecond)
}

func (s *SessionCommandsSuite) TestStoppingOneCommandLeavesTheCommandAfterItRunning() {
	opened := s.writing()
	first, second := s.utils.uuid(), s.utils.uuid()
	asked := s.submit(opened.Id, first, "First question")

	stopped := s.stop(opened.Id, first)
	s.Equal("cancelled", stopped.State)
	s.Equal(asked.AssistantMessageId, stopped.AssistantMessageId)
	s.Equal(asked.UserMessageId, stopped.UserMessageId)

	running := s.submit(opened.Id, second, "Second question")
	s.NotEqual(asked.AssistantMessageId, running.AssistantMessageId)
	s.Equal("thinking", s.receipt(opened.Id, second).State)
}

func (s *SessionCommandsSuite) TestAStopThatArrivesLateDoesNotReachTheReplyBeingWritten() {
	opened := s.writing()
	first, second := s.utils.uuid(), s.utils.uuid()
	s.submit(opened.Id, first, "First question")
	stopped := s.stop(opened.Id, first)
	s.submit(opened.Id, second, "Second question")

	s.Equal(stopped, s.stop(opened.Id, first), "the same stop settles the same way")

	s.Equal("thinking", s.receipt(opened.Id, second).State, "the running command keeps running")
	s.Require().Eventually(func() bool {
		return s.receipt(opened.Id, second).State == "completed"
	}, settleFor, 10*time.Millisecond, "the running command should finish on its own")
	s.Equal("cancelled", s.receipt(opened.Id, first).State)
}

func (s *SessionCommandsSuite) TestACommandNobodySubmittedStopsNothing() {
	opened, command := s.writing(), s.utils.uuid()
	asked := s.submit(opened.Id, command, "First question")

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/commands/"+s.utils.uuid()+"/interrupt", nil)

	s.Equal(http.StatusNotFound, status)
	s.Equal("no such command", failure)
	s.Equal("thinking", s.receipt(opened.Id, command).State,
		"a stop for a command that does not exist must not touch the one that does")
	s.Equal(asked.AssistantMessageId, s.receipt(opened.Id, command).AssistantMessageId)
}

func (s *SessionCommandsSuite) TestACommandNobodySubmittedIsNotFound() {
	opened := s.writing()

	status, _ := s.serverClient.call(http.MethodGet,
		"/v1/agents/sessions/"+opened.Id+"/commands/"+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, status)
}

func (s *SessionCommandsSuite) TestAnotherAppCannotStopACommand() {
	opened, command := s.writing(), s.utils.uuid()
	s.submit(opened.Id, command, "First question")

	status, failure := s.data.backendOfAnotherApp().failure(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/commands/"+url.PathEscape(command)+"/interrupt", nil)

	s.Equal(http.StatusNotFound, status)
	s.Equal("no such session", failure,
		"a caller who may not see the session learns nothing about its commands")
	s.Equal("thinking", s.receipt(opened.Id, command).State)
}

func (s *SessionCommandsSuite) TestTheSocketStopsTheCommandItNames() {
	opened, command := s.writing(), s.utils.uuid()
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	asked := s.submit(opened.Id, command, "First question")

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "interrupt", "request_id": command,
	}))

	s.Require().NoError(watching.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var received map[string]any
		s.Require().NoError(watching.ReadJSON(&received))
		if received["type"] != "command_stopped" {
			continue
		}
		receipt, ok := received["command"].(map[string]any)
		s.Require().True(ok)
		s.Equal(command, receipt["request_id"])
		s.Equal("cancelled", receipt["state"])
		s.Equal(asked.AssistantMessageId, receipt["assistant_message_id"])
		return
	}
}

// writing opens a conversation in writing on the model that is a while in the writing, so
// a command is still being answered when the test asks about it.
func (s *SessionCommandsSuite) writing() Session {
	request := textSession(nil)
	request.Llm = pointerTo("slow/slow-model")
	return s.serverClient.createSession(request)
}

// submit accepts one durable command and returns its receipt.
func (s *SessionCommandsSuite) submit(id, command, text string) CommandReceipt {
	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond",
		RespondRequest{RequestId: &command, Text: text}, &receipt))
	return receipt
}

func (s *SessionCommandsSuite) receipt(id, command string) CommandReceipt {
	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/sessions/"+id+"/commands/"+url.PathEscape(command), nil, &receipt))
	return receipt
}

func (s *SessionCommandsSuite) stop(id, command string) CommandReceipt {
	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/commands/"+url.PathEscape(command)+"/interrupt", nil, &receipt))
	return receipt
}
