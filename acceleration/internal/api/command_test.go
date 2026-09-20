package api

import (
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// answers lets one held reply through.
func (s *SessionAPISuite) answers() {
	s.model.mu.Lock()
	held := s.model.held
	s.model.mu.Unlock()
	s.Require().NotNil(held, "this test forgot to hold the model")
	held <- struct{}{}
}

// writes opens a text session keeping a durable command ledger, with its model held until
// the test lets an answer through.
func (s *SessionAPISuite) writesCommandConversation() Session {
	yes := true
	s.model.mu.Lock()
	s.model.held = make(chan struct{}, 8)
	s.model.mu.Unlock()
	target := "en-low-latency"
	return s.creates(CreateSessionRequest{Text: &yes, PersistConversation: &yes, Llm: &target, Stt: &target, Tts: &target})
}

// submits accepts one durable command and returns its receipt.
func (s *SessionAPISuite) submits(id, commandID, text string) CommandReceipt {
	response := s.send(http.MethodPost, "/v1/agents/sessions/"+id+"/respond", "acme",
		RespondRequest{CommandId: &commandID, Text: text})
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var receipt CommandReceipt
	s.decodeBody(response, &receipt)
	return receipt
}

// stops asks for one named command to stop and returns the raw response.
func (s *SessionAPISuite) stops(id, commandID, customerID string) *http.Response {
	return s.send(http.MethodPost,
		"/v1/agents/sessions/"+id+"/commands/"+url.PathEscape(commandID)+"/interrupt", customerID, nil)
}

func (s *SessionAPISuite) TestStoppingOneCommandLeavesTheCommandAfterItRunning() {
	created := s.writesCommandConversation()
	first := s.submits(created.Id, "command-a", "First question")

	response := s.stops(created.Id, "command-a", "acme")
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var stopped CommandReceipt
	s.decodeBody(response, &stopped)
	s.Equal("cancelled", stopped.State)
	s.Equal(first.AssistantMessageId, stopped.AssistantMessageId)
	s.Equal(first.UserMessageId, stopped.UserMessageId)

	second := s.submits(created.Id, "command-b", "Second question")
	s.NotEqual(first.AssistantMessageId, second.AssistantMessageId)

	// The stop is repeated after a new command started, which is the delayed stop that
	// must not reach the reply now being written.
	repeated := s.stops(created.Id, "command-a", "acme")
	s.Require().Equal(http.StatusOK, repeated.StatusCode)
	var again CommandReceipt
	s.decodeBody(repeated, &again)
	s.Equal(stopped, again)

	live := s.reads(created.Id, "command-b", "acme")
	s.Equal("thinking", live.State, "the running command keeps running")
	s.Equal(second.AssistantMessageId, live.AssistantMessageId)

	s.answers()
	s.Require().Eventually(func() bool {
		return s.reads(created.Id, "command-b", "acme").State == "completed"
	}, settleFor, 10*time.Millisecond, "the running command should finish on its own")
	s.Equal("cancelled", s.reads(created.Id, "command-a", "acme").State)
}

// reads looks a command up through the API.
func (s *SessionAPISuite) reads(id, commandID, customerID string) CommandReceipt {
	response := s.send(http.MethodGet,
		"/v1/agents/sessions/"+id+"/commands/"+url.PathEscape(commandID), customerID, nil)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var receipt CommandReceipt
	s.decodeBody(response, &receipt)
	return receipt
}

func (s *SessionAPISuite) TestACommandNobodySubmittedStopsNothing() {
	created := s.writesCommandConversation()
	running := s.submits(created.Id, "command-a", "First question")

	response := s.stops(created.Id, "command-never-submitted", "acme")
	s.Equal(http.StatusNotFound, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Equal("no such command", failure.Error)

	missing := s.send(http.MethodGet, "/v1/agents/sessions/"+created.Id+"/commands/command-never-submitted", "acme", nil)
	s.Equal(http.StatusNotFound, missing.StatusCode)

	s.Equal("thinking", s.reads(created.Id, "command-a", "acme").State,
		"a stop for a command that does not exist must not touch the one that does")
	s.Equal(running.AssistantMessageId, s.reads(created.Id, "command-a", "acme").AssistantMessageId)
}

func (s *SessionAPISuite) TestAnotherCustomerCannotStopACommand() {
	created := s.writesCommandConversation()
	s.submits(created.Id, "command-a", "First question")

	response := s.stops(created.Id, "command-a", "somebody-else")
	s.Equal(http.StatusNotFound, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Equal("no such session", failure.Error,
		"a caller who may not see the session learns nothing about its commands")

	s.Equal("thinking", s.reads(created.Id, "command-a", "acme").State)
}

func (s *SessionAPISuite) TestAStopWithAnUnknownOutcomeIsNotReportedAsStopped() {
	created := s.writesCommandConversation()
	s.submits(created.Id, "command-a", "First question")

	// The conversation's durable record is made unwritable, so the stop may or may not
	// have taken and nothing may claim it did.
	state := filepath.Join(s.outbox, strings.TrimPrefix(value(created.ConversationId), "agent:"), "state.json")
	s.Require().NoError(os.Remove(state))
	s.Require().NoError(os.Mkdir(state, 0700))

	response := s.stops(created.Id, "command-a", "acme")
	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "persistence outcome unknown")

	s.Require().NoError(os.Remove(state))
	retried := s.stops(created.Id, "command-a", "acme")
	s.Require().Equal(http.StatusOK, retried.StatusCode)
	var receipt CommandReceipt
	s.decodeBody(retried, &receipt)
	s.Equal("cancelled", receipt.State, "the same stop settles once it can be recorded")
}

func (s *SessionAPISuite) TestTheSocketStopsTheCommandItNames() {
	created := s.writesCommandConversation()
	connection := s.watches(created.Id, "acme")
	accepted := s.submits(created.Id, "command-a", "First question")

	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "interrupt", "command_id": "command-a"}))

	frame := s.await(connection, "command_stopped")
	receipt, ok := frame["command"].(map[string]any)
	s.Require().True(ok)
	s.Equal("command-a", receipt["command_id"])
	s.Equal("cancelled", receipt["state"])
	s.Equal(accepted.AssistantMessageId, receipt["assistant_message_id"])
}
