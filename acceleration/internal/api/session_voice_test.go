//go:build integration

package api

import (
	"net/http"
	"testing"
)

// SessionVoiceSuite covers starting and stopping voice on a session: the agent joins the
// call named after the session and leaves it, and the conversation is one either way.
type SessionVoiceSuite struct {
	RouterSuite
}

func TestSessionVoiceSuite(t *testing.T) {
	runSuite(t, new(SessionVoiceSuite))
}

func (s *SessionVoiceSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionVoiceSuite) TestStartingVoicePutsTheAgentOnTheCallNamedAfterTheSession() {
	opened := s.serverClient.createSession(textSession(nil))
	s.Empty(opened.CallId, "a session held in writing is on no call")

	started := s.startVoice(opened.Id)

	s.Equal(opened.Id, started.CallId)
	s.Equal(value(opened.ConversationId), value(started.ConversationId), "the conversation stays in its channel")
	s.Equal(opened.Id, s.serverClient.getSession(opened.Id).CallId)
}

func (s *SessionVoiceSuite) TestASessionHeldByAStringIdIsOnTheCallNamedAfterIt() {
	id := "order_4471-" + s.utils.uuid()[24:]
	request := textSession(&id)
	request.StartVoice = pointerTo(true)

	s.Equal(id, s.serverClient.createSession(request).CallId)
}

func (s *SessionVoiceSuite) TestASessionCreatedToStartVoiceIsOnItsCall() {
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true)})

	s.Equal(opened.Id, opened.CallId)
}

func (s *SessionVoiceSuite) TestStoppingVoiceCarriesTheConversationOnInWriting() {
	request := textSession(nil)
	request.Llm = pointerTo("recites/recites-model")
	opened := s.serverClient.createSession(request)
	s.startVoice(opened.Id)
	s.answerTo(opened.Id, "Where is order 4471?")

	var stopped Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodDelete,
		"/v1/agents/sessions/"+opened.Id+"/voice", nil, &stopped))

	s.Empty(stopped.CallId)
	s.Equal(value(opened.ConversationId), value(stopped.ConversationId))
	s.Contains(s.answerTo(opened.Id, "And when?"), "Where is order 4471?",
		"what was typed while voice was on is part of the conversation")
}

func (s *SessionVoiceSuite) TestStartingVoiceTwiceKeepsTheAgentOnTheSameCall() {
	opened := s.serverClient.createSession(textSession(nil))
	s.startVoice(opened.Id)

	again := s.startVoice(opened.Id)

	s.Equal(opened.Id, again.CallId)
}

func (s *SessionVoiceSuite) TestStartingVoiceOnASessionNobodyHasIsNotFound() {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+s.utils.uuid()+"/voice", nil)

	s.Equal(http.StatusNotFound, status)
}

// startVoice starts voice on a session and returns what it says of itself.
func (s *SessionVoiceSuite) startVoice(id string) Session {
	var started Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/voice", nil, &started))
	return started
}

// answerTo asks a running session a question in writing and returns what it answered.
func (s *SessionVoiceSuite) answerTo(id, question string) string {
	watching := s.serverClient.opens("/v1/agents/sessions/" + id + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/responses", CreateResponseRequest{Text: question}, nil))
	answered, _ := s.await(watching, "responded")["text"].(string)
	return answered
}
