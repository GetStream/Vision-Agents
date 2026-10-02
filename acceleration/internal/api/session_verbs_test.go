//go:build integration

package api

import (
	"net/http"
	"testing"
	"time"
)

// SessionVerbsSuite covers what can be done to a session once it is open: speaking,
// answering, changing what it runs on, renaming it, and carrying it on somewhere else.
type SessionVerbsSuite struct {
	RouterSuite
}

func TestSessionVerbsSuite(t *testing.T) {
	runSuite(t, new(SessionVerbsSuite))
}

func (s *SessionVerbsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionVerbsSuite) TestSayingSomethingSpeaksItWithoutTheModel() {
	said := "one moment, " + s.utils.uuid()
	opened := s.onACall()

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/say", SayRequest{Text: said}, nil))

	s.Require().Eventually(func() bool {
		for _, spoken := range s.voice.spoken() {
			if spoken == said {
				return true
			}
		}
		return false
	}, settleFor, 5*time.Millisecond, "nothing was said")
}

func (s *SessionVerbsSuite) TestSayingNothingIsRefused() {
	opened := s.onACall()

	status, failure := s.serverClient.failure(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/say", SayRequest{Text: ""})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nothing to say")
}

func (s *SessionVerbsSuite) TestChangingASessionsModelsLeavesEveryOtherSessionOnItsOwn() {
	opened, untouched := s.onACall(), s.onACall()

	var changed Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(
		http.MethodPatch, "/v1/agents/sessions/"+opened.Id+"/settings",
		SessionSettingsRequest{Llm: pointerTo("vision/vision-model"), Voice: pointerTo("ada")}, &changed))

	s.Equal("vision/vision-model", value(changed.Llm))
	s.Equal("ada", value(changed.Voice))
	s.Equal("vision/vision-model", value(s.serverClient.getSession(opened.Id).Llm))
	s.NotEqual("vision/vision-model", value(s.serverClient.getSession(untouched.Id).Llm),
		"only the session asked about changes")
}

func (s *SessionVerbsSuite) TestAModelNobodyDeployedIsRefusedAndChangesNothing() {
	opened := s.onACall()
	was := s.serverClient.getSession(opened.Id)

	status, _ := s.serverClient.call(http.MethodPatch,
		"/v1/agents/sessions/"+opened.Id+"/settings",
		SessionSettingsRequest{Sts: pointerTo("openai/gpt-realtime-2")})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(value(was.Mode), value(s.serverClient.getSession(opened.Id).Mode),
		"a refused change leaves the session as it was")
}

func (s *SessionVerbsSuite) TestAConversationInWritingHasNoVoiceToChange() {
	opened := s.serverClient.createSession(textSession(nil))

	status, _ := s.serverClient.call(http.MethodPatch,
		"/v1/agents/sessions/"+opened.Id+"/settings",
		SessionSettingsRequest{Tts: pointerTo("en-low-latency")})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SessionVerbsSuite) TestChangingASessionNobodyOpenedIsNotFound() {
	status, _ := s.serverClient.call(http.MethodPatch,
		"/v1/agents/sessions/"+s.utils.uuid()+"/settings",
		SessionSettingsRequest{Llm: pointerTo("vision/vision-model")})

	s.Equal(http.StatusNotFound, status)
}

func (s *SessionVerbsSuite) TestASessionIsRenamedAndMovedInOneRequest() {
	opened := s.serverClient.createSession(textSession(nil))

	s.Require().Equal(http.StatusOK, s.serverClient.do(
		http.MethodPatch, "/v1/agents/sessions/"+opened.Id, UpdateSessionRequest{
			Title:        pointerTo("Pricing"),
			Description:  pointerTo("Asked twice"),
			Custom:       &map[string]any{"pinned": true},
			Instructions: pointerTo("Answer in French."),
			Llm:          pointerTo("vision/vision-model"),
		}, nil))

	read := s.serverClient.getSession(opened.Id)
	s.Equal("Pricing", value(read.Title))
	s.Equal("Asked twice", value(read.Description))
	s.Equal(map[string]any{"pinned": true}, value(read.Custom))
	s.Equal("Answer in French.", value(read.Instructions))
	s.Equal("vision/vision-model", value(read.Llm))
}

func (s *SessionVerbsSuite) TestASessionBeingRecordedCannotTakeItBack() {
	opened := s.serverClient.createSession(textSession(nil))

	var updated Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(
		http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
		map[string]any{"id": s.utils.uuid(), "incognito": true, "title": "Renamed"}, &updated))

	s.Equal(opened.Id, updated.Id, "an update cannot move a session to another id either")
	s.Nil(updated.Incognito)
	s.Equal("Renamed", value(updated.Title))
}

func (s *SessionVerbsSuite) TestARefusedModelLeavesTheWholeUpdateUndone() {
	opened := s.serverClient.createSession(textSession(nil))
	s.serverClient.updateSession(opened.Id, UpdateSessionRequest{Title: pointerTo("First ask")})

	status, _ := s.serverClient.call(http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
		UpdateSessionRequest{Title: pointerTo("Renamed"), Sts: pointerTo("openai/gpt-realtime-2")})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("First ask", value(s.serverClient.getSession(opened.Id).Title))
}

func (s *SessionVerbsSuite) TestAnotherAppsSessionIsNotTheirsToRename() {
	opened := s.serverClient.createSession(textSession(nil))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
			UpdateSessionRequest{Title: pointerTo("Mine now")}, nil)
	})
}

func (s *SessionVerbsSuite) TestAskingForAResponseNamesTheTurnItAnswersAs() {
	opened := s.serverClient.createSession(textSession(nil))

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: "Is Stream better than Sendbird?"}, &answering))

	s.Equal(opened.Id, answering.SessionId)
	s.Equal(AgentResponseStatusRunning, answering.Status,
		"it answers once the turn has started, not once it has finished")
	s.NotEmpty(answering.Id, "the turn it will be written down as")
}

func (s *SessionVerbsSuite) TestAResponseWithNothingToAnswerIsRefused() {
	opened := s.serverClient.createSession(textSession(nil))

	status, _ := s.serverClient.call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses", CreateResponseRequest{})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SessionVerbsSuite) TestAnotherAppCannotAskASessionAnything() {
	opened := s.serverClient.createSession(textSession(nil))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
			CreateResponseRequest{Text: "What did they ask?"}, nil)
	})
}

func (s *SessionVerbsSuite) TestAForkSaysWhatItCameFrom() {
	request := textSession(nil)
	request.Title, request.ProjectId = pointerTo("The first ask"), pointerTo("Health")
	opened := s.serverClient.createSession(request)

	var forked Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{Title: pointerTo("Asked again")}, &forked))

	s.NotEqual(opened.Id, forked.Id)
	s.Equal(opened.Id, value(forked.ForkedFrom))
	s.Equal("Asked again", value(forked.Title), "the fork's own title wins")
	s.Equal("Health", value(forked.ProjectId), "what the fork did not mention it inherits")
}

func (s *SessionVerbsSuite) TestForkingASessionThatRecordsNothingIsRefused() {
	request := textSession(nil)
	request.Incognito = pointerTo(true)
	opened := s.serverClient.createSession(request)

	status, failure := s.serverClient.failure(
		http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork", ForkSessionRequest{})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nothing to fork from")
}

func (s *SessionVerbsSuite) TestAForkAtAResponseCannotLeaveTheHistoryBehind() {
	// Forking at a turn is forking the conversation up to it, so asking for the turn
	// without the messages is asking for two different things at once.
	opened := s.serverClient.createSession(textSession(nil))

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{ResponseId: pointerTo("first"), Messages: pointerTo(false)})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "messages false")
}

func (s *SessionVerbsSuite) TestRewindingNeedsAResponseToCarryOnFrom() {
	opened := s.serverClient.createSession(textSession(nil))

	status, _ := s.serverClient.call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/rewind", RewindSessionRequest{})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SessionVerbsSuite) TestRewindingToATurnNobodyTookIsRefused() {
	opened := s.serverClient.createSession(textSession(nil))

	status, _ := s.serverClient.call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/rewind", RewindSessionRequest{ResponseId: s.utils.uuid()})

	s.Equal(http.StatusBadRequest, status)
}

func (s *SessionVerbsSuite) TestAnotherAppCannotRewindASession() {
	opened := s.serverClient.createSession(textSession(nil))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/rewind",
			RewindSessionRequest{ResponseId: "first"}, nil)
	})
}

// onACall is a session with an agent in a call, which is the one that can speak.
func (s *SessionVerbsSuite) onACall() Session {
	call := s.utils.callID()
	return s.serverClient.createSession(CreateSessionRequest{CallId: &call})
}
