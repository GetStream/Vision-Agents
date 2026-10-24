//go:build integration

package api

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

func (s *SessionEventsSuite) TestOnlyAWatcherThatRunsTheToolIsAskedToRunIt() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("")
	s.respondFor(opened.Id)

	s.Equal(true, s.await(backend, "tool_call")["run"])
	s.Equal(false, s.await(browser, "tool_call")["run"], "the browser sees the call, to show it, and is not asked to run it")
}

func (s *SessionEventsSuite) TestTheHoldersResultIsTheOneTheModelSeesWhoeverAnswersFirst() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("")
	s.respondFor(opened.Id)
	asked := s.await(browser, "tool_call")
	s.await(backend, "tool_call")

	s.Require().NoError(browser.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "error": "nothing is registered as lookup_order",
	}))
	s.Require().NoError(backend.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))

	ran := s.await(backend, "tool_ran")
	s.Equal("it ships tomorrow", ran["result"])
	s.Empty(ran["error"])
}

func (s *SessionEventsSuite) TestABrowserThatDeclaresTheBackendsToolIsNeverAskedToRunIt() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("?tools=" + lookupOrder)
	s.respondFor(opened.Id)
	asked := s.await(browser, "tool_call")
	s.await(backend, "tool_call")

	s.Equal(false, asked["run"], "a tool a backend declared runs with the backend's credentials")
	s.Require().NoError(browser.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "run in the browser",
	}))
	s.Require().NoError(backend.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))
	s.Equal("it ships tomorrow", s.await(backend, "tool_ran")["result"])
}

func (s *SessionEventsSuite) TestAWatcherSaysWhichToolsItRunsWhenItAttaches() {
	opened := s.withATool()
	observer := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events?tools=")
	holder := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events?tools=" + lookupOrder)
	s.respond(opened.Id)

	s.Equal(false, s.await(observer, "tool_call")["run"], "it said it runs nothing")
	s.Equal(true, s.await(holder, "tool_call")["run"])
}

func (s *SessionEventsSuite) TestACallNobodyWatchingCanRunFailsAtOnce() {
	opened := s.serverClient.actingFor(s.client).createSession(s.forItsUser(s.askingForATool()))
	browser := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	started := time.Now()
	s.respondFor(opened.Id)

	ran := s.await(browser, "tool_ran")
	s.NotEmpty(ran["error"])
	s.Less(time.Since(started), defaultToolWait/2, "the model is not left waiting out the timeout")
}

// defaultToolWait is how long the router waits on a caller's tool that names no timeout.
const defaultToolWait = 30 * time.Second

// backendsToolWatchedByItsUser is a conversation a backend opened for the suite's user with
// its own tool, watched by the backend and by the user's browser, which attaches with query.
func (s *SessionEventsSuite) backendsToolWatchedByItsUser(query string) (Session, *websocket.Conn, *websocket.Conn) {
	opened := s.serverClient.actingFor(s.client).createSession(s.forItsUser(s.askingForATool()))
	backend := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	browser := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events" + query)
	return opened, backend, browser
}

// forItsUser is request kept out of the transcript, on the model that reaches for the tool.
func (s *SessionEventsSuite) forItsUser(request CreateSessionRequest) CreateSessionRequest {
	request.Incognito = pointerTo(true)
	return request
}

func (s *SessionEventsSuite) respondFor(id string) {
	s.Require().Equal(http.StatusNoContent, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond", SayRequest{Text: "where is my order"}, nil))
}
