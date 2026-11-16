//go:build integration

package api

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// SessionReopenSettingsSuite covers what a conversation carried on keeps of what its caller
// opened it with, the tool_timeout_ms and the declared tools, and what resumeSession says
// over them. A stopped session is no longer held, so the next request carries it on from its
// row, as a router that never held it would.
type SessionReopenSettingsSuite struct {
	RouterSuite
}

func TestSessionReopenSettingsSuite(t *testing.T) {
	runSuite(t, new(SessionReopenSettingsSuite))
}

func (s *SessionReopenSettingsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionReopenSettingsSuite) TestAConversationCarriedOnFromItsRowKeepsItsToolTimeoutAndTools() {
	ended := s.endedWith(180_000, s.declared("lookup_order"))

	s.respond(ended)

	held := s.heldSpec(ended)
	s.Equal(180_000, held.ToolTimeoutMs)
	s.Equal(s.harnessTools("lookup_order"), held.Tools)
}

func (s *SessionReopenSettingsSuite) TestAResumeWithNothingToSayCarriesItOnWithWhatItRanWith() {
	ended := s.endedWith(180_000, s.declared("lookup_order"))

	s.resume(ended, ResumeSessionRequest{})

	held := s.heldSpec(ended)
	s.Equal(180_000, held.ToolTimeoutMs)
	s.Equal(s.harnessTools("lookup_order"), held.Tools)
}

func (s *SessionReopenSettingsSuite) TestAResumesToolTimeoutWinsOverTheOneItRanWith() {
	ended := s.endedWith(180_000, nil)

	s.resume(ended, ResumeSessionRequest{ToolTimeoutMs: pointerTo(60_000)})

	s.Equal(60_000, s.heldSpec(ended).ToolTimeoutMs)
	s.Eventually(func() bool { return s.storedToolTimeout(ended) == 60_000 }, settleFor, 20*time.Millisecond,
		"the row keeps the resume's, for the next time it is carried on")
}

func (s *SessionReopenSettingsSuite) TestAResumesToolsWinOverTheOnesItWasOpenedWith() {
	ended := s.endedWith(0, s.declared("lookup_order"))

	s.resume(ended, ResumeSessionRequest{Tools: s.declared("book_flight")})

	s.Equal(s.harnessTools("book_flight"), s.heldSpec(ended).Tools)
}

func (s *SessionReopenSettingsSuite) TestAResumeWithNoToolsOffersNone() {
	ended := s.endedWith(0, s.declared("lookup_order"))

	s.resume(ended, ResumeSessionRequest{Tools: &[]SessionTool{}})

	s.Empty(s.heldSpec(ended).Tools)
}

func (s *SessionReopenSettingsSuite) TestAResumedRunningSessionWaitsForItsToolsAsLongAsTheResumeSays() {
	request := textSession(nil)
	request.Llm = pointerTo("tooling/tool-model")
	// No timeout of the tool's own, which would replace the session's.
	request.Tools = &[]SessionTool{{Name: lookupOrder, Description: "find an order by its number"}}
	running := s.serverClient.createSession(request)
	watching := s.serverClient.opens("/v1/agents/sessions/" + running.Id + "/events")

	s.resume(running.Id, ResumeSessionRequest{ToolTimeoutMs: pointerTo(300)})
	s.respond(running.Id)
	s.await(watching, "tool_call")

	// await gives it settleFor, well under the 30 s the session was opened with.
	s.await(watching, "tool_cancel")
}

func (s *SessionReopenSettingsSuite) TestARunningSessionsResumedToolTimeoutOutlivesItsEnding() {
	running := s.serverClient.createSession(textSession(nil))
	s.Require().Eventually(func() bool { return s.stored(running.Id) }, settleFor, 20*time.Millisecond)

	s.resume(running.Id, ResumeSessionRequest{ToolTimeoutMs: pointerTo(60_000)})
	s.Require().Eventually(func() bool { return s.storedToolTimeout(running.Id) == 60_000 }, settleFor, 20*time.Millisecond)
	s.serverClient.stopSession(running.Id)
	s.respond(running.Id)

	s.Equal(60_000, s.heldSpec(running.Id).ToolTimeoutMs)
}

func (s *SessionReopenSettingsSuite) TestAForkOfAnEndedSessionStillHasTheDefaultToolTimeout() {
	ended := s.endedWith(180_000, nil)

	var forked Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+ended+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)}, &forked))

	s.Zero(s.heldSpec(forked.Id).ToolTimeoutMs)
}

func (s *SessionReopenSettingsSuite) TestOnlyItsOwnerResumesASession() {
	owner := s.data.createUser()
	opened := owner.createSession(textSession(nil))

	s.Equal(http.StatusOK, owner.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/resume", ResumeSessionRequest{}, nil))
	s.Equal(http.StatusNotFound,
		s.data.createUser().do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/resume", ResumeSessionRequest{}, nil))
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/resume", ResumeSessionRequest{}, nil)
	})
}

func (s *SessionReopenSettingsSuite) TestANegativeToolTimeoutOnAResumeIsRefused() {
	opened := s.serverClient.createSession(textSession(nil))

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/resume",
		ResumeSessionRequest{ToolTimeoutMs: pointerTo(-1)}, nil))
}

// endedWith is a conversation the backend opened with timeout and tools, whose row was
// written and which was then stopped.
func (s *SessionReopenSettingsSuite) endedWith(timeout int, tools *[]SessionTool) string {
	request := textSession(nil)
	if timeout > 0 {
		request.ToolTimeoutMs = &timeout
	}
	request.Tools = tools
	opened := s.serverClient.createSession(request)
	s.Require().Eventually(func() bool { return s.stored(opened.Id) }, settleFor, 20*time.Millisecond,
		"the row is written")
	s.serverClient.stopSession(opened.Id)
	return opened.Id
}

// declared is one tool of the caller's own by name, with every setting a tool may carry.
func (s *SessionReopenSettingsSuite) declared(name string) *[]SessionTool {
	executor := SessionToolExecutorClient
	return &[]SessionTool{{
		Name: name, Description: "the " + name + " tool",
		Parameters:    &map[string]any{"type": "object"},
		Executor:      &executor,
		DisplayTitle:  pointerTo("Running " + name),
		Interruptible: pointerTo(true),
		TimeoutMs:     pointerTo(5000),
		Approval:      &SessionToolApproval{Title: "Run " + name + "?", Message: pointerTo("It is asked first.")},
	}}
}

// harnessTools is declared as the session's model is offered it.
func (s *SessionReopenSettingsSuite) harnessTools(name string) []harness.Tool {
	return declaredTools(*s.declared(name))
}

func (s *SessionReopenSettingsSuite) resume(id string, request ResumeSessionRequest) {
	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+id+"/resume", request)
	s.Require().Equal(http.StatusOK, status, string(body))
}

func (s *SessionReopenSettingsSuite) respond(id string) {
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond", SayRequest{Text: "where is my order"}, nil))
}

// heldSpec is the spec of the session id this node holds.
func (s *SessionReopenSettingsSuite) heldSpec(id string) session.Spec {
	held, ok := s.manager.Get(id, session.Owner{CustomerID: s.customerID(), Kind: auth.KindServer})
	s.Require().True(ok, "session %s is held here", id)
	return held.Spec()
}

func (s *SessionReopenSettingsSuite) stored(id string) bool {
	var found string
	return s.store.ShardDB().QueryRowContext(context.Background(),
		"SELECT id FROM agent_sessions WHERE id = ?", id).Scan(&found) == nil
}

// storedToolTimeout is the row's tool_timeout_ms, zero for none or no row.
func (s *SessionReopenSettingsSuite) storedToolTimeout(id string) int {
	var timeout sql.NullInt64
	err := s.store.ShardDB().QueryRowContext(context.Background(),
		"SELECT tool_timeout_ms FROM agent_sessions WHERE id = ?", id).Scan(&timeout)
	if errors.Is(err, sql.ErrNoRows) {
		return 0
	}
	s.Require().NoError(err)
	return int(timeout.Int64)
}
