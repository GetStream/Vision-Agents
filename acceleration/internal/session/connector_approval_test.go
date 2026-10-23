//go:build integration

package session

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorApprovalSuite is a call of a tool its binding's require_approval names: the
// dispatcher holds it until the session's user answers on its step, and sends it only once
// they allow it. The conversation and the socket are run end to end in the api package
// (connector_approvals_test.go); here the dispatcher is handed the conversation's hooks, which
// ask alice for the command "command-1".
type ConnectorApprovalSuite struct {
	connectorFixture
	mu sync.Mutex
	// asked are the questions shown on calls' steps, settled how each was settled, and stopped
	// the calls whose reply was stopped, by call id.
	asked   map[string]persistent.ToolApproval
	settled map[string]agent.ToolApprovalDecided
	stopped map[string]bool
}

func TestConnectorApprovalSuite(t *testing.T) {
	suite.Run(t, new(ConnectorApprovalSuite))
}

func (s *ConnectorApprovalSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.asked, s.settled, s.stopped = map[string]persistent.ToolApproval{}, map[string]agent.ToolApprovalDecided{}, map[string]bool{}
}

func (s *ConnectorApprovalSuite) TestACallIsSentOnlyOnceTheUserAllowsIt() {
	d, connection := s.holding("whoami")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return len(answer) > 0 }, 200*time.Millisecond, 20*time.Millisecond, "it waits for the person")
	s.Zero(s.provider.calls("primary"), "nothing reached the provider before they answered")
	s.True(d.answerApproval("call-1", "command-1", true, ""))
	got := s.answerOf(answer)

	s.Require().NoError(got.err)
	s.Equal("primary", got.said)
	s.Equal(persistent.ToolApproval{Title: "Allow crm to run whoami?"}, *s.questionOf("call-1"))
	s.Equal(agent.ToolApprovalDecided{ID: "call-1", Allowed: true}, s.settledOf("call-1"))
	s.Equal(1, s.provider.calls("primary"))
	s.Empty(s.rows(connection)[0].ErrorType)
}

func (s *ConnectorApprovalSuite) TestADeclinedCallIsNotSentAndTheModelIsTold() {
	d, connection := s.holding("whoami")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.True(d.answerApproval("call-1", "command-1", false, "Not now"))
	got := s.answerOf(answer)

	s.Require().NoError(got.err)
	s.JSONEq(`{"status":"declined","message":"The user declined crm__whoami, so it was not run. Do not call it again unless they ask."}`, got.said)
	s.Equal(agent.ToolApprovalDecided{ID: "call-1", Summary: "Not now"}, s.settledOf("call-1"))
	s.Zero(s.provider.calls("primary"))
	row := s.rows(connection)[0]
	s.Equal(store.InvocationDenied, row.ErrorType)
	s.Equal("declined by the person", row.ErrorMessage)
}

// An answer is matched on the call and on the command its reply answers, so an answer for
// another command, or for a call that is not held, moves nothing; and a call is answered once.
func (s *ConnectorApprovalSuite) TestAnAnswerNamingAnotherCommandOrCallIsRefused() {
	d, _ := s.holding("whoami")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)

	s.False(d.answerApproval("call-1", "command-2", true, ""), "another command")
	s.False(d.answerApproval("call-1", "", true, ""), "no command")
	s.False(d.answerApproval("call-2", "command-1", true, ""), "a call that is not held")
	s.Never(func() bool { return len(answer) > 0 }, 200*time.Millisecond, 20*time.Millisecond, "still waiting")
	s.True(d.answerApproval("call-1", "command-1", false, ""))
	s.False(d.answerApproval("call-1", "command-1", true, ""), "answered already")
	s.Contains(s.answerOf(answer).said, `"status":"declined"`)
	s.Zero(s.provider.calls("primary"))
}

// Nobody answered within the wait, ten minutes in production: the call is declined, its step
// says so, and the model is told.
func (s *ConnectorApprovalSuite) TestACallNobodyAllowsInTimeIsDeclined() {
	d, connection := s.holding("whoami")
	d.approvals.wait = 300 * time.Millisecond

	got := s.answerOf(s.callInBackground(s.ctx, d, "call-1", "crm__whoami"))

	s.Require().NoError(got.err)
	s.JSONEq(`{"status":"declined","message":"Nobody allowed crm__whoami in time, so it was declined and not run. Tell the user, and that they can ask again."}`, got.said)
	s.Equal(agent.ToolApprovalDecided{ID: "call-1", Summary: "Not allowed in time"}, s.settledOf("call-1"))
	s.Zero(s.provider.calls("primary"))
	s.Equal("declined by timeout", s.rows(connection)[0].ErrorMessage)
}

// A session with no conversation, such as a headless run, has nobody to ask: the call is
// refused at once, recorded as declined.
func (s *ConnectorApprovalSuite) TestACallWithNobodyToAskIsRefusedAtOnce() {
	connection := s.connection("", "primary")
	binding := s.fixed("crm", connection, "whoami")
	binding.RequireApproval = []string{"whoami"}
	d, _, _, err := s.attach(s.spec(s.config(binding), "alice", nil))
	s.Require().NoError(err)

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.JSONEq(`{"status":"unavailable","message":"crm__whoami runs only once the user allows it, and nobody can be asked here, so it was not run. Tell the user it needs their approval in a chat of their own."}`, said)
	s.Zero(s.provider.calls("primary"))
	row := s.rows(connection)[0]
	s.Equal(store.InvocationDenied, row.ErrorType)
	s.Equal("declined: nobody can approve it here", row.ErrorMessage)
}

// The conversation cannot ask the session's user, as in a shared chat or a reply to an MCP
// event: refused at once too.
func (s *ConnectorApprovalSuite) TestACallTheConversationCannotAskAboutIsRefusedAtOnce() {
	binding := s.fixed("crm", s.connection("", "primary"), "whoami")
	binding.RequireApproval = []string{"whoami"}
	d := s.asking(s.config(binding), "bob")

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Contains(said, `"status":"unavailable"`)
	s.Zero(s.provider.calls("primary"))
}

// An interruption ends the wait, whatever the binding's policy: the call was never sent.
func (s *ConnectorApprovalSuite) TestATurnThatEndsStopsTheWait() {
	d, connection := s.holdingWith(func(binding *store.ConnectorBinding) {
		binding.Policy = &store.BindingPolicy{OnInterrupt: store.InterruptWait}
	}, "whoami")
	turn, end := context.WithCancel(s.ctx)
	defer end()

	answer := s.callInBackground(turn, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	end()
	got := s.answerOf(answer)

	s.ErrorIs(got.err, context.Canceled)
	s.Zero(s.provider.calls("primary"))
	s.Equal(store.InvocationClientTimeout, s.rows(connection)[0].ErrorType)
	s.False(d.answerApproval("call-1", "command-1", true, ""), "the call no longer waits")
}

// The reply ended and left the call running, as an interruption that lets tools finish does
// (agent.stopTools). The call stops waiting by itself, and is not run for nobody.
func (s *ConnectorApprovalSuite) TestAStoppedReplysCallStopsWaiting() {
	d, _ := s.holding("whoami")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.stop("call-1")
	got := s.answerOf(answer)

	s.Require().NoError(got.err)
	s.Contains(got.said, `"status":"stopped"`)
	s.Zero(s.provider.calls("primary"))
}

// An allow that comes after the reply was stopped runs nothing either.
func (s *ConnectorApprovalSuite) TestAnAllowAfterTheReplyStoppedRunsNothing() {
	d, connection := s.holding("whoami")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.stop("call-1")
	s.True(d.answerApproval("call-1", "command-1", true, ""))
	got := s.answerOf(answer)

	s.Contains(got.said, `"status":"stopped"`)
	s.Zero(s.provider.calls("primary"))
	s.Equal("declined: the reply ended before the person answered", s.rows(connection)[0].ErrorMessage)
	s.Empty(s.settledOf("call-1").ID, "the ended reply settles its own step")
}

// A tool the binding grants and require_approval does not name is sent at once.
func (s *ConnectorApprovalSuite) TestAToolTheListDoesNotNameIsSentAtOnce() {
	d, _ := s.holding("echo")

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Nil(s.questionOfAny(), "nobody was asked")
}

func (s *ConnectorApprovalSuite) TestAPatternHoldsTheToolsItMatches() {
	d, _ := s.holding("who*")

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")

	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.True(d.answerApproval("call-1", "command-1", true, ""))
	s.Equal("primary", s.answerOf(answer).said)
}

// Approval written into the config while a session runs holds that session's next call, as
// the config is read again before every call (recheck).
func (s *ConnectorApprovalSuite) TestApprovalAddedToTheConfigHoldsTheNextCall() {
	connection := s.connection("", "primary")
	binding := s.fixed("crm", connection, "whoami")
	config := s.config(binding)
	d := s.asking(config, "alice")
	binding.RequireApproval = []string{"whoami"}
	s.rebind(config, binding)

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")

	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.Zero(s.provider.calls("primary"))
	s.True(d.answerApproval("call-1", "command-1", false, ""))
	s.Contains(s.answerOf(answer).said, `"status":"declined"`)
}

// Approval taken off the config does not free the calls of a session opened with it.
func (s *ConnectorApprovalSuite) TestApprovalTakenOffTheConfigStillHoldsTheSessionsCalls() {
	connection := s.connection("", "primary")
	binding := s.fixed("crm", connection, "whoami")
	binding.RequireApproval = []string{"whoami"}
	config := s.config(binding)
	d := s.asking(config, "alice")
	binding.RequireApproval = nil
	s.rebind(config, binding)

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")

	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.True(d.answerApproval("call-1", "command-1", false, ""))
	s.Contains(s.answerOf(answer).said, `"status":"declined"`)
	s.Zero(s.provider.calls("primary"))
}

// Only the session's own user, or its customer's backend, answers its held call.
func (s *ConnectorApprovalSuite) TestOnlyTheSessionsUserOrItsBackendMayAnswer() {
	d, _ := s.holding("whoami")
	session := &Session{spec: d.spec, connectors: d}

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)

	s.False(session.AnswerConnectorApproval("call-1", "command-1", true, "",
		Owner{CustomerID: s.customerID, UserID: "bob", Kind: auth.KindAuthenticated}), "another user")
	s.False(session.AnswerConnectorApproval("call-1", "command-1", true, "",
		Owner{CustomerID: s.customerID, Kind: auth.KindAuthenticated}), "a user naming nobody")
	s.False((&Session{spec: d.spec}).AnswerConnectorApproval("call-1", "command-1", true, "",
		Owner{CustomerID: s.customerID, Kind: auth.KindServer}), "a session with no connectors")
	s.False(session.AnswerConnectorApproval("call-1", "command-1", true, "",
		Owner{CustomerID: "another-customer", UserID: "alice", Kind: auth.KindServer}), "another customer's backend")
	s.Never(func() bool { return len(answer) > 0 }, 200*time.Millisecond, 20*time.Millisecond, "still waiting")
	s.True(session.AnswerConnectorApproval("call-1", "command-1", true, "",
		Owner{CustomerID: s.customerID, Kind: auth.KindServer}), "the backend")
	s.Equal("primary", s.answerOf(answer).said)

	again := s.callInBackground(s.ctx, d, "call-2", "crm__whoami")
	s.Require().Eventually(func() bool { return s.questionOf("call-2") != nil }, 5*time.Second, 10*time.Millisecond)
	s.True(session.AnswerConnectorApproval("call-2", "command-1", true, "",
		Owner{CustomerID: s.customerID, UserID: "alice", Kind: auth.KindAuthenticated}), "the user")
	s.Equal("primary", s.answerOf(again).said)
}

// A binding without require_approval sends its call as on base (probe on 29091a59: the
// call answered "primary", reached the provider once, and its row has no error_type).
func (s *ConnectorApprovalSuite) TestABindingWithoutApprovalSendsTheCallAsBefore() {
	connection := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", connection, "whoami")), "alice", nil))
	s.Require().NoError(err)

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(1, s.provider.calls("primary"))
	s.Empty(s.rows(connection)[0].ErrorType)
}

// holding is a dispatcher for alice's session of a config binding the app's primary account
// as crm, granting whoami and echo, with require_approval approval, whose conversation asks
// alice on the call's step for command-1, and its connection.
func (s *ConnectorApprovalSuite) holding(approval ...string) (*dispatcher, string) {
	return s.holdingWith(func(*store.ConnectorBinding) {}, approval...)
}

// holdingWith is holding with the binding changed by change.
func (s *ConnectorApprovalSuite) holdingWith(change func(*store.ConnectorBinding), approval ...string) (*dispatcher, string) {
	connection := s.connection("", "primary")
	binding := s.fixed("crm", connection, "whoami", "echo")
	binding.RequireApproval = approval
	change(&binding)
	return s.asking(s.config(binding), "alice"), connection
}

// asking is a dispatcher for user's session of config whose conversation asks only alice, as
// holding's does.
func (s *ConnectorApprovalSuite) asking(config store.AgentConfig, user string) *dispatcher {
	d, _, _, err := s.attach(s.spec(config, user, nil))
	s.Require().NoError(err)
	d.approvals.await = func(owner, callID string, approval persistent.ToolApproval) (string, bool) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if owner != "alice" {
			return "", false
		}
		s.asked[callID] = approval
		return "command-1", true
	}
	d.approvals.waits = func(callID string) bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		_, asked := s.asked[callID]
		_, over := s.settled[callID]
		return asked && !over && !s.stopped[callID]
	}
	d.approvals.decided = func(event agent.ToolApprovalDecided) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.settled[event.ID] = event
	}
	return d
}

// callInBackground runs the call id of name on turn, and hands back what it said once it
// answers.
func (s *ConnectorApprovalSuite) callInBackground(turn context.Context, d *dispatcher, id, name string) chan answered {
	answer := make(chan answered, 1)
	go func() {
		parts, err := d.Run(turn, llm.ToolCall{ID: id, Name: name, Arguments: "{}"})
		answer <- answered{said: llm.TextOf(parts), err: err}
	}()
	return answer
}

// answerOf waits for a call run in the background to answer, past loginPoll.
func (s *ConnectorApprovalSuite) answerOf(answer chan answered) answered {
	select {
	case got := <-answer:
		return got
	case <-time.After(2*loginPoll + 5*time.Second):
		s.FailNow("the call never answered")
		return answered{}
	}
}

func (s *ConnectorApprovalSuite) questionOf(callID string) *persistent.ToolApproval {
	s.mu.Lock()
	defer s.mu.Unlock()
	found, ok := s.asked[callID]
	if !ok {
		return nil
	}
	return &found
}

func (s *ConnectorApprovalSuite) questionOfAny() map[string]persistent.ToolApproval {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.asked) == 0 {
		return nil
	}
	return s.asked
}

func (s *ConnectorApprovalSuite) settledOf(callID string) agent.ToolApprovalDecided {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.settled[callID]
}

func (s *ConnectorApprovalSuite) stop(callID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopped[callID] = true
}

// rows are the connection's invocation rows, once the first is written.
func (s *ConnectorApprovalSuite) rows(connection string) []store.ConnectorInvocation {
	var rows []store.ConnectorInvocation
	s.Require().Eventually(func() bool {
		var err error
		rows, err = s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		return err == nil && len(rows) > 0
	}, 5*time.Second, 20*time.Millisecond)
	return rows
}
