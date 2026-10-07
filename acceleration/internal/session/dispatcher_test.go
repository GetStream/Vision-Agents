//go:build integration

package session

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DispatcherSuite is what a session's connector tool call does once the session is open: the
// check against the config and the connection as they are now, before anything is sent, and
// the envelope around the call.
type DispatcherSuite struct {
	connectorFixture
}

func TestDispatcherSuite(t *testing.T) {
	suite.Run(t, new(DispatcherSuite))
}

// TestTwoAccountsOfOneUserNeverCross: the prototype's
// TestSessionSelectsBetweenTwoAccountsAndDisconnectBlocksAnOpenRuntime, first half.
func (s *DispatcherSuite) TestTwoAccountsOfOneUserNeverCross() {
	primary, secondary := s.connection("alice", "primary"), s.connection("alice", "secondary")
	config := s.config(s.chosen("primary", "whoami"), s.chosen("secondary", "whoami"))
	d, tools, _, err := s.attach(s.spec(config, "alice", map[string]string{"primary": primary, "secondary": secondary}))
	s.Require().NoError(err)

	said, err := s.call(d, "primary__whoami", "{}")

	s.Require().NoError(err)
	s.ElementsMatch([]string{"primary__whoami", "secondary__whoami"}, names(tools))
	s.Equal("primary", said)
	s.Equal(1, s.provider.calls("primary"))
	s.Zero(s.provider.calls("secondary"))
}

// TestAGrantRemovedFromAnOpenSessionIsNotCalled: the prototype's case, and a grant given
// back works again.
func (s *DispatcherSuite) TestAGrantRemovedFromAnOpenSessionIsNotCalled() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	d, _, _, err := s.attach(s.spec(config, "", nil))
	s.Require().NoError(err)

	s.rebind(config, s.fixed("crm", app))
	_, refused := s.call(d, "crm__whoami", "{}")
	s.rebind(config, s.fixed("crm", app, "whoami"))
	said, err := s.call(d, "crm__whoami", "{}")

	s.ErrorContains(refused, `connector "crm": the tool is no longer granted, so crm__whoami was not called`)
	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(1, s.provider.calls("primary"), "the refused call never reached the provider")
}

func (s *DispatcherSuite) TestABindingRemovedFromAnOpenSessionIsNotCalled() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	d, _, _, err := s.attach(s.spec(config, "", nil))
	s.Require().NoError(err)

	s.rebind(config)
	_, err = s.call(d, "crm__whoami", "{}")

	s.ErrorContains(err, "the agent config no longer binds it")
	s.Zero(s.provider.calls("primary"))
}

// TestAConnectionThatNowReachesElsewhereIsNotCalled: the prototype's changed endpoint. The
// connection's input now builds another endpoint, which must not receive the call or the
// credential.
func (s *DispatcherSuite) TestAConnectionThatNowReachesElsewhereIsNotCalled() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)

	_, err = s.store.DB().NewUpdate().Model((*store.ConnectorConnection)(nil)).
		Set("inputs = ?", `{"account":"moved"}`).Where("id = ?", app).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.call(d, "crm__whoami", "{}")

	s.ErrorContains(err, "the connection changed where it reaches since the session opened")
	s.Zero(s.provider.calls("primary"))
	s.Empty(s.provider.sent("moved"))
}

// TestADisconnectedAccountIsNotCalledAndTheOtherIsNotUsedInstead: the prototype's second
// half. A deleted connection is refused, and the session does not fall back to the
// user's other account.
func (s *DispatcherSuite) TestADisconnectedAccountIsNotCalledAndTheOtherIsNotUsedInstead() {
	primary, secondary := s.connection("alice", "primary"), s.connection("alice", "secondary")
	config := s.config(s.chosen("primary", "whoami"), s.chosen("secondary", "whoami"))
	d, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"primary": primary, "secondary": secondary}))
	s.Require().NoError(err)

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, primary))
	_, refused := s.call(d, "primary__whoami", "{}")
	said, err := s.call(d, "secondary__whoami", "{}")

	s.ErrorContains(refused, "the connection is gone")
	s.Zero(s.provider.calls("primary"))
	s.Require().NoError(err)
	s.Equal("secondary", said)
}

func (s *DispatcherSuite) TestAConnectionThatNeedsReauthorizationIsNotCalled() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)

	s.setState(app, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	_, err = s.call(d, "crm__whoami", "{}")

	s.ErrorContains(err, "the connection is needs_reauthorization")
	s.Zero(s.provider.calls("primary"))
}

// TestAnUngrantedToolUnderABoundAliasIsNotCallable: the model names a tool the provider has
// and the grant does not. It is not handed to the rest of the session either.
func (s *DispatcherSuite) TestAnUngrantedToolUnderABoundAliasIsNotCallable() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	d.next = &answering{}

	_, err = s.call(d, "crm__secret", "{}")

	s.ErrorContains(err, "crm__secret is not a tool this session can run")
	s.Zero(s.provider.calls("primary"))
	s.Empty(d.next.(*answering).asked)
}

func (s *DispatcherSuite) TestANameItDidNotOpenGoesToTheRestOfTheSession() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	d.next = &answering{}

	said, err := s.call(d, "lookup_order", "{}")

	s.Require().NoError(err)
	s.Equal("the caller's answer", said)
	s.Equal([]string{"lookup_order"}, d.next.(*answering).asked)
}

// TestACallTheTimeoutCutsOffIsAnUnknownOutcome: the provider may have done the work, so the
// model is not told the call failed.
func (s *DispatcherSuite) TestACallTheTimeoutCutsOffIsAnUnknownOutcome() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 200
	d, _, _, err := s.attach(s.spec(s.config(binding), "", nil))
	s.Require().NoError(err)

	said, err := s.call(d, "crm__slow", "{}")

	s.Require().NoError(err, "not an error the model would retry")
	s.Contains(said, "outcome_unknown: crm__slow did not answer within 200ms")
	s.Equal(1, s.provider.calls("primary"))
}

// TestAnInterruptedCallIsCancelledAtTheProvider: the turn's context ends, and the provider
// is sent notifications/cancelled for the call.
func (s *DispatcherSuite) TestAnInterruptedCallIsCancelledAtTheProvider() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 30000
	d, _, _, err := s.attach(s.spec(s.config(binding), "", nil))
	s.Require().NoError(err)
	turn, interrupt := context.WithCancel(s.ctx)
	time.AfterFunc(200*time.Millisecond, interrupt)

	_, err = d.Run(turn, llm.ToolCall{ID: uuid.NewString(), Name: "crm__slow", Arguments: "{}"})

	s.ErrorIs(err, context.Canceled)
	s.Eventually(func() bool { return slices.Contains(s.provider.sent("primary"), "notifications/cancelled") },
		5*time.Second, 20*time.Millisecond)
}

// answering is the rest of a session's tool chain, which answers whatever it is asked.
type answering struct {
	asked []string
}

func (a *answering) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	a.asked = append(a.asked, call.Name)
	return llm.TextParts("the caller's answer"), nil
}
