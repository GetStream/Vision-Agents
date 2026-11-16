//go:build integration

package session

import (
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func (s *SessionSuite) TestABalanceTheRouterCouldNotReadIsTheRoutersFault() {
	// Catches a billing read outage blamed on the customer's balance: the refusal a real
	// enforcer gives when it cannot read the balance wraps ErrNoCredits, and counted as such it
	// would end the session as create_no_credit or model_no_credit.
	unanswered, err := usage.NewClient(usage.ClientOptions{URL: "http://127.0.0.1:1", Token: "none"})
	s.Require().NoError(err)
	money := policytest.Start(s.T(), "session_unread", func(settings *policy.Settings) {
		settings.Usage = unanswered
		settings.MaxStale = time.Millisecond
	})
	stamp := uuid.NewString()
	owner := routing.Owner{CustomerID: "app-" + stamp, Attribution: routing.Attribution{OrganizationID: "org-" + stamp}}
	s.Require().NoError(money.Store.JoinOrganization(s.ctx, owner.CustomerID, owner.Attribution.OrganizationID))
	money.Enforcer.Join(owner.CustomerID, owner.Attribution.OrganizationID, usage.DefaultTier)
	var unread error
	s.eventually(func() bool {
		admission, err := money.Enforcer.Admit(s.ctx, owner)
		unread = admission.Spend
		return err == nil && unread != nil
	}, "the enforcer never refused the balance it could not read")
	s.Require().ErrorIs(unread, policy.ErrNoCredits, "the refusal reads as no credit")
	s.Require().True(policy.Transient(unread), "and passes once the balance can be read")

	s.speech = func() (sts.STS, error) { return &openSpeech{emitter: sts.NewEmitter(sts.EmitterBuffer)}, nil }
	s.speechGate = refusingGate{err: unread}
	s.manages()
	before := s.allEnds()
	_, err = s.manager.Create(s.ctx, Spec{CallID: "call-1", CustomerID: "acme", STSTarget: "en-low-latency"})
	s.Require().ErrorIs(err, policy.ErrNoCredits)
	s.endedOnce(before, EndCreateFailed)

	session := &Session{}
	lost := ""
	session.observeTurn(agent.Error{Context: "llm", Err: unread}, &lost)
	session.agentLeft.Store(true)
	s.Equal(EndAgentFailed, session.endLabel(), "a turn refused for the unread balance")
	s.Equal(FaultRouter, session.fault())
}
