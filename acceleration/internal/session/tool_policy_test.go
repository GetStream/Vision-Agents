package session

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ToolPolicySuite is what an interruption does to a caller's or a worker's tool in flight.
type ToolPolicySuite struct {
	suite.Suite
}

func TestToolPolicySuite(t *testing.T) {
	suite.Run(t, new(ToolPolicySuite))
}

func (s *ToolPolicySuite) TestAnInterruptibleToolIsCancelledByAnInterruption() {
	policy := interruptible([]harness.Tool{{Name: "search_flights", Interruptible: true}}, nil)

	s.True(policy("search_flights").Cancels)
}

func (s *ToolPolicySuite) TestAToolThatDoesNotSayOutlivesAnInterruption() {
	policy := interruptible([]harness.Tool{{Name: "search_flights", Interruptible: true}, {Name: "book_flight"}}, nil)

	s.False(policy("book_flight").Cancels)
}

func (s *ToolPolicySuite) TestABindingsPolicyIsKeptBesideIt() {
	binding := func(name string) agent.ToolPolicy {
		return agent.ToolPolicy{PreSpeech: "One moment", Waits: name == "crm__lookup"}
	}

	policy := interruptible([]harness.Tool{{Name: "search_flights", Interruptible: true}}, binding)

	s.Equal(agent.ToolPolicy{PreSpeech: "One moment", Cancels: true}, policy("search_flights"))
	s.Equal(agent.ToolPolicy{PreSpeech: "One moment", Waits: true}, policy("crm__lookup"))
}

func (s *ToolPolicySuite) TestASessionWithNoInterruptibleToolKeepsTheBindingsPolicyAsIs() {
	s.Nil(interruptible([]harness.Tool{{Name: "book_flight"}}, nil))
}

// offering is a worker hosting tools of its own choosing.
type offering struct{ tools []dispatch.Tool }

func (o offering) HostedTools(string, string) ([]dispatch.Tool, time.Duration) {
	return o.tools, time.Minute
}

func (o offering) RunHosted(context.Context, string, string, dispatch.ToolCall) ([]llm.ContentPart, error) {
	return llm.TextParts("done"), nil
}

func (s *ToolPolicySuite) TestAWorkersToolKeepsItsPolicies() {
	m := &Manager{logger: slog.Default(), hosts: offering{tools: []dispatch.Tool{
		{Name: "investigate_sdk", Description: "Read SDK source", Interruptible: true, Timeout: time.Second},
	}}}

	tools, _ := m.hostedTools(Spec{CustomerID: "acme", AgentID: "support"}, "session-1", nil, nil, nil)

	s.Require().Len(tools, 1)
	s.True(tools[0].Interruptible)
	s.Equal(time.Second, tools[0].Timeout)
}

func (s *ToolPolicySuite) TestAWorkersToolAsksItsApprovalQuestion() {
	m := &Manager{logger: slog.Default(), hosts: offering{tools: []dispatch.Tool{
		{Name: "book_flight", Description: "Book a flight", Approval: "Book flight {flight_id}?"},
		{Name: "search_flights", Description: "Search flights"},
	}}}

	tools, _ := m.hostedTools(Spec{CustomerID: "acme", AgentID: "support"}, "session-1", nil, nil, nil)

	s.Require().Len(tools, 2)
	s.Equal(&harness.ToolApproval{Title: "Book flight {flight_id}?"}, tools[0].Approval)
	s.Nil(tools[1].Approval)
}
