package conversation

import (
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// allowEcho is the question a session asks before it sends a held connector call of crm's echo.
var allowEcho = ToolApproval{Title: "Allow crm to run echo?"}

// The step of a held connector call is shown whatever visible_tools says, addressed to the
// person who asked, and answers with the command its reply answers. Allowed, it runs.
func (s *DisplaySuite) TestAHeldCallAsksTheOwnerOnItsStepAndRunsOnceAllowed() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	command, asked := c.AwaitApproval("employee", "call-1", allowEcho)

	s.Require().True(asked)
	s.Equal("command-a", command)
	parts := current(c).Parts
	s.Require().Len(parts, 1)
	s.Equal("awaiting_approval", parts[0].Status)
	s.Equal("crm__echo", parts[0].Name)
	s.Equal("employee", parts[0].TargetUserID)
	s.Equal(Approval{Title: "Allow crm to run echo?"}, *parts[0].Approval)
	s.True(c.AwaitsApproval("call-1"))
	c.Observe(agent.ToolApprovalDecided{ID: "call-1", Allowed: true})
	s.False(c.AwaitsApproval("call-1"), "allowed, it runs")
	s.Equal("running", current(c).Parts[0].Status)
	c.Observe(agent.ToolRan{ID: "call-1", Result: "said"})
	s.Equal("completed", current(c).Parts[0].Status)
}

// The question is written to Chat at once, so a client that opens the conversation while the
// call waits can answer it.
func (s *DisplaySuite) TestAHeldCallsQuestionIsWrittenToChat() {
	c := s.open("on_call")
	receipt, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	_, asked := c.AwaitApproval("employee", "call-1", allowEcho)
	s.Require().True(asked)
	saved(s.T(), c)

	s.Contains(s.raw(receipt.AssistantMessageID), "awaiting_approval")
}

// A call the config shows already has its step, which then waits rather than a second one.
func (s *DisplaySuite) TestAShownCallWaitsOnTheStepItHas() {
	c := s.open("on_call")
	c.ShowTools([]string{"crm__echo"})
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	s.Require().Len(current(c).Parts, 1)

	_, asked := c.AwaitApproval("employee", "call-1", allowEcho)

	s.Require().True(asked)
	s.Require().Len(current(c).Parts, 1)
	s.Equal("awaiting_approval", current(c).Parts[0].Status)
}

// A tool's name can be long, and the question is cut as a declared tool's is (maxApprovalTitle).
func (s *DisplaySuite) TestAHeldCallsQuestionIsCutToTheTitlesLength() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	_, asked := c.AwaitApproval("employee", "call-1", ToolApproval{Title: " Allow crm to run " + strings.Repeat("e", 100) + "? "})

	s.Require().True(asked)
	s.Equal("Allow crm to run "+strings.Repeat("e", 62)+"…", current(c).Parts[0].Approval.Title)
}

// A declined call never ran, whatever its result says.
func (s *DisplaySuite) TestADeclinedHeldCallsStepIsCancelled() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, asked := c.AwaitApproval("employee", "call-1", allowEcho)
	s.Require().True(asked)

	c.Observe(agent.ToolApprovalDecided{ID: "call-1", Summary: "Not now"})
	c.Observe(agent.ToolRan{ID: "call-1", Result: `{"status":"declined"}`})

	part := current(c).Parts[0]
	s.Equal("cancelled", part.Status)
	s.Equal("declined", part.Approval.Decision)
	s.Equal("Not now", part.Summary)
}

// A scheduled task's reply is a follow-up of the owner's, so its held call is asked of them.
func (s *DisplaySuite) TestAFollowUpsHeldCallIsAskedOfTheOwner() {
	c := s.open("on_call")
	_, err := c.BeginFollowUp("follow-up-a", "Remind me to tell Nash a joke")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	command, asked := c.AwaitApproval("employee", "call-1", allowEcho)

	s.True(asked)
	s.Equal("follow-up-a", command)
}

// Only the conversation's owner is asked, and only for a call of the reply being written.
func (s *DisplaySuite) TestAHeldCallIsAskedOnlyOfTheOwnerForACallOfTheReply() {
	c := s.open("on_call")
	s.False(c.AwaitsApproval("call-1"), "no reply is being written")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	_, asked := c.AwaitApproval("someone-else", "call-1", allowEcho)
	s.False(asked, "not the owner")
	_, asked = c.AwaitApproval("employee", "call-2", allowEcho)
	s.False(asked, "a call the reply did not make")
	s.Empty(current(c).Parts)
	s.False(c.AwaitsApproval("call-1"))
}

// A reply that ends ends its steps, so nothing waits for the answer any more.
func (s *DisplaySuite) TestAHeldCallStopsWaitingWhenItsReplyEnds() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, asked := c.AwaitApproval("employee", "call-1", allowEcho)
	s.Require().True(asked)

	c.Cancel()

	s.False(c.AwaitsApproval("call-1"))
	s.Equal("cancelled", current(c).Parts[0].Status)
}
