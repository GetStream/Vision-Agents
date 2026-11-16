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

	command, _, asked := c.AwaitApproval("employee", "call-1", `{"text":"secret plan"}`, allowEcho)

	s.Require().True(asked)
	s.Equal("command-a", command)
	parts := current(c).Parts
	s.Require().Len(parts, 1)
	s.Equal("awaiting_approval", parts[0].Status)
	s.Equal("crm__echo", parts[0].Name)
	s.Equal("employee", parts[0].TargetUserID)
	s.Equal(Approval{Title: "Allow crm to run echo?"}, *parts[0].Approval)
	s.JSONEq(`{"text":"secret plan"}`, string(parts[0].Arguments), "the person sees what the call would send")
	s.True(c.AwaitsApproval("call-1"))
	s.True(c.DecideApproval(agent.ToolApprovalDecided{ID: "call-1", Allowed: true}))
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

	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)
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

	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)

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

	_, _, asked := c.AwaitApproval("employee", "call-1", "", ToolApproval{Title: " Allow crm to run " + strings.Repeat("e", 100) + "? "})

	s.Require().True(asked)
	s.Equal("Allow crm to run "+strings.Repeat("e", 62)+"…", current(c).Parts[0].Approval.Title)
}

// A declined call never ran, whatever its result says.
func (s *DisplaySuite) TestADeclinedHeldCallsStepIsCancelled() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)
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
	_, err := c.BeginFollowUp("follow-up-a", "Remind me to tell Nash a joke", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	command, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)

	s.True(asked)
	s.Equal("follow-up-a", command)
}

// A follow-up answering a writer's message asks the writer, though the conversation has an
// owner, as one a caller opened on a channel of a type a config answers in has.
func (s *DisplaySuite) TestAWritersFollowUpAsksTheWriterNotTheOwner() {
	c := s.open("on_call")
	_, err := c.BeginFollowUp("follow-up-a", "check the crm for my shift", "anya")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	_, asked, ok := c.AwaitApproval("employee", "call-1", "", allowEcho)

	s.True(ok)
	s.Equal("anya", asked)
	s.Equal("anya", current(c).Parts[0].TargetUserID)
}

// Only the conversation's owner is asked, and only for a call of the reply being written.
func (s *DisplaySuite) TestAHeldCallIsAskedOnlyOfTheOwnerForACallOfTheReply() {
	c := s.open("on_call")
	s.False(c.AwaitsApproval("call-1"), "no reply is being written")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

	_, _, asked := c.AwaitApproval("someone-else", "call-1", "", allowEcho)
	s.False(asked, "not the owner")
	_, _, asked = c.AwaitApproval("employee", "call-2", "", allowEcho)
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
	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)
	s.Require().True(asked)

	c.Cancel()

	s.False(c.AwaitsApproval("call-1"))
	s.Equal("cancelled", current(c).Parts[0].Status)
}

// The arguments are shown as a client tool's are: valid JSON up to maxClientArguments. Any
// other arguments show as a cut preview in the approval's message, so none is approved blind.
func (s *DisplaySuite) TestAHeldCallShowsLargeOrInvalidArgumentsAsAPreview() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	large := `{"text":"` + strings.Repeat("e", maxClientArguments) + `"}`
	padded := strings.Repeat("\n", maxApprovalMessage+60) + `{"to":"x"`
	for id, arguments := range map[string]string{"call-1": `{"text":`, "call-2": large, "call-3": padded} {
		c.Observe(agent.ToolStarted{ID: id, Tool: "crm__echo", StartedAt: time.Now().UTC()})
		_, _, asked := c.AwaitApproval("employee", id, arguments, allowEcho)
		s.Require().True(asked)
	}

	for _, part := range current(c).Parts {
		s.Empty(part.Arguments, part.ID)
		s.Require().NotNil(part.Approval, part.ID)
		switch part.ID {
		case "call-1":
			s.Equal(`{"text":`, part.Approval.Message)
		case "call-2":
			s.Len([]rune(part.Approval.Message), maxApprovalMessage)
			s.True(strings.HasPrefix(part.Approval.Message, `{"text":"eee`))
			s.True(strings.HasSuffix(part.Approval.Message, "…"))
		case "call-3":
			s.Equal(`{"to":"x"`, part.Approval.Message)
		}
	}
}

// An answer settles the step only while it still waits: once the reply ended, it settles
// nothing and says so, so the dispatcher does not send the call.
func (s *DisplaySuite) TestAnAnswerAfterTheReplyEndedSettlesNothing() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)
	s.Require().True(asked)

	c.Cancel()

	s.False(c.DecideApproval(agent.ToolApprovalDecided{ID: "call-1", Allowed: true}))
	s.Equal("cancelled", current(c).Parts[0].Status)
}

// A held call that was declined or stopped was never sent, so its tools entry says it failed
// rather than finished.
func (s *DisplaySuite) TestAHeldCallThatWasNotSentIsNotFinishedInTools() {
	for _, status := range []string{"declined", "stopped"} {
		c := s.open("on_call")
		_, err := c.BeginCommand("command-a", "Tell Nash a joke on the crm", "")
		s.Require().NoError(err)
		c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})

		c.Observe(agent.ToolRan{ID: "call-1", Result: `{"status":"` + status + `","message":"not run"}`})

		tool := current(c).Tools[0]
		s.Equal("failed", tool.Status, status)
		s.Equal(status, tool.Summary)
	}
}

// A write waiting in the queue, and an update already handed to the conversation's listener,
// keep the approval they were made with while the reply being written has its approval
// answered, so neither is encoded while it changes.
func (s *DisplaySuite) TestQueuedAndPublishedCopiesKeepTheirOwnApproval() {
	c := s.open("on_call")
	var published []Updated
	c.Attach(func(update Updated) { published = append(published, update) })
	c.mu.Lock()
	defer c.mu.Unlock()
	reply := Message{ID: "reply", Role: "assistant", Parts: []Part{{
		Type: partToolCall, ID: "call", Status: "awaiting_approval", Approval: &Approval{Title: "Send it?"},
	}}}
	c.enqueue(reply, false)
	c.publish(reply)
	queued := c.data.Pending[len(c.data.Pending)-1]

	s.True(decided(&reply, "call", true, "", time.Now()))

	s.Equal("allowed", reply.Parts[0].Approval.Decision)
	s.Empty(queued.Message.Parts[0].Approval.Decision, "the queued write")
	s.Require().Len(published, 1)
	s.Empty(published[0].Message.Parts[0].Approval.Decision, "the published update")
}

// The copy a live update sends is encoded after the lock is let go: answering the approval
// meanwhile changes only the live reply.
func (s *DisplaySuite) TestTheLiveUpdatesCopyKeepsItsOwnApproval() {
	reply := Message{ID: "reply", Role: "assistant", Parts: []Part{{
		Type: partToolCall, ID: "call", Status: "awaiting_approval", Approval: &Approval{Title: "Send it?"},
	}}}
	sent := liveCopy(&reply)

	s.True(decided(&reply, "call", true, "", time.Now()))

	s.Equal("allowed", reply.Parts[0].Approval.Decision)
	s.Empty(sent.Parts[0].Approval.Decision, "the live update")
}

// expiredLine is what a reply says once its session closed with its turn while a call waited
// for a person to allow it (AI-3302).
const expiredLine = "The approval expired before anyone answered, so nothing ran. Ask again if you still need it."

// heldReply is a reply to Anya's command that said text and then made crm's echo wait for her.
func (s *DisplaySuite) heldReply(text string) (*Conversation, CommandReceipt) {
	c := s.open("on_call")
	receipt, err := c.BeginCommand("command-a", "check the crm for my shift", "")
	s.Require().NoError(err)
	c.Observe(agent.Responding{TurnID: "turn-a"})
	if text != "" {
		c.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: text})
	}
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, _, asked := c.AwaitApproval("employee", "call-1", "", allowEcho)
	s.Require().True(asked)
	return c, receipt
}

// The session of a reply waiting for Anya's approval closes with its turn: the step is
// declined as one nobody allowed in time, and the reply, once it ends, says so in Chat.
func (s *DisplaySuite) TestAnExpiredApprovalIsSaidInTheReply() {
	c, receipt := s.heldReply("Let me check.")

	c.ExpireApprovals()
	c.Release()
	saved(s.T(), c)

	s.False(c.AwaitsApproval("call-1"))
	part := current(c).Parts[0]
	s.Equal("cancelled", part.Status)
	s.Equal("declined", part.Approval.Decision)
	s.Equal("Not allowed in time", part.Summary)
	stored := s.stored(receipt.AssistantMessageID)
	s.Equal("Let me check.\n\n"+expiredLine, stored.Text)
	s.Equal("cancelled", stored.State)
}

// A reply that said nothing before the call, as on staging, is the line alone; one whose next
// round had begun keeps the break that round wrote, and gets no second one.
func (s *DisplaySuite) TestAnExpiredApprovalsLineFollowsWhatTheReplySaid() {
	silent, _ := s.heldReply("")
	silent.ExpireApprovals()
	s.Equal(expiredLine, current(silent).Text)

	begun, _ := s.heldReply("Let me check.")
	begun.Observe(agent.Responding{TurnID: "turn-a"})
	s.Require().Equal("Let me check.\n\n", current(begun).Text)
	begun.ExpireApprovals()
	s.Equal("Let me check.\n\n"+expiredLine, current(begun).Text)
}

// Two calls waiting at once are both declined, and the reply says it once.
func (s *DisplaySuite) TestEveryWaitingCallExpiresAndTheReplySaysItOnce() {
	c, _ := s.heldReply("Let me check.")
	c.Observe(agent.ToolStarted{ID: "call-2", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	_, _, asked := c.AwaitApproval("employee", "call-2", "", allowEcho)
	s.Require().True(asked)

	c.ExpireApprovals()
	c.ExpireApprovals()

	s.False(c.AwaitsApproval("call-1"))
	s.False(c.AwaitsApproval("call-2"))
	s.Equal("Let me check.\n\n"+expiredLine, current(c).Text)
}

// A call Anya allowed, or the backend declined, waits no more: the reply ending says nothing of
// an approval, and the steps keep their answers.
func (s *DisplaySuite) TestAnAnsweredApprovalDoesNotExpire() {
	allowed, _ := s.heldReply("Let me check.")
	s.Require().True(allowed.DecideApproval(agent.ToolApprovalDecided{ID: "call-1", Allowed: true}))
	allowed.ExpireApprovals()
	s.Equal("Let me check.", current(allowed).Text)
	s.Equal("running", current(allowed).Parts[0].Status)

	declined, _ := s.heldReply("Let me check.")
	s.Require().True(declined.DecideApproval(agent.ToolApprovalDecided{ID: "call-1", Summary: "Not now"}))
	declined.ExpireApprovals()
	s.Equal("Let me check.", current(declined).Text)
	s.Equal("Not now", current(declined).Parts[0].Summary)
}

// A reply whose call waits for no approval, or no reply at all, is left as it is.
func (s *DisplaySuite) TestNothingExpiresWithoutAWaitingCall() {
	c := s.open("on_call")
	c.ShowTools([]string{"crm__echo"})
	c.ExpireApprovals()

	_, err := c.BeginCommand("command-a", "check the crm for my shift", "")
	s.Require().NoError(err)
	c.Observe(agent.Responding{TurnID: "turn-a"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "Let me check."})
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "crm__echo", StartedAt: time.Now().UTC()})
	c.ExpireApprovals()

	s.Equal("Let me check.", current(c).Text)
	s.Require().Len(current(c).Parts, 1)
	s.Equal("running", current(c).Parts[0].Status)
}

// A router shutting down writes nothing more to a conversation (finish): no line either.
func (s *DisplaySuite) TestNoApprovalExpiresOnceTheServiceStops() {
	c, _ := s.heldReply("Let me check.")
	s.service.Close()

	c.ExpireApprovals()

	s.Equal("Let me check.", current(c).Text)
	s.True(c.AwaitsApproval("call-1"))
}
