package session

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

type ToolCallStateSuite struct {
	suite.Suite
	calls *toolCalls
	began time.Time
}

func TestToolCallState(t *testing.T) {
	suite.Run(t, new(ToolCallStateSuite))
}

func (s *ToolCallStateSuite) SetupTest() {
	s.calls = &toolCalls{approves: func(tool string) bool { return tool == "book_flight" }}
	s.began = time.Now().UTC().Add(-time.Second)
}

// apply feeds the tracker event and returns the change it made.
func (s *ToolCallStateSuite) apply(event agent.Event) (ToolCallChanged, bool) {
	return s.calls.observe(event)
}

func (s *ToolCallStateSuite) start(id, tool string) ToolCallChanged {
	changed, ok := s.apply(agent.ToolStarted{ID: id, TurnID: "turn-1", Tool: tool, StartedAt: s.began})
	s.Require().True(ok)
	return changed
}

func (s *ToolCallStateSuite) TestACallThatNeedsApprovalWaitsThenRunsOnceAllowedAndSucceeds() {
	started := s.start("call-1", "book_flight")
	s.Equal(ToolCallStarted, started.Phase)
	s.Equal(ToolCallWaiting, started.State.Status)

	allowed, ok := s.apply(agent.ToolApprovalDecided{ID: "call-1", TurnID: "turn-1", Allowed: true})
	s.Require().True(ok)
	s.Equal(ToolCallProgress, allowed.Phase)
	s.Equal(ToolCallInProgress, allowed.State.Status)
	s.False(allowed.State.ApprovedAt.IsZero())

	finished, ok := s.apply(agent.ToolRan{ID: "call-1", TurnID: "turn-1", Tool: "book_flight", Result: "booked"})
	s.Require().True(ok)
	s.Equal(ToolCallFinished, finished.Phase)
	s.Equal(ToolCallSuccess, finished.State.Status)
	s.Equal(allowed.State.ApprovedAt, finished.State.ApprovedAt)
	s.GreaterOrEqual(finished.State.Duration, time.Second)
}

func (s *ToolCallStateSuite) TestADeclinedCallFinishesCancelledWithWhenItWasRejected() {
	s.start("call-1", "book_flight")

	declined, ok := s.apply(agent.ToolApprovalDecided{ID: "call-1", TurnID: "turn-1", Summary: "Not booked"})

	s.Require().True(ok)
	s.Equal(ToolCallFinished, declined.Phase)
	s.Equal(ToolCallCancelled, declined.State.Status)
	s.False(declined.State.RejectedAt.IsZero())
	s.True(declined.State.ApprovedAt.IsZero())
	s.Equal("Not booked", declined.State.Error)
	_, late := s.apply(agent.ToolRan{ID: "call-1", Tool: "book_flight", Err: errors.New("declined")})
	s.False(late, "a call is finished once")
}

func (s *ToolCallStateSuite) TestAnUpdateWhileTheCallRunsIsItsProgress() {
	s.start("call-1", "search_flights")

	progress, ok := s.apply(agent.ToolUpdated{ID: "call-1", Tool: "search_flights", Text: "Searching for flights"})

	s.Require().True(ok)
	s.Equal(ToolCallProgress, progress.Phase)
	s.Equal("Searching for flights", progress.State.Progress)
	s.Equal(ToolCallInProgress, progress.State.Status)
	_, again := s.apply(agent.ToolUpdated{ID: "call-1", Tool: "search_flights", Text: "Searching for flights"})
	s.False(again, "the same update changes nothing")
}

func (s *ToolCallStateSuite) TestACallStoppedByItsContextFinishesCancelled() {
	s.start("call-1", "search_flights")

	finished, ok := s.apply(agent.ToolRan{ID: "call-1", Tool: "search_flights",
		Err: fmt.Errorf("session: search_flights stopped: %w", context.Canceled)})

	s.Require().True(ok)
	s.Equal(ToolCallCancelled, finished.State.Status)
	s.True(finished.State.RejectedAt.IsZero(), "nobody declined it")
}

func (s *ToolCallStateSuite) TestACallThatErrsFinishesFailedSayingWhy() {
	s.start("call-1", "search_flights")

	finished, ok := s.apply(agent.ToolRan{ID: "call-1", Tool: "search_flights", Err: errors.New("the airline is down")})

	s.Require().True(ok)
	s.Equal(ToolCallFailed, finished.State.Status)
	s.Equal("the airline is down", finished.State.Error)
}

func (s *ToolCallStateSuite) TestAVoiceCallWaitsOnItsQuestionAndFinishesWithTheConfirmationThatRanIt() {
	s.start("call-1", "transfer_money")

	waiting, ok := s.apply(agent.ToolAwaitingApproval{ID: "call-1", Tool: "transfer_money", Question: "Send 20 euros to Ana?"})
	s.Require().True(ok)
	s.Equal(ToolCallWaiting, waiting.State.Status)
	s.Equal("Send 20 euros to Ana?", waiting.State.Progress)
	_, asked := s.apply(agent.ToolRan{ID: "call-1", Tool: "transfer_money", Result: "Ask the caller", Asked: true})
	s.False(asked, "the question is not the call's outcome")
	_, confirmation := s.apply(agent.ToolStarted{ID: "call-2", Tool: agent.ConfirmTool, StartedAt: s.began})
	s.False(confirmation, "the confirmation is not a call of its own")
	_, ok = s.apply(agent.ToolApprovalDecided{ID: "call-1", Allowed: true})
	s.Require().True(ok)

	finished, ok := s.apply(agent.ToolRan{ID: "call-2", Tool: agent.ConfirmTool,
		Arguments: `{"tool_call_id":"call-1","allowed":true}`, Result: "sent"})

	s.Require().True(ok)
	s.Equal("call-1", finished.State.ID)
	s.Equal(ToolCallSuccess, finished.State.Status)
	s.False(finished.State.ApprovedAt.IsZero())
}

func (s *ToolCallStateSuite) TestAVoiceQuestionNobodyAnsweredInTimeIsRejected() {
	s.start("call-1", "transfer_money")
	s.apply(agent.ToolAwaitingApproval{ID: "call-1", Tool: "transfer_money", Question: "Send it?"})

	lapsed, ok := s.apply(agent.ToolApprovalDecided{ID: "call-1", Summary: "Not confirmed in time"})

	s.Require().True(ok)
	s.Equal(ToolCallCancelled, lapsed.State.Status)
	s.False(lapsed.State.RejectedAt.IsZero())
}

func (s *ToolCallStateSuite) TestAConnectorCallHeldBeforeItsStartCameThroughStartsWaiting() {
	_, early := s.apply(agent.ToolAwaitingApproval{ID: "call-1", Tool: "github__merge", Question: "Allow github to run merge?"})
	s.False(early)

	started := s.start("call-1", "github__merge")

	s.Equal(ToolCallWaiting, started.State.Status)
	s.Equal("Allow github to run merge?", started.State.Progress)
}
