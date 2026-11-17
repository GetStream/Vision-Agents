package agent

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func (s *AgentSuite) TestANativeModelIsToldToAskBeforeACallThatNeedsApprovalRuns() {
	peer := s.nativeAsked(time.Minute)

	asked := s.nativeFrame(peer, "conversation.item.create")
	s.Contains(string(asked["item"]), "Ask them now, in these words")
	s.Contains(string(asked["item"]), "Book order 12?")
	s.nativeFrame(peer, "response.create")
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "the call ran before the caller said yes")
}

func (s *AgentSuite) TestANativeModelsConfirmationRunsTheCallWithItsOwnID() {
	peer := s.nativeAsked(time.Minute)
	s.nativeFrame(peer, "conversation.item.create")
	s.nativeQuestionSaid(peer)
	var approved bool
	s.runner.run = func(ctx context.Context, _ llm.ToolCall) ([]llm.ContentPart, error) {
		approved = Approved(ctx)
		return llm.TextParts("order 12 booked"), nil
	}

	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r3"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r3","call_id":"call2","name":"confirm_tool_call","arguments":"{\"tool_call_id\":\"call1\",\"allowed\":true}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r3","status":"completed"}}`)

	result := s.nativeFrame(peer, "conversation.item.create")
	s.Contains(string(result["item"]), "order 12 booked")
	s.Contains(string(result["item"]), "call2", "the result answers the confirmation the model has open")
	s.Require().Len(s.runner.asked(), 1)
	s.Equal("call1", s.runner.asked()[0].ID)
	s.True(approved)
}

func (s *AgentSuite) TestANativeCallerWhoSaysNothingDeclinesOnceTheQuestionWasSaid() {
	peer := s.nativeAsked(200 * time.Millisecond)
	s.nativeFrame(peer, "conversation.item.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	time.Sleep(500 * time.Millisecond)
	s.Zero(countOf[ToolApprovalDecided](s.reported()), "the wait started while the question was still being said")

	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)

	s.eventually(func() bool { return countOf[ToolApprovalDecided](s.reported()) == 1 }, "silence never declined the call")
	decided, _ := firstOf[ToolApprovalDecided](s.reported())
	s.Equal(ToolApprovalDecided{ID: "call1", TurnID: decided.TurnID, Summary: "Not confirmed in time"}, decided)
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "a call nobody allowed ran")
}

// nativeAsked is a native call on which the model asked for lookup_order, which needs the
// caller's yes and waits wait for it.
func (s *AgentSuite) nativeAsked(wait time.Duration) *nativePeer {
	s.ownsTools("order 12 booked")
	s.approves()
	peer := s.joinNativeWith(func(options *Options) {
		options.Tools = s.tools
		options.ToolRunner = s.runner
		options.Skills = harness.Skills{}
		options.ApprovalWait = wait
	})
	setup := s.nativeFrame(peer, "session.update")
	s.Contains(string(setup["session"]), confirmTool)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"lookup_order","arguments":"{\"order\":\"12\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	return peer
}

// nativeQuestionSaid is the model saying the question it was told to ask.
func (s *AgentSuite) nativeQuestionSaid(peer *nativePeer) {
	s.nativeFrame(peer, "response.create")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
}
