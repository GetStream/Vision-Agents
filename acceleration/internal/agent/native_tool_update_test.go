package agent

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

func (s *AgentSuite) TestANativeModelTellsTheCallerHowAnOpenCallIsGoing() {
	peer, runner := s.nativeRunningAHeldTool()

	s.agent.ToolUpdate("call1", "searching 3 airlines", false)

	update := s.nativeFrame(peer, "response.create")
	s.Contains(string(update["response"]), "searching 3 airlines")
	s.Contains(string(update["response"]), `"tool_choice":"none"`, "a reply about an open call must not call it again")
	s.never(func() bool { return countOf[ToolRan](s.reported()) > 0 }, "the update ended the call")

	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.agent.ToolUpdate("call1", "found two seats", false)
	s.Contains(string(s.nativeFrame(peer, "response.create")["response"]), "found two seats",
		"the line stayed taken after the update was said")
	runner.finish()
}

func (s *AgentSuite) TestANativeModelIsAskedForTheWordsAToolWantsSaid() {
	peer, runner := s.nativeRunningAHeldTool()

	s.agent.ToolUpdate("call1", "Got a seat, moving your booking now.", true)

	update := s.nativeFrame(peer, "response.create")
	s.Contains(string(update["response"]), "in these words or close to them")
	s.Contains(string(update["response"]), "Got a seat, moving your booking now.")
	runner.finish()
}

// nativeRunningAHeldTool is a native call on which the model asked for lookup_order, which
// runs until released, and whose reply asking for it has ended.
func (s *AgentSuite) nativeRunningAHeldTool() (*nativePeer, *heldTool) {
	s.ownsTools("")
	runner := &heldTool{holdingTool: holdingTool{began: make(chan struct{}), release: make(chan struct{})}}
	runner.finish = sync.OnceFunc(func() { close(runner.release) })
	peer := s.joinNativeWith(func(options *Options) {
		options.Tools = s.tools
		options.ToolRunner = runner
		options.Skills = harness.Skills{}
	})
	s.T().Cleanup(runner.finish)
	s.nativeFrame(peer, "session.update")
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"lookup_order","arguments":"{\"order\":\"12\"}"}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	select {
	case <-runner.began:
	case <-time.After(settleFor):
		s.FailNow("the tool never started")
	}
	return peer, runner
}
