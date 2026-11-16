package agent

import (
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (s *AgentSuite) TestAHeldCallOutlivesAnInterruptionThatWouldHaveCancelledIt() {
	runner, participant := s.runningACancellableTool()

	s.True(s.agent.HoldTool("call-1"))
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.replyWentOut()
	s.mutters(participant, "hello?")

	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.HasPrefix(m.Content, stillRunning) })
	}, "the held call was not answered as still running")
	select {
	case <-runner.stopped:
		s.Fail("talking over the agent cancelled a call that was held")
	default:
	}
}

func (s *AgentSuite) TestACallAnInterruptionAlreadyCancelledCannotBeHeld() {
	runner, participant := s.runningACancellableTool()
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.replyWentOut()
	s.mutters(participant, "hello?")
	select {
	case <-runner.stopped:
	case <-time.After(settleFor):
		s.FailNow("the interruption did not cancel the call")
	}

	s.False(s.agent.HoldTool("call-1"), "the interruption came first")
}

func (s *AgentSuite) TestACallNobodyIsRunningCannotBeHeld() {
	s.join(true)

	s.False(s.agent.HoldTool("call-nobody-made"))
}

// runningACancellableTool is a call to a tool an interruption cancels, started and still
// running, and the caller who asked for it.
func (s *AgentSuite) runningACancellableTool() (*cancellationTool, stt.Participant) {
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Cancels: true} }
	s.join(true)
	runner := &cancellationTool{began: make(chan struct{}), stopped: make(chan struct{})}
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(settleFor):
		s.FailNow("tool did not start")
	}
	return runner, participant
}
