package agent

import (
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (s *AgentSuite) TestAToolsStatusIsSaidInTheAgentsOwnWordsWhileItRuns() {
	s.runningAHeldTool()
	s.model.composes = []string{"Still checking three airlines for you."}

	s.agent.ToolUpdate("call-1", "searching 3 airlines", false)

	s.eventually(func() bool { return s.spokenText("Still checking three airlines for you.") }, "the status was never said")
	composed := s.model.composeRequests()
	s.Require().NotEmpty(composed)
	asked := composed[len(composed)-1].Input
	s.Equal(fmt.Sprintf(composeNote, fmt.Sprintf(statusPurpose, "searching 3 airlines")), asked[len(asked)-1].Content)
	updated, found := firstOf[ToolUpdated](s.reported())
	s.Require().True(found)
	s.Equal(ToolUpdated{ID: "call-1", TurnID: updated.TurnID, Tool: "lookup_order", Text: "searching 3 airlines"}, updated)
}

func (s *AgentSuite) TestWordsAToolAsksToSayAreSaidAsWritten() {
	s.runningAHeldTool()
	composedBefore := len(s.model.composeRequests())

	s.agent.ToolUpdate("call-1", "Got a seat, moving your booking now.", true)

	s.eventually(func() bool { return s.spokenText("Got a seat, moving your booking now.") }, "the words were never said")
	s.Len(s.model.composeRequests(), composedBefore, "words to say as written are not rewritten")
}

func (s *AgentSuite) TestANewerUpdateForTheSameCallReplacesTheOneWaiting() {
	s.runningAHeldTool()
	s.holdTheLine()

	s.agent.ToolUpdate("call-1", "First words.", true)
	s.agent.ToolUpdate("call-1", "Second words.", true)
	s.releaseTheLine()

	s.eventually(func() bool { return s.spokenText("Second words.") }, "the newer update was never said")
	s.False(s.spokenText("First words."), "an update that was replaced was still said")
}

func (s *AgentSuite) TestAnUpdateTheResultBeatIsNeverSaid() {
	runner := s.runningAHeldTool()
	s.holdTheLine()

	s.agent.ToolUpdate("call-1", "Almost there.", true)
	runner.finish()
	s.eventually(func() bool { return countOf[ToolRan](s.reported()) == 1 }, "the tool never finished")
	s.releaseTheLine()

	s.never(func() bool { return s.spokenText("Almost there.") }, "an update was said after the result it was about")
}

func (s *AgentSuite) TestAnUpdateWaitsForTheCallerToStopTalking() {
	s.runningAHeldTool()
	participant := stt.Participant{ID: "alice"}
	talking := make(chan struct{})
	var done sync.WaitGroup
	done.Go(func() {
		words := "actually"
		for {
			select {
			case <-talking:
				return
			case <-time.After(20 * time.Millisecond):
				words += " and"
				s.mutters(participant, words)
			}
		}
	})
	s.eventually(func() bool { return s.agent.converse.Listening() }, "the caller was never heard talking")

	s.agent.ToolUpdate("call-1", "Still on it.", true)

	s.never(func() bool { return s.spokenText("Still on it.") }, "an update was said over the caller")
	close(talking)
	done.Wait()
}

func (s *AgentSuite) TestAnUpdateForACallThatIsNotRunningIsDropped() {
	s.join(true)

	s.agent.ToolUpdate("call-nobody-made", "Still on it.", true)

	s.never(func() bool { return countOf[ToolUpdated](s.reported()) > 0 }, "an update for no call was reported")
}

func (s *AgentSuite) TestAStatusInWritingIsReportedAndNotPosted() {
	s.ownsTools("")
	s.joinText()
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	s.agent.options.ToolRunner = runner
	var done sync.WaitGroup
	done.Go(func() {
		s.agent.runTool(harness.ToolRequested{TurnID: "turn-1", Call: llm.ToolCall{ID: "call-1", Name: "lookup_order", Arguments: `{}`}})
	})
	<-runner.began

	s.agent.ToolUpdate("call-1", "searching 3 airlines", false)

	s.eventually(func() bool { return countOf[ToolUpdated](s.reported()) == 1 }, "the status was not reported")
	s.Zero(countOf[Responded](s.reported()), "a status in writing is shown on the call, not posted")
	close(runner.release)
	done.Wait()
}

// runningAHeldTool is a call to a tool that runs until released, started on a voice call
// once the reply that asked for it was said, so the line is free.
func (s *AgentSuite) runningAHeldTool() *heldTool {
	s.ownsTools("")
	s.join(true)
	runner := &heldTool{holdingTool: holdingTool{began: make(chan struct{}), release: make(chan struct{})}}
	runner.finish = sync.OnceFunc(func() { close(runner.release) })
	s.T().Cleanup(runner.finish)
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
	s.eventually(func() bool { return s.spokenText("Let me check the order.") }, "the reply was never said")
	s.replyWentOut()
	return runner
}

// heldTool is a holdingTool released once, by the test or when it ends.
type heldTool struct {
	holdingTool
	finish func()
}

// holdTheLine keeps the agent writing a line of its own, so nothing else is said until
// releaseTheLine.
func (s *AgentSuite) holdTheLine() {
	s.agent.mu.Lock()
	s.agent.composing = true
	s.agent.mu.Unlock()
}

func (s *AgentSuite) releaseTheLine() {
	s.agent.mu.Lock()
	s.agent.composing = false
	s.agent.mu.Unlock()
}
