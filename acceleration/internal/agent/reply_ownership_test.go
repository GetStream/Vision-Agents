package agent

import (
	"errors"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// interruptBeforeCompletion leaves the old stream's terminal event behind the next
// reply, so these tests do not depend on how the two provider goroutines are scheduled.
func (s *AgentSuite) interruptBeforeCompletion(fragment string) (string, func()) {
	release := make(chan struct{})
	once := new(sync.Once)
	settle := func() { once.Do(func() { close(release) }) }
	s.T().Cleanup(settle)
	s.model.reply = nil
	s.model.holdCompletion = release

	first, err := s.agent.RespondTo(s.ctx, "explain the old request", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(first) != nil }, "the old reply never opened")
	s.model.writes(first, fragment)
	s.eventually(func() bool { return countOf[ResponseDelta](s.reported()) > 0 },
		"the old reply never began")
	s.agent.Interrupt()
	s.model.mu.Lock()
	s.model.holdCompletion = nil
	s.model.mu.Unlock()
	return first, settle
}

func (s *AgentSuite) TestAReplacementReplyDoesNotInheritAnInterruptedSentence() {
	s.join(false)
	s.interruptBeforeCompletion("The old answer was [whis")

	second, err := s.agent.RespondTo(s.ctx, "answer the replacement request", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(second) != nil }, "the replacement never opened")
	s.model.writes(second, "Only the replacement answer.")
	s.model.finishes(second)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the replacement never finished")

	s.Equal([]string{"Only the replacement answer."}, s.answered())
	for _, request := range s.voice.spoken() {
		s.NotContains(request.Text, "old answer")
		s.NotContains(request.Text, "whis")
	}
	for _, message := range s.history() {
		if message.Role == llm.Assistant {
			s.Equal("Only the replacement answer.", message.Content)
		}
	}
}

func (s *AgentSuite) TestAFailedReplacementReplyDoesNotReuseInterruptedText() {
	s.join(false)
	s.interruptBeforeCompletion("The old answer was ")

	second, err := s.agent.RespondTo(s.ctx, "answer the replacement request", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(second) != nil }, "the replacement never opened")
	s.model.script(second).Fail(errors.New("provider unavailable"), "read")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the failed reply never finished")

	s.Equal([]string{lostReply}, s.answered())
	s.True(s.spokenText(lostReply), "the caller never heard that the replacement failed")
}

func (s *AgentSuite) TestALateCompletionDoesNotEraseTheReplacementToolTag() {
	s.ownsTools("order found")
	s.joinText()
	first, settle := s.interruptBeforeCompletion("Let me check. ")

	second, err := s.agent.RespondTo(s.ctx, "check order twelve", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(second) != nil }, "the replacement never opened")
	s.model.writes(second, `<call tool="lookup_`)
	s.model.script(second).ReasoningText("looking up the order")
	s.eventually(func() bool { return countOf[ReasoningDelta](s.reported()) == 1 },
		"the replacement's partial tool tag was never consumed")
	settle()
	// Once the old pump has returned, its completion is ahead of the next delta in
	// the shared channel, even if the consumer has not handled it yet.
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.streams[first] == nil
	}, "the old completion was never delivered")
	s.model.writes(second, `order" order="12"/>`)
	s.model.finishes(second)
	s.eventually(func() bool { return len(s.runner.asked()) == 1 },
		"the old completion erased the replacement's tool call")

	s.Equal("lookup_order", s.runner.asked()[0].Name)
	s.JSONEq(`{"order":"12"}`, s.runner.asked()[0].Arguments)
	s.NotContains(strings.Join(s.answered(), " "), "lookup_")
}

// A reply whose provider breaks as the caller cuts it off is not reported failed: the turn
// was given up on first. A failure of the turn after it still is.
func (s *AgentSuite) TestAReplyThatBreaksAsItIsInterruptedIsNotReportedFailed() {
	s.joinText()
	s.model.reply = nil
	s.model.breaks = errors.New("connection reset by peer")
	first, err := s.agent.RespondTo(s.ctx, "explain the old request", nil, options.LLM{})
	s.Require().NoError(err)
	s.model.writes(first, "The old answer was ")
	s.eventually(func() bool { return countOf[ResponseDelta](s.reported()) > 0 },
		"the old reply never began")
	s.agent.Interrupt()
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.streams[first] == nil
	}, "the old reply was never drained")

	second, err := s.agent.RespondTo(s.ctx, "answer the replacement request", nil, options.LLM{})
	s.Require().NoError(err)
	unavailable := errors.New("provider unavailable")
	s.model.opened(second).Fail(unavailable, "read")
	// The replies share one channel, so the old reply's failure is handled before this.
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the failed reply never finished")

	var failures []error
	for _, event := range s.reported() {
		if failed, ok := event.(Error); ok {
			failures = append(failures, failed.Err)
		}
	}
	s.Equal([]error{unavailable}, failures, "only the turn still being answered failed")
}
