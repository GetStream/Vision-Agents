package agent

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (s *AgentSuite) TestAProviderTimeoutSettlesTheReply() {
	s.providerTimeoutSettlesTheReply()
}

func (s *AgentSuite) TestAnEarlyProviderTimeoutSettlesTheReply() {
	s.speculates = true
	s.providerTimeoutSettlesTheReply()
}

func (s *AgentSuite) providerTimeoutSettlesTheReply() {
	s.join(false)
	s.model.refuses = fmt.Errorf("provider request timed out: %w", context.DeadlineExceeded)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is our forecast")

	s.eventually(func() bool { return s.spokenText(lostReply) }, "the timeout left the caller in silence")
	s.eventually(func() bool { return !s.agent.Busy() }, "the timed-out reply still owns the floor")
	s.Equal(1, countOf[Error](s.reported()))

	s.model.mu.Lock()
	s.model.refuses = nil
	s.model.mu.Unlock()
	s.says(participant, "try again please")
	s.eventually(func() bool { return s.spokenText("Hello there") }, "the agent never recovered from the timeout")
}
