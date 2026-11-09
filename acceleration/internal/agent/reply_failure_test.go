package agent

import (
	"context"
	"crypto/rand"
	"errors"
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
	s.eventually(func() bool { return countOf[Error](s.reported()) == 1 }, "the timeout was never reported")

	s.model.mu.Lock()
	s.model.refuses = nil
	s.model.mu.Unlock()
	s.says(participant, "try again please")
	s.eventually(func() bool { return s.spokenText("Hello there") }, "the agent never recovered from the timeout")
}

func (s *AgentSuite) TestAReplyRequestThatNeverStartedIsLoggedWithItsProviderAndModel() {
	s.replyRequestIsLoggedWithItsProviderAndModel()
}

func (s *AgentSuite) TestAnEarlyReplyRequestThatNeverStartedIsLoggedWithItsProviderAndModel() {
	s.speculates = true
	s.replyRequestIsLoggedWithItsProviderAndModel()
}

// replyRequestIsLoggedWithItsProviderAndModel has the model refuse a reply's request, and
// checks its failure line names what the session routes to, so a search for either finds it.
func (s *AgentSuite) replyRequestIsLoggedWithItsProviderAndModel() {
	s.logs = &logLines{}
	s.join(false)
	s.model.refuses = errors.New("502 Bad Gateway: failed to connect to local service")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is our forecast")

	s.eventually(func() bool { return len(s.logs.records("agent failure")) == 1 }, "the failure was never logged")
	failure := s.logs.records("agent failure")[0]
	s.Equal("llm", failure["context"])
	s.Equal("stub", failure["provider"])
	s.Equal("stub-model", failure["model"])
	s.Contains(failure["error"], "failed to connect to local service")
}

// A provider that echoes the request's headers in its error puts the key it was sent into the
// failure; the failure line keeps the provider's reason and not the key.
func (s *AgentSuite) TestAFailureLineKeepsTheProvidersReasonAndNotTheKeyItEchoed() {
	s.logs = &logLines{}
	key := rand.Text()
	s.join(false)
	s.model.refuses = errors.New("502 Bad Gateway: upstream error: failed to connect to local service\n" +
		"request headers: Authorization: Bearer sk-test-" + key)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is our forecast")

	s.eventually(func() bool { return len(s.logs.records("agent failure")) == 1 }, "the failure was never logged")
	failure := s.logs.records("agent failure")[0]
	s.Contains(failure["error"], "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]")
	s.NotContains(s.logs.buf.String(), key)
}
