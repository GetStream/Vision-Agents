package agent

import (
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// A line the reply model could not write is logged with the provider and model that failed
// to write it, so a search for either finds the failure.
func (s *AgentSuite) requireLoggedWithReplyModel(message string) {
	s.eventually(func() bool { return len(s.logs.records(message)) == 1 }, "the failure was never logged")
	line := s.logs.records(message)[0]
	s.Equal("stub", line["provider"])
	s.Equal("stub-model", line["model"])
	s.Contains(line["error"], "502 Bad Gateway")
}

func (s *AgentSuite) TestAGreetingTheModelCannotVaryIsLoggedWithItsProviderAndModel() {
	s.logs = &logLines{}
	s.join(true)
	s.model.composeFails = errors.New("502 Bad Gateway: failed to connect to local service")

	s.Require().NoError(s.agent.Greet(s.ctx, "Northwind, how can I help?", true))

	s.requireLoggedWithReplyModel("could not vary the greeting, saying it as written")
}

func (s *AgentSuite) TestACheckInTheModelCannotWriteIsLoggedWithItsProviderAndModel() {
	s.logs = &logLines{}
	s.join(true)
	s.model.composeFails = errors.New("502 Bad Gateway: failed to connect to local service")
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.agent.checkIn(participant, idlePurpose)

	s.requireLoggedWithReplyModel(unwritten)
}

func (s *AgentSuite) TestAToolStatusTheModelCannotWriteIsLoggedWithItsProviderAndModel() {
	s.logs = &logLines{}
	s.runningAHeldTool()
	s.model.composeFails = errors.New("502 Bad Gateway: failed to connect to local service")

	s.agent.ToolUpdate("call-1", "searching 3 airlines", false)

	s.requireLoggedWithReplyModel("could not write a tool's status to say")
}
