package conversation

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// What a running call reports is its step's summary until it ends, when the outcome is.
func (s *DisplaySuite) TestARunningCallsStatusIsItsStepsSummaryUntilItEnds() {
	c := s.open("athena")
	c.ShowTools([]string{"search_*"})
	_, err := c.BeginCommand("command-a", "Look it up", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "search_web", StartedAt: time.Now().UTC()})
	before := current(c).Sequence

	c.Observe(agent.ToolUpdated{ID: "call-1", Tool: "search_web", Text: "searching 3 airlines"})

	reported := current(c)
	s.Greater(reported.Sequence, before, "the status is published")
	s.Equal("searching 3 airlines", reported.Parts[0].Summary)
	s.Equal("searching 3 airlines", reported.Tools[0].Summary)

	c.Observe(agent.ToolRan{ID: "call-1", Tool: "search_web", Result: `{}`})
	s.Empty(current(c).Parts[0].Summary, "a finished call no longer reports it is searching")
	s.Equal("Finished", current(c).Tools[0].Summary)
}

// Words a call has said are the reply's, not its step's status.
func (s *DisplaySuite) TestWordsACallSaysAreNotItsStatus() {
	c := s.open("athena")
	c.ShowTools([]string{"search_*"})
	_, err := c.BeginCommand("command-a", "Look it up", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "search_web", StartedAt: time.Now().UTC()})

	c.Observe(agent.ToolUpdated{ID: "call-1", Tool: "search_web", Text: "One moment.", Verbatim: true})

	s.Empty(current(c).Parts[0].Summary)
}

// A status that comes after its call ended changes nothing.
func (s *DisplaySuite) TestALateStatusDoesNotReopenAFinishedStep() {
	c := s.open("athena")
	c.ShowTools([]string{"search_*"})
	_, err := c.BeginCommand("command-a", "Look it up", "")
	s.Require().NoError(err)
	c.Observe(agent.ToolStarted{ID: "call-1", Tool: "search_web", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "call-1", Tool: "search_web", Result: `{}`})
	finished := current(c)

	c.Observe(agent.ToolUpdated{ID: "call-1", Tool: "search_web", Text: "searching 3 airlines"})

	s.Equal(finished, current(c))
}
