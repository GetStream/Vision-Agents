package agent

import (
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

type ClockSuite struct {
	suite.Suite
}

func TestClockSuite(t *testing.T) {
	suite.Run(t, new(ClockSuite))
}

// withoutClock is a turn's instructions without the clock line they end with, for a test
// about the rest of them.
func withoutClock(instructions string) string {
	before, _, _ := strings.Cut(instructions, "\n\nToday is ")
	return before
}

func (s *ClockSuite) zone(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	s.Require().NoError(err)
	return loc
}

func (s *ClockSuite) TestWithoutTheUsersZoneTheLineIsInUTCAndSaysTheZoneIsUnknown() {
	s.Equal("Today is Wednesday, 28 October 2026 in UTC, between 14:00 and 15:00 UTC. "+
		"The user's timezone is not known: take a time of day they give in their own local time, "+
		"and ask for their zone only when they bring up another one or a tool needs it.",
		clockLine(time.Date(2026, 10, 28, 14, 42, 0, 0, time.UTC), nil))
}

func (s *ClockSuite) TestTheLineReadsTheUsersDayAndHourBesideUTC() {
	s.Equal("Today is Wednesday, 21 October 2026 for the user (Europe/Amsterdam), "+
		"between 01:00 and 02:00 there, between 23:00 and 00:00 UTC.",
		clockLine(time.Date(2026, 10, 20, 23, 30, 0, 0, time.UTC), s.zone("Europe/Amsterdam")))
}

func (s *ClockSuite) TestTheUsersHourIsTheSameHourAsUTCsInAZoneOffByHalfAnHour() {
	s.Equal("Today is Wednesday, 28 October 2026 for the user (Asia/Kolkata), "+
		"between 19:30 and 20:30 there, between 14:00 and 15:00 UTC.",
		clockLine(time.Date(2026, 10, 28, 14, 42, 0, 0, time.UTC), s.zone("Asia/Kolkata")))
}

func (s *ClockSuite) TestTheLineIsTheSameForEveryMinuteOfAnHour() {
	zone := s.zone("Europe/Rome")

	s.Equal(clockLine(time.Date(2026, 10, 28, 14, 0, 0, 0, time.UTC), zone),
		clockLine(time.Date(2026, 10, 28, 14, 59, 59, 0, time.UTC), zone))
}

func (s *AgentSuite) TestATurnIsToldTheDateAndHourInUTCWhenTheUsersZoneIsNotKnown() {
	s.joinText()
	s.agent.now = func() time.Time { return time.Date(2026, 10, 28, 14, 42, 0, 0, time.UTC) }

	_, err := s.agent.RespondTo(s.ctx, "can I come in next Wednesday?", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the turn never reached the model")
	told := s.model.requests()[0].Instructions
	s.True(strings.HasPrefix(told, "be brief\n\n"), told)
	s.True(strings.HasSuffix(told, "\n\nToday is Wednesday, 28 October 2026 in UTC, between 14:00 and 15:00 UTC. "+
		"The user's timezone is not known: take a time of day they give in their own local time, "+
		"and ask for their zone only when they bring up another one or a tool needs it."), told)
}

func (s *AgentSuite) TestATurnIsToldTheUsersDayInTheirZone() {
	s.timezone = "Europe/Rome"
	s.joinText()
	s.agent.now = func() time.Time { return time.Date(2026, 10, 28, 14, 42, 0, 0, time.UTC) }

	_, err := s.agent.RespondTo(s.ctx, "can I come in tomorrow?", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the turn never reached the model")
	s.True(strings.HasSuffix(s.model.requests()[0].Instructions, "\n\nToday is Wednesday, 28 October 2026 "+
		"for the user (Europe/Rome), between 15:00 and 16:00 there, between 14:00 and 15:00 UTC."),
		s.model.requests()[0].Instructions)
}

func (s *AgentSuite) TestTheClockMovesOnBetweenTheTurnsOfOneSession() {
	s.timezone = "Europe/Rome"
	s.joinText()
	s.agent.now = func() time.Time { return time.Date(2026, 10, 28, 14, 42, 0, 0, time.UTC) }
	_, err := s.agent.RespondTo(s.ctx, "hello", nil, options.LLM{})
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the first turn never reached the model")
	s.eventually(func() bool { return !s.agent.Busy() }, "the first turn never finished")

	s.agent.now = func() time.Time { return time.Date(2026, 10, 29, 8, 5, 0, 0, time.UTC) }
	_, err = s.agent.RespondTo(s.ctx, "and what day is it now?", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the second turn never reached the model")
	s.Contains(s.model.requests()[1].Instructions,
		"Today is Thursday, 29 October 2026 for the user (Europe/Rome), between 09:00 and 10:00 there")
}

func (s *AgentSuite) TestAnAgentForAZoneNobodyKnowsIsRefused() {
	s.model, s.flow = newStubLLM(), newStubLLM()

	_, err := New(Options{
		Text: true, CustomerID: "acme", Timezone: "Nowhere/Special",
		LLM: s.reasoner(slog.New(slog.DiscardHandler)), LLMTarget: "en-low-latency",
	})

	s.ErrorContains(err, `"Nowhere/Special" is not a timezone I know`)
}
