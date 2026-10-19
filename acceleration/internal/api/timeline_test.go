package api

import (
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

// TimelineSuite pairs turns with the lines of a transcript the way a call stores them: a
// caller's line when they start talking, the agent's once its turn has started.
type TimelineSuite struct {
	suite.Suite

	start time.Time
}

func TestTimelineSuite(t *testing.T) {
	suite.Run(t, new(TimelineSuite))
}

func (s *TimelineSuite) SetupTest() {
	s.start = time.Date(2026, 10, 19, 18, 26, 0, 0, time.UTC)
}

func (s *TimelineSuite) TestSeveralThingsTheCallerSaidBeforeOneAnswerAreHeardTogether() {
	timeline := timelineOf(s.turns(11_940), []chatlog.Spoken{
		s.caller(9_000, "I need a table."),
		s.caller(10_608, "For two, at eight."),
		s.agent(12_500, "Booked for two at eight."),
	}, nil)

	s.Require().Len(timeline, 1)
	s.Equal("I need a table. For two, at eight.", value(timeline[0].Heard))
	s.Equal("Booked for two at eight.", value(timeline[0].Said))
}

func (s *TimelineSuite) TestATurnCutOffBeforeTheAgentSaidAnythingHasNoAnswer() {
	// An interrupted reply nobody heard is stored empty, which a transcript leaves out.
	timeline := timelineOf(s.turns(11_940, 15_000), []chatlog.Spoken{
		s.caller(10_608, "What time is it?"),
		s.caller(14_200, "Never mind, what day is it?"),
		s.agent(15_800, "It is Monday."),
	}, nil)

	s.Require().Len(timeline, 2)
	s.Equal("What time is it?", value(timeline[0].Heard))
	s.Nil(timeline[0].Said)
	s.Equal("Never mind, what day is it?", value(timeline[1].Heard))
	s.Equal("It is Monday.", value(timeline[1].Said))
}

func (s *TimelineSuite) TestWhatTheCallerSaysOverAnAnswerIsHeardByTheNextTurn() {
	timeline := timelineOf(s.turns(11_940, 15_000), []chatlog.Spoken{
		s.caller(10_608, "Tell me about Rome."),
		s.agent(12_500, "Rome was founded"),
		s.caller(13_900, "Actually, Paris."),
		s.agent(15_600, "Paris is the capital of France."),
	}, nil)

	s.Require().Len(timeline, 2)
	s.Equal("Tell me about Rome.", value(timeline[0].Heard))
	s.Equal("Rome was founded", value(timeline[0].Said))
	s.Equal("Actually, Paris.", value(timeline[1].Heard))
	s.Equal("Paris is the capital of France.", value(timeline[1].Said))
}

func (s *TimelineSuite) TestALineStoredOnlyOnceHeardStillBelongsToItsTurn() {
	// A line nobody saw being spoken is stored when the turn is decided, which is after the
	// transcript settled and the turn started, and still before the agent answers.
	timeline := timelineOf(s.turns(11_940), []chatlog.Spoken{
		s.caller(12_100, "Hi."),
		s.agent(12_900, "Hello there."),
	}, nil)

	s.Require().Len(timeline, 1)
	s.Equal("Hi.", value(timeline[0].Heard))
	s.Equal("Hello there.", value(timeline[0].Said))
}

func (s *TimelineSuite) TestATurnTheAgentStartedHeardNothing() {
	// A greeting, or the answer after a tool ran, is a turn nobody asked for in words.
	timeline := timelineOf(s.turns(1_000, 11_940), []chatlog.Spoken{
		s.agent(1_500, "Hi, how can I help?"),
		s.caller(10_608, "What is the weather?"),
		s.agent(12_500, "Sunny."),
	}, nil)

	s.Require().Len(timeline, 2)
	s.Nil(timeline[0].Heard)
	s.Equal("Hi, how can I help?", value(timeline[0].Said))
	s.Equal("What is the weather?", value(timeline[1].Heard))
	s.Equal("Sunny.", value(timeline[1].Said))
}

// turns are turns starting at offsets from the start, in milliseconds.
func (s *TimelineSuite) turns(offsets ...int) []live.Turn {
	turns := make([]live.Turn, 0, len(offsets))
	for _, offset := range offsets {
		turns = append(turns, live.Turn{TurnID: "turn-" + strconv.Itoa(offset), StartedAt: s.at(offset)})
	}
	return turns
}

func (s *TimelineSuite) caller(offset int, text string) chatlog.Spoken {
	return chatlog.Spoken{Speaker: "alice", Text: text, At: s.at(offset)}
}

func (s *TimelineSuite) agent(offset int, text string) chatlog.Spoken {
	return chatlog.Spoken{Speaker: "agent", Agent: true, Text: text, At: s.at(offset)}
}

func (s *TimelineSuite) at(offset int) time.Time {
	return s.start.Add(time.Duration(offset) * time.Millisecond)
}
