package api

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type CallsSuite struct {
	suite.Suite
	base time.Time
}

func TestCallsSuite(t *testing.T) {
	suite.Run(t, new(CallsSuite))
}

func (s *CallsSuite) SetupTest() {
	s.base = time.Date(2026, 8, 18, 9, 0, 0, 0, time.UTC)
}

// turn is one measured exchange starting so many seconds into the call.
func (s *CallsSuite) turn(id string, second int, roundtrip float64) store.Turn {
	ms := roundtrip
	return store.Turn{
		TurnID:      id,
		StartedAt:   s.base.Add(time.Duration(second) * time.Second),
		RoundtripMs: &ms,
	}
}

// said is one line stored so many seconds into the call.
func (s *CallsSuite) said(speaker, text string, second int) chatlog.Spoken {
	return chatlog.Spoken{
		Speaker: speaker,
		Text:    text,
		At:      s.base.Add(time.Duration(second) * time.Second),
	}
}

func (s *CallsSuite) TestEachExchangeCarriesWhatWasSaidInIt() {
	turns := []store.Turn{s.turn("t1", 0, 420), s.turn("t2", 10, 380)}
	said := []chatlog.Spoken{
		s.said("caller", "do you open on Sundays?", 1),
		s.said("agent", "we do, from ten.", 2),
		s.said("caller", "thanks", 11),
		s.said("agent", "any time.", 12),
	}

	timeline := timelineOf(turns, said)

	s.Require().Len(timeline, 2)
	s.Require().NotNil(timeline[0].Heard)
	s.Equal("do you open on Sundays?", *timeline[0].Heard)
	s.Require().NotNil(timeline[0].Said)
	s.Equal("we do, from ten.", *timeline[0].Said)
	s.Require().NotNil(timeline[1].Heard)
	s.Equal("thanks", *timeline[1].Heard)
	s.Require().NotNil(timeline[1].RoundtripMs)
	s.InDelta(380, *timeline[1].RoundtripMs, 0.001)
}

func (s *CallsSuite) TestAnExchangeNobodyWroteDownIsStillMeasured() {
	// The timings are this service's own, and they are the point of the view. A call
	// whose transcript was never stored still shows what the caller waited for.
	timeline := timelineOf([]store.Turn{s.turn("t1", 0, 500)}, nil)

	s.Require().Len(timeline, 1)
	s.Nil(timeline[0].Heard)
	s.Nil(timeline[0].Said)
	s.Require().NotNil(timeline[0].RoundtripMs)
	s.InDelta(500, *timeline[0].RoundtripMs, 0.001)
}

func (s *CallsSuite) TestWhatWasSaidAfterTheLastExchangeStillBelongsToIt() {
	turns := []store.Turn{s.turn("t1", 0, 400)}
	said := []chatlog.Spoken{
		s.said("caller", "goodbye", 1),
		s.said("agent", "goodbye.", 30),
	}

	timeline := timelineOf(turns, said)

	s.Require().Len(timeline, 1)
	s.Require().NotNil(timeline[0].Said)
	s.Equal("goodbye.", *timeline[0].Said)
}

func (s *CallsSuite) TestALineSaidBeforeTheFirstExchangeIsNotPartOfIt() {
	// A greeting is spoken on joining, before anybody has said anything to answer, so it
	// belongs to no exchange at all.
	turns := []store.Turn{s.turn("t1", 10, 400)}
	said := []chatlog.Spoken{
		s.said("agent", "thanks for calling.", 1),
		s.said("caller", "are you open?", 11),
		s.said("agent", "we are.", 12),
	}

	timeline := timelineOf(turns, said)

	s.Require().Len(timeline, 1)
	s.Require().NotNil(timeline[0].Heard)
	s.Equal("are you open?", *timeline[0].Heard)
}
