//go:build integration

package live

import (
	"time"
)

// turn is one exchange that started at, defaulting what a test does not care about.
func (s *LiveSuite) turn(turnID string, at time.Time, roundtripMs float64) Turn {
	return Turn{CustomerID: s.customer, AgentID: "agent-1", TurnID: turnID, StartedAt: at, RoundtripMs: &roundtripMs}
}

// decision is one judgement made at, defaulting what a test does not care about.
func (s *LiveSuite) decision(kind, reason string, at time.Time) CallEvent {
	latency := 30.0
	return CallEvent{
		CustomerID: s.customer, CallID: "call-1", AgentID: "agent-1", At: at,
		Kind: kind, Reason: reason, TurnID: "turn-1", Participant: "caller",
		Said: "book a table for four", LatencyMs: &latency,
	}
}

func (s *LiveSuite) TestACallsTurnsAreReadBackInTheOrderTheyStarted() {
	now := time.Now()
	queuedMs, audibleMs, heldMs := 880.0, 940.0, 620.0
	first := s.turn("turn-1", now.Add(-2*time.Minute), 1400)
	first.FirstFrameQueuedMs, first.FirstAudibleFrameMs = &queuedMs, &audibleMs
	first.ReplyHoldMs = &heldMs
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("turn-2", now.Add(-time.Minute), 900)))
	s.Require().NoError(s.client.RecordTurn(s.ctx, first))

	turns, err := s.client.CallTurns(s.ctx, s.customer, "agent-1", now.Add(-5*time.Minute), nil)

	s.Require().NoError(err)
	s.Require().Len(turns, 2)
	s.Equal("turn-1", turns[0].TurnID)
	s.InDelta(1400, *turns[0].RoundtripMs, 0.001)
	s.InDelta(880, *turns[0].FirstFrameQueuedMs, 0.001)
	s.InDelta(940, *turns[0].FirstAudibleFrameMs, 0.001)
	s.InDelta(620, *turns[0].ReplyHoldMs, 0.001)
	s.Nil(turns[1].FirstFrameQueuedMs, "an edge that does not report them leaves them out")
}

func (s *LiveSuite) TestACallIsItsAgentsTurnsInTheWindowItRan() {
	now := time.Now()
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("before", now.Add(-10*time.Minute), 100)))
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("during", now.Add(-5*time.Minute), 100)))
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("after", now.Add(-time.Minute), 100)))
	other := s.turn("someone else's", now.Add(-5*time.Minute), 100)
	other.AgentID = "agent-2"
	s.Require().NoError(s.client.RecordTurn(s.ctx, other))

	ended := now.Add(-2 * time.Minute)
	turns, err := s.client.CallTurns(s.ctx, s.customer, "agent-1", now.Add(-6*time.Minute), &ended)

	s.Require().NoError(err)
	s.Require().Len(turns, 1)
	s.Equal("during", turns[0].TurnID)
}

func (s *LiveSuite) TestTurnsAreKeptForAnHour() {
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("old", time.Now().Add(-2*RecentFor), 100)))
	s.Require().NoError(s.client.RecordTurn(s.ctx, s.turn("new", time.Now(), 100)))

	turns, err := s.client.CallTurns(s.ctx, s.customer, "agent-1", time.Now().Add(-3*RecentFor), nil)
	s.Require().NoError(err)
	s.Require().Len(turns, 1, "the turn older than an hour is let go when the next is written")
	s.Equal("new", turns[0].TurnID)

	ttl, err := s.client.Redis().Do(s.ctx, s.client.Redis().B().Pttl().Key(turnsKey(s.customer, "agent-1")).Build()).AsInt64()
	s.Require().NoError(err)
	s.InDelta(RecentFor.Milliseconds(), ttl, float64(time.Minute.Milliseconds()))
}

func (s *LiveSuite) TestWhatTheConversationDecidedIsReadBackInTheOrderItHappened() {
	// The trail only explains a call if it is in order: a wait followed by an answer is
	// an agent giving somebody time to finish, and the same two the other way round is
	// an agent talking over them.
	now := time.Now()
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, []CallEvent{
		s.decision("answer", "a complete thought addressed to the agent", now.Add(-time.Second)),
		s.decision("wait", "the caller has not finished the thought", now.Add(-2*time.Second)),
	}))

	read, err := s.client.CallEvents(s.ctx, s.customer, "call-1", now.Add(-time.Minute), nil, 0)

	s.Require().NoError(err)
	s.Require().Len(read, 2)
	s.Equal("wait", read[0].Kind)
	s.Equal("the caller has not finished the thought", read[0].Reason)
	s.Equal("answer", read[1].Kind)
	s.Equal("caller", read[1].Participant)
	s.Require().NotNil(read[1].LatencyMs)
	s.InDelta(30, *read[1].LatencyMs, 0.001)
}

func (s *LiveSuite) TestOneCallsReasoningIsNotAnothers() {
	mine := s.decision("answer", "mine", time.Now())
	theirs := s.decision("answer", "theirs", time.Now())
	theirs.CallID = "call-2"
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, []CallEvent{mine, theirs}))

	read, err := s.client.CallEvents(s.ctx, s.customer, "call-1", time.Now().Add(-time.Minute), nil, 0)

	s.Require().NoError(err)
	s.Require().Len(read, 1)
	s.Equal("mine", read[0].Reason)
}

func (s *LiveSuite) TestAnEarlierCallOnTheSameIdIsNotThisOnes() {
	// A call id is the room, not the conversation: dialling the same room again is a
	// second call, and the reasoning behind the first one does not explain it.
	started := time.Now().Add(-time.Minute)
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, []CallEvent{
		s.decision("answer", "the call before", started.Add(-10*time.Minute)),
		s.decision("answer", "this call", started.Add(time.Second)),
	}))

	read, err := s.client.CallEvents(s.ctx, s.customer, "call-1", started, nil, 0)

	s.Require().NoError(err)
	s.Require().Len(read, 1)
	s.Equal("this call", read[0].Reason)
}

func (s *LiveSuite) TestADecisionOlderThanAnHourIsGone() {
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, []CallEvent{
		s.decision("answer", "long ago", time.Now().Add(-2*RecentFor)),
		s.decision("answer", "just now", time.Now()),
	}))

	read, err := s.client.CallEvents(s.ctx, s.customer, "call-1", time.Now().Add(-3*RecentFor), nil, 0)

	s.Require().NoError(err)
	s.Require().Len(read, 1)
	s.Equal("just now", read[0].Reason)
}

func (s *LiveSuite) TestALimitKeepsTheEarliestDecisions() {
	now := time.Now()
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, []CallEvent{
		s.decision("wait", "first", now.Add(-3*time.Second)),
		s.decision("wait", "second", now.Add(-2*time.Second)),
		s.decision("answer", "third", now.Add(-time.Second)),
	}))

	read, err := s.client.CallEvents(s.ctx, s.customer, "call-1", now.Add(-time.Minute), nil, 2)

	s.Require().NoError(err)
	s.Require().Len(read, 2)
	s.Equal("second", read[1].Reason)
}

func (s *LiveSuite) TestRecordingNoDecisionsIsNotAnError() {
	s.Require().NoError(s.client.RecordCallEvents(s.ctx, nil))
}
