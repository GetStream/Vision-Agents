package agent

import (
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The cadence is driven at a fraction of its real gap, because what these assert is what
// is decided rather than how long the deciding waits.
const (
	testGap    = 10 * time.Millisecond
	testRetry  = 20 * time.Millisecond
	testWithin = time.Second
)

var caller = stt.Participant{ID: "caller", UserID: "caller", Name: "Alex"}

type ConverseSuite struct {
	suite.Suite

	settling *cadence
	emitter  *Emitter
	converse *converse

	drained sync.WaitGroup
	mu      sync.Mutex
	events  []Event
}

func TestConverseSuite(t *testing.T) {
	suite.Run(t, new(ConverseSuite))
}

func (s *ConverseSuite) SetupTest() {
	s.build(DuplexOptions{})
}

// build starts a conversation with the given listening options and watches everything it
// reports, which is how the decision trail is asserted on.
func (s *ConverseSuite) build(options DuplexOptions) {
	s.teardown()

	logger := slog.New(slog.DiscardHandler)
	s.settling = newCadence(testGap, testRetry, time.Hour, logger)
	s.emitter = NewEmitter(eventBuffer)
	s.converse = newConverse(s.settling, newDuplex(options), s.emitter, nil, logger)

	s.mu.Lock()
	s.events = nil
	s.mu.Unlock()

	s.drained.Add(1)
	go func() {
		defer s.drained.Done()
		for event := range s.emitter.Events() {
			s.mu.Lock()
			s.events = append(s.events, event)
			s.mu.Unlock()
		}
	}()
}

func (s *ConverseSuite) TearDownTest() { s.teardown() }

func (s *ConverseSuite) teardown() {
	if s.emitter == nil {
		return
	}
	s.settling.Close()
	s.emitter.Close()
	s.drained.Wait()
	s.emitter = nil
}

// settle says something and waits for the words to hold still, which is the state every
// ruling is made against.
func (s *ConverseSuite) settle(text string, state floor) candidate {
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        text,
	}, state)

	select {
	case ready := <-s.settling.Ready():
		return s.converse.Settled(ready, state).Candidate
	case <-time.After(testWithin):
		s.FailNow("the words never held still", "text %q", text)
		return candidate{}
	}
}

// talking is an agent mid-reply, which is what makes the floor half of a ruling matter.
func (s *ConverseSuite) talking() floor {
	return floor{Speaking: "turn-1", LastParticipant: caller}
}

func (s *ConverseSuite) quiet() floor {
	return floor{Quiet: true, LastParticipant: caller}
}

func (s *ConverseSuite) seen() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Event(nil), s.events...)
}

// eventually gives the emitter a moment to hand an event to the watcher, since sending is
// asynchronous and an assertion made straight away would race it.
func (s *ConverseSuite) eventually(satisfied func() bool, message string) {
	s.T().Helper()
	deadline := time.Now().Add(testWithin)
	for time.Now().Before(deadline) {
		if satisfied() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	s.FailNow(message)
}

func (s *ConverseSuite) heard() []Heard {
	var kept []Heard
	for _, event := range s.seen() {
		if typed, ok := event.(Heard); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func (s *ConverseSuite) overlaps() []OverlapDecided {
	var kept []OverlapDecided
	for _, event := range s.seen() {
		if typed, ok := event.(OverlapDecided); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func (s *ConverseSuite) decisions() []Decided {
	var kept []Decided
	for _, event := range s.seen() {
		if typed, ok := event.(Decided); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func kinds(actions []Action) []ActionKind {
	found := make([]ActionKind, 0, len(actions))
	for _, action := range actions {
		found = append(found, action.Kind)
	}
	return found
}

func (s *ConverseSuite) TestWhatIsDecidedAboutATurnAndWhoHasTheFloor() {
	// The two halves of a ruling are independent: what the caller said settles whether
	// they get an answer, and whether the agent was talking settles when.
	cases := []struct {
		name        string
		disposition harness.Disposition
		floor       harness.Floor
		speaking    bool
		expected    []ActionKind
		clarify     bool
	}{
		{
			name:        "an unfinished thought is left alone",
			disposition: harness.Wait,
			floor:       harness.Continue,
			expected:    []ActionKind{ActWait},
		},
		{
			name:        "speech meant for somebody else is dropped",
			disposition: harness.Ignore,
			floor:       harness.Continue,
			expected:    []ActionKind{ActIgnore},
		},
		{
			name:        "a complete thought is answered",
			disposition: harness.Respond,
			floor:       harness.Continue,
			expected:    []ActionKind{ActAnswer},
		},
		{
			name:        "an ambiguous request is answered with a question",
			disposition: harness.Clarify,
			floor:       harness.Continue,
			expected:    []ActionKind{ActAnswer},
			clarify:     true,
		},
		{
			name:        "a correction takes the floor from the agent",
			disposition: harness.Respond,
			floor:       harness.Stop,
			speaking:    true,
			expected:    []ActionKind{ActInterrupt, ActAnswer},
		},
		{
			name:        "an addition cuts the reply short and waits its turn",
			disposition: harness.Respond,
			floor:       harness.Shorten,
			speaking:    true,
			expected:    []ActionKind{ActQueue, ActShorten},
		},
		{
			name:        "an acknowledgement lets the agent finish first",
			disposition: harness.Respond,
			floor:       harness.Continue,
			speaking:    true,
			expected:    []ActionKind{ActQueue},
		},
		{
			name:        "a turn ignored over the agent's speech leaves the reply alone",
			disposition: harness.Ignore,
			floor:       harness.Stop,
			speaking:    true,
			expected:    []ActionKind{ActIgnore},
		},
	}

	for _, test := range cases {
		s.Run(test.name, func() {
			s.build(DuplexOptions{})

			state := s.quiet()
			if test.speaking {
				state = s.talking()
			}
			ready := s.settle("book a table for four", state)

			actions := s.converse.Ruled(harness.Decided{
				CandidateID: ready.ID,
				Disposition: test.disposition,
				Floor:       test.floor,
				TookMs:      12,
			}, state)

			s.Equal(test.expected, kinds(actions))
			for _, action := range actions {
				if action.Kind == ActAnswer || action.Kind == ActQueue {
					s.Equal(test.clarify, action.Clarify)
					s.Equal(ready.ID, action.Candidate.ID)
				}
				if action.Kind == ActInterrupt || action.Kind == ActShorten {
					s.Equal("turn-1", action.TurnID, "the reply in flight is what is cut")
				}
			}
		})
	}
}

func (s *ConverseSuite) TestOnlyAnAcceptedTurnCountsAsSomethingTheCallerSaid() {
	// A turn the agent decided not to answer was still heard by the transcriber, and
	// reporting it as heard would put words in the conversation nobody acted on.
	ready := s.settle("mumbling in the background", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Ignore,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Empty(s.heard())

	answered := s.settle("book a table for four", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: answered.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())

	s.eventually(func() bool { return len(s.heard()) == 1 }, "the answered turn was never reported")
	s.Equal("book a table for four", s.heard()[0].Text)
}

func (s *ConverseSuite) TestAnUnfinishedThoughtIsPutAgainOnceTheCallerStops() {
	// Waiting must not lose the turn: the same words come back after a longer pause.
	ready := s.settle("book a table", s.quiet())

	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	select {
	case again := <-s.settling.Ready():
		s.Equal("book a table", again.Text)
		s.NotEqual(ready.ID, again.ID)
	case <-time.After(testWithin):
		s.Fail("the caller lost their turn to a controller that wanted to wait")
	}
}

func (s *ConverseSuite) TestARulingAboutWordsThatHaveChangedIsNotActedOn() {
	ready := s.settle("book a table", s.quiet())

	actions := s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        "book a table for four",
	}, s.quiet())

	s.Equal([]ActionKind{ActSupersede}, kinds(actions))
	s.Equal(ready.ID, actions[0].TurnID)

	s.Empty(s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet()), "answering the words as they were would answer half a sentence")
}

func (s *ConverseSuite) TestARulingForATurnThatHasMovedOnIsIgnored() {
	s.Empty(s.converse.Ruled(harness.Decided{
		CandidateID: "turn-gone",
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet()))
}

func (s *ConverseSuite) TestAControllerThatDidNotAnswerDoesNotCostTheCallerTheirTurn() {
	ready := s.settle("book a table", s.quiet())

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: "nonsense",
		Floor:       harness.Continue,
	}, s.quiet())

	s.Equal([]ActionKind{ActFail}, kinds(actions))
	s.Error(actions[0].Err)

	select {
	case again := <-s.settling.Ready():
		s.Equal("book a table", again.Text)
	case <-time.After(testWithin):
		s.Fail("a failed ruling swallowed the caller's turn")
	}
}

func (s *ConverseSuite) TestATurnHeldOverTheAgentIsAnsweredOnceItStopsTalking() {
	ready := s.settle("and make it eight o'clock", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.talking())

	_, waiting := s.converse.Waiting(s.talking())
	s.False(waiting, "answering while still speaking would have the agent talk over itself")

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(ActAnswer, action.Kind)
	s.Equal("and make it eight o'clock", action.Candidate.Text)

	_, waiting = s.converse.Waiting(s.quiet())
	s.False(waiting, "a turn is answered once")
}

func (s *ConverseSuite) TestOnlyTheLastThingSaidOverTheAgentIsAnswered() {
	// A caller who says three things while being talked over is owed an answer to the
	// last of them, not a reply to each in turn once the agent finally stops.
	for _, said := range []string{"actually", "make it eight", "make it nine"} {
		ready := s.settle(said, s.talking())
		s.converse.Ruled(harness.Decided{
			CandidateID: ready.ID,
			Disposition: harness.Respond,
			Floor:       harness.Continue,
		}, s.talking())
	}

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal("make it nine", action.Candidate.Text)
}

func (s *ConverseSuite) TestAnsweringANewTurnAbandonsTheWorkTheLastOneAskedFor() {
	first := s.settle("what is the weather", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: first.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())

	second := s.settle("never mind, book a table", s.quiet())
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: second.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Require().Len(actions, 1)
	s.Equal(first.ID, actions[0].Supersede,
		"the caller moved on, so what the last turn asked for is not wanted")
}

func (s *ConverseSuite) TestWhoKeepsTheFloorIsReported() {
	ready := s.settle("actually", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Shorten,
	}, s.talking())

	s.eventually(func() bool { return len(s.overlaps()) == 1 }, "the overlap was never reported")
	s.Equal("shorten", s.overlaps()[0].Action)
	s.Equal("turn-1", s.overlaps()[0].TurnID)
}

func (s *ConverseSuite) TestAMurmurNeedsSomethingWorthAcknowledgingAndAQuietAgent() {
	s.build(DuplexOptions{Backchannel: true, BackchannelWords: 4})

	s.Empty(kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so",
	}, s.quiet())), "acknowledging one word is interrupting")

	s.Equal([]ActionKind{ActBackchannel}, kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so I was wondering whether",
	}, s.quiet())))
}

func (s *ConverseSuite) TestTheAgentDoesNotMurmurOverItself() {
	s.build(DuplexOptions{Backchannel: true, BackchannelWords: 2})

	s.Empty(kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so I was wondering whether",
	}, s.talking())))
}

func (s *ConverseSuite) TestALongSilenceWhileWorkRunsIsFilled() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.Delegating = true
	state.LastSpokeAt = time.Now().Add(-time.Second)

	s.Equal([]ActionKind{ActBackchannel}, kinds(s.converse.Tick(state)))
}

func (s *ConverseSuite) TestACallNobodyHasSpokenOnIsAskedWhetherAnythingElseIsNeeded() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	actions := s.converse.Tick(state)
	s.Require().Equal([]ActionKind{ActCheckIn}, kinds(actions))
	s.Contains(actions[0].Text, "anything else")
}

func (s *ConverseSuite) TestACallThatHasJustGoneQuietIsLeftAlone() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Second)

	s.Empty(s.converse.Tick(state), "a pause is somebody thinking, not a call that has died")
}

func (s *ConverseSuite) TestSomebodyStillTalkingIsNotASilentCall() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Hour)
	state.LastHeardAt = time.Now()

	s.Empty(s.converse.Tick(state), "the caller spoke a moment ago, so the agent owes them an answer")
}

func (s *ConverseSuite) TestTheWordsAreReportedAsTheyArrive() {
	// A watcher sees the caller's sentence build up, which is what makes a live call
	// worth watching rather than waiting out.
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeDelta, Text: "book a ",
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeDelta, Text: "table",
	}, s.quiet())

	s.eventually(func() bool { return len(s.hearing()) == 2 }, "the words were never reported")
	s.Equal("book a", s.hearing()[0].Text)
	s.Equal("book a table", s.hearing()[1].Text,
		"a delta is a piece of a sentence, and the sentence is what somebody reads")
}

func (s *ConverseSuite) hearing() []Hearing {
	var kept []Hearing
	for _, event := range s.seen() {
		if typed, ok := event.(Hearing); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func (s *ConverseSuite) TestEveryJudgementIsReportedWithItsReason() {
	// The point of the trail is that a call can be read back. A decision with no reason
	// on it is a line in a log that explains nothing.
	ready := s.settle("book a table for four", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
		TookMs:      42,
	}, s.quiet())

	s.eventually(func() bool { return len(s.decisions()) == 2 }, "the decisions were never reported")

	asked := s.decisions()[0]
	s.Equal(string(ActAsk), asked.Kind)
	s.NotEmpty(asked.Reason)
	s.Equal(ready.ID, asked.TurnID)

	answered := s.decisions()[1]
	s.Equal(string(ActAnswer), answered.Kind)
	s.NotEmpty(answered.Reason)
	s.Equal("book a table for four", answered.Text)
	s.Equal(caller, answered.Participant)
	s.InDelta(42, answered.LatencyMs, 0.001)
	s.False(answered.At.IsZero())
}
