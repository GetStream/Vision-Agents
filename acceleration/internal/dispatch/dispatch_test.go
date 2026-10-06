package dispatch

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type PoolSuite struct {
	suite.Suite
	pool *Pool
}

func TestPoolSuite(t *testing.T) {
	suite.Run(t, new(PoolSuite))
}

func (s *PoolSuite) SetupTest() {
	s.pool = NewPool()
}

// waiting registers a worker of this capacity that does not report work finished, which is
// what a worker built against a router that never asked for it is.
func (s *PoolSuite) waiting(customerID string, capacity int) (*Worker, func()) {
	return s.pool.Register(customerID, Registration{Capacity: capacity})
}

// reporting registers a worker that reports each piece of work finished, which is what
// holds it to the capacity it declared.
func (s *PoolSuite) reporting(customerID string, capacity int) (*Worker, func()) {
	return s.pool.Register(customerID, Registration{Capacity: capacity, Tracking: true})
}

// received drains what a worker was handed, so a test asserts on calls rather than on
// channel mechanics. A released worker's channel is closed, which reads forever, so the
// closed case has to end the drain rather than being taken as a call.
func (s *PoolSuite) received(worker *Worker) []string {
	var ids []string
	for {
		select {
		case call, open := <-worker.Calls():
			if !open {
				return ids
			}
			ids = append(ids, call.CallID)
		default:
			return ids
		}
	}
}

// handed takes the next call off a worker's queue, which is also how a test learns the id
// the pool gave that piece of work so it can report it finished.
func (s *PoolSuite) handed(worker *Worker) Call {
	select {
	case call := <-worker.Calls():
		return call
	default:
		s.FailNow("the worker was handed nothing")
		return Call{}
	}
}

// answered drains the messages a worker was handed, the way received drains its calls.
func (s *PoolSuite) answered(worker *Worker) []string {
	var texts []string
	for {
		select {
		case message, open := <-worker.Messages():
			if !open {
				return texts
			}
			texts = append(texts, message.Text)
		default:
			return texts
		}
	}
}

func (s *PoolSuite) TestTwoWorkersSplitTheCallsBetweenThem() {
	first, _ := s.waiting("acme", 10)
	second, _ := s.waiting("acme", 10)

	for _, id := range []string{"call-1", "call-2", "call-3", "call-4"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1", "call-3"}, s.received(first))
	s.Equal([]string{"call-2", "call-4"}, s.received(second))
}

func (s *PoolSuite) TestAWorkerThatLeftIsNotOfferedCalls() {
	first, _ := s.waiting("acme", 10)
	second, release := s.waiting("acme", 10)

	release()
	for _, id := range []string{"call-1", "call-2"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1", "call-2"}, s.received(first))
	s.Empty(s.received(second), "a released worker's channel is closed, not written to")
}

func (s *PoolSuite) TestTheRotationSurvivesTheWorkerWhoseTurnItWasLeaving() {
	first, _ := s.waiting("acme", 10)
	_, release := s.waiting("acme", 10)

	// Advance the cursor to the second worker, then take it away.
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)
	release()

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-2"})

	s.Require().NoError(err)
	s.Equal(first.ID, assigned.ID)
	s.Equal([]string{"call-1", "call-2"}, s.received(first))
}

func (s *PoolSuite) TestAFullWorkerIsPassedOverRatherThanWaitedFor() {
	full, _ := s.waiting("acme", 1)
	free, _ := s.waiting("acme", 5)

	first, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)
	s.Equal(full.ID, first.ID)

	// full now holds its one call, so the next two both have to go to free even though
	// the rotation would otherwise come back around.
	for _, id := range []string{"call-2", "call-3"} {
		assigned, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
		s.Equal(free.ID, assigned.ID)
	}

	s.Equal([]string{"call-1"}, s.received(full))
	s.Equal([]string{"call-2", "call-3"}, s.received(free))
}

func (s *PoolSuite) TestACallWithNowhereToGoIsRefused() {
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().Error(err)
	s.True(errors.Is(err, ErrNoWorkers), "the caller has to tell this apart from a bad call")
}

func (s *PoolSuite) TestEveryWorkerBeingFullIsNotTheSameAsThereBeingNone() {
	s.waiting("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	_, err = s.pool.Assign("acme", Call{CallID: "call-2"})

	s.Require().Error(err)
	s.False(errors.Is(err, ErrNoWorkers))
	s.ErrorContains(err, "at capacity")
}

func (s *PoolSuite) TestOneCustomersCallsNeverReachAnothersWorkers() {
	ours, _ := s.waiting("acme", 10)
	theirs, _ := s.waiting("globex", 10)

	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	s.Equal([]string{"call-1"}, s.received(ours))
	s.Empty(s.received(theirs))

	_, err = s.pool.Assign("globex", Call{CallID: "call-2"})
	s.Require().NoError(err)
	s.Equal([]string{"call-2"}, s.received(theirs))
}

func (s *PoolSuite) TestACallHasToNameTheCallItIs() {
	s.waiting("acme", 10)

	_, err := s.pool.Assign("acme", Call{CalledNumber: "+15125551234"})

	s.ErrorContains(err, "needs an id")
}

func (s *PoolSuite) TestTwoWorkersSplitTheMessagesBetweenThem() {
	first, _ := s.waiting("acme", 10)
	second, _ := s.waiting("acme", 10)

	for _, text := range []string{"one", "two", "three", "four"} {
		_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: text})
		s.Require().NoError(err)
	}

	s.Equal([]string{"one", "three"}, s.answered(first))
	s.Equal([]string{"two", "four"}, s.answered(second))
}

func (s *PoolSuite) TestAWorkerHoldingItsCapacityInCallsCanStillBeWrittenTo() {
	// The two queues are separate on purpose. Answering a message costs a model call
	// rather than a call's worth of audio, so making somebody wait for a phone line to
	// free up before their message is read would be the wrong queue entirely.
	worker, _ := s.waiting("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	assigned, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-2", Text: "hello"})

	s.Require().NoError(err)
	s.Equal(worker.ID, assigned.ID)
	s.Equal([]string{"hello"}, s.answered(worker))
}

func (s *PoolSuite) TestAFullWorkerIsPassedOverForMessagesToo() {
	full, _ := s.waiting("acme", 1)
	free, _ := s.waiting("acme", 5)

	first, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "one"})
	s.Require().NoError(err)
	s.Equal(full.ID, first.ID)

	for _, text := range []string{"two", "three"} {
		assigned, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: text})
		s.Require().NoError(err)
		s.Equal(free.ID, assigned.ID)
	}

	s.Equal([]string{"one"}, s.answered(full))
	s.Equal([]string{"two", "three"}, s.answered(free))
}

func (s *PoolSuite) TestAMessageWithNowhereToGoIsRefused() {
	_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "hello"})

	s.Require().Error(err)
	s.True(errors.Is(err, ErrNoWorkers))
}

func (s *PoolSuite) TestAMessageHasToNameWhereItWasWritten() {
	// Answering anywhere else would be a reply nobody asked for in a conversation nobody
	// is reading.
	s.waiting("acme", 10)

	_, err := s.pool.AssignMessage("acme", Message{Text: "hello"})

	s.ErrorContains(err, "needs a channel or a session")
}

func (s *PoolSuite) TestOneCustomersMessagesNeverReachAnothersWorkers() {
	ours, _ := s.waiting("acme", 10)
	theirs, _ := s.waiting("globex", 10)

	_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "ours"})
	s.Require().NoError(err)

	s.Equal([]string{"ours"}, s.answered(ours))
	s.Empty(s.answered(theirs))
}

func (s *PoolSuite) TestAWorkerThatLeftIsNotOfferedMessages() {
	first, _ := s.waiting("acme", 10)
	second, release := s.waiting("acme", 10)

	release()
	_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "hello"})
	s.Require().NoError(err)

	s.Equal([]string{"hello"}, s.answered(first))
	s.Empty(s.answered(second), "a released worker's channels are closed, not written to")
}

func (s *PoolSuite) TestAWorkersLoadIsWhatItLastReported() {
	worker, _ := s.waiting("acme", 10)

	worker.Report(Load{ActiveAgents: 3, CPUPercent: 41.5, MemoryPercent: 62.0, LatencyMs: 18.25})

	load := worker.Load()
	s.Equal(3, load.ActiveAgents)
	s.InDelta(41.5, load.CPUPercent, 0.001)
	s.InDelta(18.25, load.LatencyMs, 0.001)
	s.False(load.At.IsZero(), "a report with no time on it is stamped when it arrives")
}

func (s *PoolSuite) TestAWorkerAtTheCapacityItDeclaredIsPassedOver() {
	// The channel emptying as the frame is written is not the worker finishing the call, so
	// without this a worker that promised one call would be handed its fiftieth.
	full, _ := s.reporting("acme", 1)
	free, _ := s.reporting("acme", 5)

	for _, id := range []string{"call-1", "call-2", "call-3"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1"}, s.received(full))
	s.Equal([]string{"call-2", "call-3"}, s.received(free))
}

func (s *PoolSuite) TestFinishingWorkGivesAWorkerItsRoomBack() {
	worker, _ := s.reporting("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)
	first := s.handed(worker)

	_, err = s.pool.Assign("acme", Call{CallID: "call-2"})
	s.Require().ErrorContains(err, "at capacity")

	worker.Done(first.WorkID)
	_, err = s.pool.Assign("acme", Call{CallID: "call-2"})

	s.Require().NoError(err)
	s.Equal([]string{"call-2"}, s.received(worker))
}

func (s *PoolSuite) TestTheWorkerHoldingTheLeastTakesTheNextCall() {
	// Taking turns alone would hand this to the busy worker, which is the first in the
	// rotation and already answering two callers.
	busy, _ := s.pool.Register("acme", Registration{Capacity: 4, Active: 2, Tracking: true})
	idle, _ := s.reporting("acme", 4)

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(idle.ID, assigned.ID)
	s.Empty(s.received(busy))
}

func (s *PoolSuite) TestWhatAWorkerIsHoldingIsReadAsAShareOfWhatItPromised() {
	// A worker that promised ten should be answering more callers than one that promised
	// two, not the same number.
	wide, _ := s.reporting("acme", 10)
	narrow, _ := s.reporting("acme", 2)

	for _, id := range []string{"call-1", "call-2", "call-3", "call-4", "call-5"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Len(s.received(wide), 4)
	s.Len(s.received(narrow), 1)
}

func (s *PoolSuite) TestWorkersHoldingTheSameShareTakeTurns() {
	first, _ := s.reporting("acme", 4)
	second, _ := s.reporting("acme", 4)

	for _, id := range []string{"call-1", "call-2", "call-3", "call-4"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1", "call-3"}, s.received(first))
	s.Equal([]string{"call-2", "call-4"}, s.received(second))
}

func (s *PoolSuite) TestAWorkerWhoseHostIsInTroubleIsLeftAlone() {
	struggling, _ := s.reporting("acme", 4)
	healthy, _ := s.reporting("acme", 4)
	struggling.Report(Load{CPUPercent: 95})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(healthy.ID, assigned.ID, "it is first in the rotation and holding nothing, and still should not get this")
}

func (s *PoolSuite) TestAWorkerInTroubleIsStillBetterThanNobody() {
	// The alternative to a struggling worker is a caller listening to a phone nobody picks
	// up, which is worse than a slow answer.
	struggling, _ := s.reporting("acme", 4)
	struggling.Report(Load{MemoryPercent: 97})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(struggling.ID, assigned.ID)
}

func (s *PoolSuite) TestAReportTooOldToTrustSaysNothingAboutTheHost() {
	// A worker is judged on what it said recently or not at all. One passed over for a
	// figure from an hour ago would be passed over for good.
	stale, _ := s.reporting("acme", 4)
	s.reporting("acme", 4)
	stale.Report(Load{CPUPercent: 95, At: time.Now().UTC().Add(-time.Hour)})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(stale.ID, assigned.ID, "nothing recent was said about it, so the rotation decides")
}

func (s *PoolSuite) TestACallAndAMessageFillTheSameCapacity() {
	// What a worker can hold is what it can hold. A message costs it an agent and a model
	// call, which is not free because it arrived in writing.
	worker, _ := s.reporting("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	_, err = s.pool.AssignMessage("acme", Message{ChannelID: "call-2", Text: "hello"})

	s.Require().ErrorContains(err, "at capacity")
	s.Empty(s.answered(worker))
}

func (s *PoolSuite) TestWorkCarriedThroughAReconnectCountsAgainstCapacity() {
	// The pool that handed that work out has gone, so the one taking over knows about it
	// only because the worker arrived saying so.
	worker, _ := s.pool.Register("acme", Registration{Capacity: 2, Active: 2, Tracking: true})

	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().ErrorContains(err, "at capacity")
	s.Empty(s.received(worker))
}

func (s *PoolSuite) TestFinishingWorkFromBeforeAReconnectGivesItsRoomBack() {
	// It is named by an id this pool never handed out, which is the only thing left of the
	// one that did.
	worker, _ := s.pool.Register("acme", Registration{Capacity: 1, Active: 1, Tracking: true})

	worker.Done("work-from-the-connection-before")
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal([]string{"call-1"}, s.received(worker))
}

func (s *PoolSuite) TestAWorkerIsNotHandedAKindOfWorkItDoesNotAnswer() {
	// One that only answers in writing would drop the call, and the caller would sit
	// listening to a phone nobody picks up.
	writer, _ := s.pool.Register("acme", Registration{Capacity: 4, Handles: []Kind{Messages}})
	caller, _ := s.pool.Register("acme", Registration{Capacity: 4, Handles: []Kind{Calls}})

	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)
	_, err = s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "hello"})
	s.Require().NoError(err)

	s.Equal([]string{"call-1"}, s.received(caller))
	s.Empty(s.received(writer))
	s.Equal([]string{"hello"}, s.answered(writer))
	s.Empty(s.answered(caller))
}

func (s *PoolSuite) TestWorkNobodyAnswersIsRefusedForTheReasonItWas() {
	// Not the same failure as everybody being full: nobody here was ever going to take it,
	// and an operator reading the log should not go looking for capacity.
	s.pool.Register("acme", Registration{Capacity: 4, Handles: []Kind{Messages}})

	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().Error(err)
	s.False(errors.Is(err, ErrNoWorkers), "somebody is waiting; they just do not answer phones")
	s.ErrorContains(err, "handles call")
}

func (s *PoolSuite) TestAWorkerThatOnlyHostsToolsIsHandedNothing() {
	// It registered to run functions for sessions opened elsewhere and has no handler for
	// either kind of work, which it says by naming no kinds at all.
	tools, _ := s.pool.Register("acme", Registration{Capacity: 4, Handles: []Kind{}})

	_, callErr := s.pool.Assign("acme", Call{CallID: "call-1"})
	_, messageErr := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "hello"})

	s.Require().Error(callErr)
	s.Require().Error(messageErr)
	s.Empty(s.received(tools))
	s.Empty(s.answered(tools))
}

func (s *PoolSuite) TestWhatAnOlderWorkerLastSaidStandsInForWhatItIsHolding() {
	// It never reports work finished, so this is the only number it gives. Taking turns
	// while it says it is nearly full would pile work onto it.
	older, _ := s.waiting("acme", 10)
	newer, _ := s.reporting("acme", 10)
	older.Report(Load{ActiveAgents: 9})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(newer.ID, assigned.ID)
}

func (s *PoolSuite) TestAnOlderWorkerThatHasNotSaidRecentlyIsTakenToBeIdle() {
	// Reading it as busy would leave a worker that stopped reporting idle for good, which
	// is a worse answer than giving it a call it may be able to take.
	older, _ := s.waiting("acme", 10)
	s.reporting("acme", 10)
	older.Report(Load{ActiveAgents: 9, At: time.Now().UTC().Add(-time.Hour)})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(older.ID, assigned.ID)
}

func (s *PoolSuite) TestHowMuchAWorkerIsHoldingIsReadable() {
	worker, _ := s.pool.Register("acme", Registration{Capacity: 4, Active: 1, Tracking: true})
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	s.Equal(2, worker.Working())

	worker.Done(s.handed(worker).WorkID)

	s.Equal(1, worker.Working(), "what it carried in is still being answered")
}

func (s *PoolSuite) TestTheWorkersWaitingForACustomerAreReportedInRotationOrder() {
	first, _ := s.waiting("acme", 10)
	second, _ := s.waiting("acme", 10)
	s.waiting("globex", 10)

	waiting := s.pool.Workers("acme")

	s.Require().Len(waiting, 2)
	s.Equal(first.ID, waiting[0].ID)
	s.Equal(second.ID, waiting[1].ID)
}
