package dispatch

import (
	"errors"
	"testing"

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
	first, _ := s.pool.Register("acme", 10)
	second, _ := s.pool.Register("acme", 10)

	for _, id := range []string{"call-1", "call-2", "call-3", "call-4"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1", "call-3"}, s.received(first))
	s.Equal([]string{"call-2", "call-4"}, s.received(second))
}

func (s *PoolSuite) TestAWorkerThatLeftIsNotOfferedCalls() {
	first, _ := s.pool.Register("acme", 10)
	second, release := s.pool.Register("acme", 10)

	release()
	for _, id := range []string{"call-1", "call-2"} {
		_, err := s.pool.Assign("acme", Call{CallID: id})
		s.Require().NoError(err)
	}

	s.Equal([]string{"call-1", "call-2"}, s.received(first))
	s.Empty(s.received(second), "a released worker's channel is closed, not written to")
}

func (s *PoolSuite) TestTheRotationSurvivesTheWorkerWhoseTurnItWasLeaving() {
	first, _ := s.pool.Register("acme", 10)
	_, release := s.pool.Register("acme", 10)

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
	full, _ := s.pool.Register("acme", 1)
	free, _ := s.pool.Register("acme", 5)

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
	s.pool.Register("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	_, err = s.pool.Assign("acme", Call{CallID: "call-2"})

	s.Require().Error(err)
	s.False(errors.Is(err, ErrNoWorkers))
	s.ErrorContains(err, "at capacity")
}

func (s *PoolSuite) TestOneCustomersCallsNeverReachAnothersWorkers() {
	ours, _ := s.pool.Register("acme", 10)
	theirs, _ := s.pool.Register("globex", 10)

	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	s.Equal([]string{"call-1"}, s.received(ours))
	s.Empty(s.received(theirs))

	_, err = s.pool.Assign("globex", Call{CallID: "call-2"})
	s.Require().NoError(err)
	s.Equal([]string{"call-2"}, s.received(theirs))
}

func (s *PoolSuite) TestACallHasToNameTheCallItIs() {
	s.pool.Register("acme", 10)

	_, err := s.pool.Assign("acme", Call{CalledNumber: "+15125551234"})

	s.ErrorContains(err, "needs an id")
}

func (s *PoolSuite) TestTwoWorkersSplitTheMessagesBetweenThem() {
	first, _ := s.pool.Register("acme", 10)
	second, _ := s.pool.Register("acme", 10)

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
	worker, _ := s.pool.Register("acme", 1)
	_, err := s.pool.Assign("acme", Call{CallID: "call-1"})
	s.Require().NoError(err)

	assigned, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-2", Text: "hello"})

	s.Require().NoError(err)
	s.Equal(worker.ID, assigned.ID)
	s.Equal([]string{"hello"}, s.answered(worker))
}

func (s *PoolSuite) TestAFullWorkerIsPassedOverForMessagesToo() {
	full, _ := s.pool.Register("acme", 1)
	free, _ := s.pool.Register("acme", 5)

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

func (s *PoolSuite) TestAMessageHasToNameTheChannelItWasWrittenIn() {
	// Answering anywhere else would be a reply nobody asked for in a conversation nobody
	// is reading.
	s.pool.Register("acme", 10)

	_, err := s.pool.AssignMessage("acme", Message{Text: "hello"})

	s.ErrorContains(err, "needs a channel")
}

func (s *PoolSuite) TestOneCustomersMessagesNeverReachAnothersWorkers() {
	ours, _ := s.pool.Register("acme", 10)
	theirs, _ := s.pool.Register("globex", 10)

	_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "ours"})
	s.Require().NoError(err)

	s.Equal([]string{"ours"}, s.answered(ours))
	s.Empty(s.answered(theirs))
}

func (s *PoolSuite) TestAWorkerThatLeftIsNotOfferedMessages() {
	first, _ := s.pool.Register("acme", 10)
	second, release := s.pool.Register("acme", 10)

	release()
	_, err := s.pool.AssignMessage("acme", Message{ChannelID: "call-1", Text: "hello"})
	s.Require().NoError(err)

	s.Equal([]string{"hello"}, s.answered(first))
	s.Empty(s.answered(second), "a released worker's channels are closed, not written to")
}

func (s *PoolSuite) TestAWorkersLoadIsWhatItLastReported() {
	worker, _ := s.pool.Register("acme", 10)

	worker.Report(Load{ActiveAgents: 3, CPUPercent: 41.5, MemoryPercent: 62.0, LatencyMs: 18.25})

	load := worker.Load()
	s.Equal(3, load.ActiveAgents)
	s.InDelta(41.5, load.CPUPercent, 0.001)
	s.InDelta(18.25, load.LatencyMs, 0.001)
	s.False(load.At.IsZero(), "a report with no time on it is stamped when it arrives")
}

func (s *PoolSuite) TestLoadDoesNotChangeWhoseTurnItIs() {
	busy, _ := s.pool.Register("acme", 10)
	idle, _ := s.pool.Register("acme", 10)
	busy.Report(Load{ActiveAgents: 99, CPUPercent: 99})
	idle.Report(Load{ActiveAgents: 0, CPUPercent: 1})

	assigned, err := s.pool.Assign("acme", Call{CallID: "call-1"})

	s.Require().NoError(err)
	s.Equal(busy.ID, assigned.ID, "round robin takes turns; it does not read load yet")
}

func (s *PoolSuite) TestTheWorkersWaitingForACustomerAreReportedInRotationOrder() {
	first, _ := s.pool.Register("acme", 10)
	second, _ := s.pool.Register("acme", 10)
	s.pool.Register("globex", 10)

	waiting := s.pool.Workers("acme")

	s.Require().Len(waiting, 2)
	s.Equal(first.ID, waiting[0].ID)
	s.Equal(second.ID, waiting[1].ID)
}
