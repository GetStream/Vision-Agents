package session

import (
	"os"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// testTimeout is a timeout short enough to wait out in a test.
const testTimeout = 50 * time.Millisecond

func (s *SessionSuite) timedOut(created *Session) {
	s.eventually(func() bool {
		return created.State() == Ended && !s.manager.Running(created.ID())
	}, "the session did not end at its timeout")
	_, found := s.manager.Get(created.ID(), OwnerOf(created.Spec()))
	s.False(found, "an ended session is still found")
	s.Zero(s.manager.Len(), "an ended session is still counted")
}

// outlives checks the session is still running several timeouts later.
func (s *SessionSuite) outlives(created *Session) {
	time.Sleep(3 * testTimeout)
	s.Equal(Live, created.State())
	s.True(s.manager.Running(created.ID()), "the session was taken off")
}

func (s *SessionSuite) TestASessionWithoutATimeoutKeepsRunning() {
	s.manages()
	created := s.joins(Spec{})

	s.outlives(created)
}

func (s *SessionSuite) TestASessionEndsWhenItsTimeoutIsReached() {
	s.manages()
	created := s.joins(Spec{})

	set := time.Now()
	created.SetTimeout(testTimeout)

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(set), testTimeout, "the session ended before its timeout")
	s.True(s.edges[0].gone(), "the agent is still in the call")
}

func (s *SessionSuite) TestSettingTheTimeoutAgainMovesTheMoment() {
	s.manages()
	created := s.joins(Spec{})

	created.SetTimeout(testTimeout)
	time.Sleep(testTimeout / 2)
	created.SetTimeout(4 * testTimeout)

	time.Sleep(2 * testTimeout)
	s.Equal(Live, created.State(), "the session ended at the timeout that was moved")
	s.timedOut(created)
}

func (s *SessionSuite) TestClearingTheTimeoutKeepsTheSessionRunning() {
	s.manages()
	created := s.joins(Spec{})

	created.SetTimeout(testTimeout)
	created.ClearTimeout()

	s.outlives(created)
}

func (s *SessionSuite) TestAZeroTimeoutEndsTheSessionNow() {
	s.manages()
	created := s.joins(Spec{})

	created.SetTimeout(0)

	s.timedOut(created)
}

func (s *SessionSuite) TestATimeoutEndsTheSessionWithAnInterrupt() {
	// Nobody is left to hear the end of the reply, so the session does not wait for it the
	// way a stop does (finishWithin).
	s.gated = newGatedLLM()
	s.manages()
	created := s.joins(Spec{})
	s.says(created, "hello")
	select {
	case <-s.gated.entered:
	case <-time.After(settleFor):
		s.FailNow("the model was never asked")
	}

	set := time.Now()
	created.SetTimeout(0)

	s.timedOut(created)
	s.Less(time.Since(set), finishWithin/2, "the session waited for the reply instead of interrupting it")
}

// holdsClaims gives the manager a directory when a Redis is at hand, so a test can read
// which sessions this node claims.
func (s *SessionSuite) holdsClaims() bool {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		return false
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	directory, err := node.NewDirectory(node.DirectoryOptions{
		Redis: client, Address: "node-a:1", Prefix: "timeout-test:" + time.Now().Format(time.RFC3339Nano),
	})
	s.Require().NoError(err)
	s.T().Cleanup(directory.Close)
	s.directory = directory
	return true
}

func (s *SessionSuite) TestAReopenedSessionIsNotEndedByTheOldTimeout() {
	// A reopened conversation takes the id of the session that ended, so an ending the old
	// one had already started must leave the new one, and its claim, alone.
	claims := s.holdsClaims()
	s.persists()
	old := s.commands()
	s.Require().NoError(old.Close())

	reopened, err := s.manager.Create(s.ctx, Spec{
		ID:                  old.ID(),
		CustomerID:          "acme",
		Text:                true,
		PersistConversation: true,
		ConversationID:      old.Spec().ConversationID,
		Reopened:            old.CreatedAt(),
		LLMTarget:           "en-low-latency",
		Caller:              routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = reopened.Close() })

	// The old session's timer fired just before it closed.
	old.end()

	s.Equal(Live, reopened.State())
	s.True(s.manager.Running(reopened.ID()), "the reopened session was taken off")
	if claims {
		address, err := s.directory.Node(s.ctx, reopened.ID())
		s.Require().NoError(err)
		s.Equal("node-a:1", address, "the reopened session's claim was released")
	}
}

func (s *SessionSuite) TestATimeoutDuringStopOrShutdownIsHarmless() {
	s.manages()
	for range 5 {
		created := s.joins(Spec{})
		var stopped sync.WaitGroup
		stopped.Go(func() { created.SetTimeout(0) })
		stopped.Go(func() {
			_, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
			s.NoError(err)
		})
		stopped.Wait()
		s.timedOut(created)
	}

	created := s.joins(Spec{})
	created.SetTimeout(0)
	s.Require().NoError(s.manager.Shutdown())
	s.Equal(Ended, created.State())
	s.Zero(s.manager.Len())
}

func (s *SessionSuite) TestTimeoutCallsOnAnEndedSessionDoNothing() {
	s.manages()
	created := s.joins(Spec{})
	ended := make(chan struct{}, 4)
	created.mu.Lock()
	created.end = func() { ended <- struct{}{} }
	created.mu.Unlock()

	// A leave reported while the session is still closing must not set a new timeout.
	edge := s.edges[0]
	edge.holdSpeech(true)
	closed := make(chan error, 1)
	go func() { closed <- created.Close() }()
	s.eventually(func() bool {
		created.mu.Lock()
		defer created.mu.Unlock()
		return created.closing
	}, "the session never started closing")
	created.SetTimeout(0)
	edge.holdSpeech(false)
	s.Require().NoError(<-closed)

	created.SetTimeout(0)
	created.ClearTimeout()

	select {
	case <-ended:
		s.Fail("a timeout set on an ending session ended it again")
	case <-time.After(3 * testTimeout):
	}
}
