package session

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

// neverGrace is how long past waitFor a check of never may run. A check reads a store or a
// fake, which takes milliseconds, or seconds under load; a read stuck in I/O wait lasted
// 1m50s once. 30s is past the first and well short of go test's 10m timeout.
const neverGrace = 30 * time.Second

// never fails t if condition holds at any of its checks, made every tick for waitFor. A check
// that outlives waitFor+neverGrace fails t, so a stuck check fails its test, not the binary.
func never(t assert.TestingT, condition func() bool, waitFor, tick time.Duration, msgAndArgs ...any) bool {
	if h, ok := t.(interface{ Helper() }); ok {
		h.Helper()
	}
	return neverWithin(t, condition, waitFor, tick, neverGrace, msgAndArgs...)
}

// neverWithin is never with its grace given. Each check runs on a goroutine that the test
// waits for, and waitFor returns only after the check running at its end has finished, so a
// check that reads a suite's fields cannot race the next test's setup rewriting them: testify's
// Never returns at waitFor and leaves its check running.
func neverWithin(t assert.TestingT, condition func() bool, waitFor, tick, grace time.Duration, msgAndArgs ...any) bool {
	if h, ok := t.(interface{ Helper() }); ok {
		h.Helper()
	}
	window := time.NewTimer(waitFor)
	defer window.Stop()
	limit := time.NewTimer(waitFor + grace)
	defer limit.Stop()
	ticker := time.NewTicker(tick)
	defer ticker.Stop()
	for {
		held, finished := make(chan bool, 1), make(chan struct{})
		go func() {
			defer close(finished)
			held <- condition()
		}()
		select {
		case <-finished:
			select {
			case holds := <-held:
				if holds {
					return assert.Fail(t, "Condition satisfied", msgAndArgs...)
				}
			default:
				// The check left through FailNow, which has failed the test itself.
				return false
			}
		case <-limit.C:
			return assert.Fail(t, fmt.Sprintf("a check was still running %s after the %s it was to run for", grace, waitFor), msgAndArgs...)
		}
		select {
		case <-window.C:
			return true
		case <-ticker.C:
		}
	}
}

type NeverSuite struct {
	suite.Suite
}

func TestNeverSuite(t *testing.T) {
	suite.Run(t, new(NeverSuite))
}

// failures is a TestingT that keeps what it was failed with.
type failures struct{ messages []string }

func (f *failures) Errorf(format string, args ...any) {
	f.messages = append(f.messages, fmt.Sprintf(format, args...))
}

// A check that outlasts waitFor within the grace has finished by the time never returns, so nothing it reads
// is read after its test.
func (s *NeverSuite) TestACheckStillRunningAtTheDeadlineHasFinishedWhenItReturns() {
	var running atomic.Bool
	held := never(s.T(), func() bool {
		running.Store(true)
		time.Sleep(50 * time.Millisecond)
		running.Store(false)
		return false
	}, 10*time.Millisecond, time.Millisecond)

	s.True(held)
	s.False(running.Load(), "a check was still running")
}

// A condition that holds only at a later check fails the test with its message.
func (s *NeverSuite) TestAConditionThatHoldsAtALaterCheckFails() {
	var checks atomic.Int32
	t := &failures{}
	held := never(t, func() bool { return checks.Add(1) == 3 }, time.Second, time.Millisecond, "it changed")

	s.False(held)
	s.Equal(int32(3), checks.Load())
	s.Require().Len(t.messages, 1)
	s.Contains(t.messages[0], "Condition satisfied")
	s.Contains(t.messages[0], "it changed")
}

// A condition that comes true half way through waitFor fails the test, and it is checked once
// per tick: a shortened window passes it, and a loop that ignores the tick checks it far more
// often than waitFor/tick+1 times.
func (s *NeverSuite) TestTheWindowIsWaitForAndTheCheckIsMadeOncePerTick() {
	const waitFor, tick = 400 * time.Millisecond, 20 * time.Millisecond
	start := time.Now()
	var checks atomic.Int32
	t := &failures{}
	held := never(t, func() bool {
		checks.Add(1)
		return time.Since(start) >= waitFor/2
	}, waitFor, tick)

	s.False(held, "a condition true half way through the window went unseen")
	s.LessOrEqual(checks.Load(), int32(waitFor/tick+1))
}

// A check that blocks past the grace fails its test then, with the message, rather than holding
// the run until go test's timeout. The check is let go when the test ends.
func (s *NeverSuite) TestACheckStuckPastTheGraceFailsItsTest() {
	release := make(chan struct{})
	s.T().Cleanup(func() { close(release) })
	start := time.Now()
	t := &failures{}
	held := neverWithin(t, func() bool {
		<-release
		return false
	}, 10*time.Millisecond, time.Millisecond, 100*time.Millisecond, "it is stuck")

	s.False(held)
	s.Less(time.Since(start), 5*time.Second, "never waited on the stuck check")
	s.Require().Len(t.messages, 1)
	s.Contains(t.messages[0], "a check was still running 100ms after the 10ms")
	s.Contains(t.messages[0], "it is stuck")
}

// A check that leaves through FailNow has failed its test itself: never returns at once, rather
// than waiting out the grace for an answer that will not come.
func (s *NeverSuite) TestACheckThatLeavesThroughFailNowReturnsAtOnce() {
	start := time.Now()
	t := &failures{}
	held := neverWithin(t, func() bool {
		runtime.Goexit()
		return false
	}, time.Second, time.Millisecond, 10*time.Second)

	s.False(held)
	s.Less(time.Since(start), 2*time.Second)
	s.Empty(t.messages)
}
