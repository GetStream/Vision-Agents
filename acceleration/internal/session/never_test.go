package session

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

// never fails t if condition holds at any of its checks, made every tick for waitFor. It is
// testify's Never checked on the test's own goroutine: testify's runs each check on a goroutine
// and returns at waitFor without waiting for the one still running, so a check that reads a
// suite's fields would race the next test's setup rewriting them.
func never(t assert.TestingT, condition func() bool, waitFor, tick time.Duration, msgAndArgs ...any) bool {
	if h, ok := t.(interface{ Helper() }); ok {
		h.Helper()
	}
	timer := time.NewTimer(waitFor)
	defer timer.Stop()
	ticker := time.NewTicker(tick)
	defer ticker.Stop()
	for {
		if condition() {
			return assert.Fail(t, "Condition satisfied", msgAndArgs...)
		}
		select {
		case <-timer.C:
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

// A check that outlasts waitFor has finished by the time never returns, so nothing it reads
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
