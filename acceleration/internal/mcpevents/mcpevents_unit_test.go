package mcpevents

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// UnitSuite is what is decided before any row or request: keys, refresh times, sealing
// bounds and the options New refuses. The whole path is internal/api's ConnectionEventsSuite.
type UnitSuite struct {
	suite.Suite
	now time.Time
}

func TestUnitSuite(t *testing.T) {
	suite.Run(t, new(UnitSuite))
}

func (s *UnitSuite) SetupTest() {
	s.now = time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)
}

func (s *UnitSuite) TestTheSameFiltersInAnotherOrderAreOneKey() {
	first := Key("issue.created", map[string]any{"project": "web", "level": "error"})
	second := Key("issue.created", map[string]any{"level": "error", "project": "web"})

	s.Equal(first, second)
}

func (s *UnitSuite) TestNoFiltersAndEmptyFiltersAreOneKey() {
	s.Equal(Key("issue.created", nil), Key("issue.created", map[string]any{}))
}

func (s *UnitSuite) TestAnotherEventOrFilterIsAnotherKey() {
	key := Key("issue.created", map[string]any{"project": "web"})

	s.NotEqual(key, Key("issue.closed", map[string]any{"project": "web"}))
	s.NotEqual(key, Key("issue.created", map[string]any{"project": "api"}))
}

func (s *UnitSuite) TestAGrantIsAskedForAgainTenMinutesBeforeItExpires() {
	expires := s.now.Add(time.Hour)

	next := refreshAt(s.now, &expires)

	s.Require().NotNil(next)
	s.Equal(expires.Add(-10*time.Minute), *next)
}

func (s *UnitSuite) TestAGrantShorterThanTenMinutesIsAskedForAgainHalfwayThere() {
	expires := s.now.Add(4 * time.Minute)

	next := refreshAt(s.now, &expires)

	s.Require().NotNil(next)
	s.Equal(s.now.Add(2*time.Minute), *next)
}

func (s *UnitSuite) TestAGrantThatDoesNotExpireIsNeverAskedForAgain() {
	s.Nil(refreshAt(s.now, nil))
}

// TestARefusedSubscriptionWaitsTwiceAsLongEachTimeUpToADay: 15 min, 30 min, 1 h ... and a
// day at most, so a server that offers no events is not asked every 15 minutes forever.
func (s *UnitSuite) TestARefusedSubscriptionWaitsTwiceAsLongEachTimeUpToADay() {
	s.Equal(15*time.Minute, retryWait(1))
	s.Equal(30*time.Minute, retryWait(2))
	s.Equal(time.Hour, retryWait(3))
	s.Equal(16*time.Hour, retryWait(7))
	s.Equal(24*time.Hour, retryWait(8))
	s.Equal(24*time.Hour, retryWait(1000))
}

// TestAWaitOnARenewalEndsALeaseBeforeTheGrantDoes: a grant ending in 5 minutes is asked for
// again a lease before then, not 15 minutes from now.
func (s *UnitSuite) TestAWaitOnARenewalEndsALeaseBeforeTheGrantDoes() {
	ends := s.now.Add(5 * time.Minute)

	s.Equal(ends.Add(-time.Minute), waitUntil(s.now, &ends, time.Minute))
}

func (s *UnitSuite) TestAWaitOnARenewalWithAGrantFarAwayIsFifteenMinutes() {
	ends := s.now.Add(time.Hour)

	s.Equal(s.now.Add(15*time.Minute), waitUntil(s.now, &ends, time.Minute))
}

func (s *UnitSuite) TestAWaitOnARenewalIsNeverShorterThanALease() {
	ends := s.now.Add(30 * time.Second)

	s.Equal(s.now.Add(time.Minute), waitUntil(s.now, &ends, time.Minute))
}

func (s *UnitSuite) TestAWaitOnARenewalWithNoGrantLeftOrNoneThatEndsIsFifteenMinutes() {
	ended := s.now.Add(-time.Minute)

	s.Equal(s.now.Add(15*time.Minute), waitUntil(s.now, &ended, time.Minute))
	s.Equal(s.now.Add(15*time.Minute), waitUntil(s.now, nil, time.Minute))
}

// TestASecretIsBoundToItsOwnSubscription: no two subscriptions share additional data, so a
// sealed secret copied onto another row does not open there, even when the parts run into
// each other.
func (s *UnitSuite) TestASecretIsBoundToItsOwnSubscription() {
	s.NotEqual(secretAAD("app", "conn-1", "token"), secretAAD("app", "conn-1", "token2"))
	s.NotEqual(secretAAD("app", "conn", "1token"), secretAAD("app", "conn1", "token"))
	s.NotEqual(secretAAD("app", "conn-1", "token"), secretAAD("app2", "conn-1", "token"))
}

func (s *UnitSuite) TestNewRefusesMissingOptions() {
	_, err := New(Options{})

	s.ErrorContains(err, "a store, sessions, transports and a sealer are required")
}
