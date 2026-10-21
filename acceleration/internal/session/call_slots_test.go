package session

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// CallSlotsSuite is the cap on a connection's connector calls in flight in one process, and
// the memory it keeps: none for a connection with no call waiting or in flight.
type CallSlotsSuite struct {
	suite.Suite
	slots callSlots
}

func TestCallSlotsSuite(t *testing.T) {
	suite.Run(t, new(CallSlotsSuite))
}

func (s *CallSlotsSuite) SetupTest() {
	s.slots = callSlots{}
}

func (s *CallSlotsSuite) TestNoLimitHoldsNothing() {
	for range 3 {
		_, err := s.slots.take(s.T().Context(), "conn", 0)
		s.Require().NoError(err)
	}

	s.Empty(s.slots.byConnection)
}

func (s *CallSlotsSuite) TestAConnectionIsForgottenAfterItsLastCallEnds() {
	first, err := s.slots.take(s.T().Context(), "conn", 2)
	s.Require().NoError(err)
	second, err := s.slots.take(s.T().Context(), "conn", 2)
	s.Require().NoError(err)

	first()
	s.Len(s.slots.byConnection, 1, "one call is still in flight")
	second()

	s.Empty(s.slots.byConnection)
}

func (s *CallSlotsSuite) TestACallThatStopsWaitingIsCountedOut() {
	held, err := s.slots.take(s.T().Context(), "conn", 1)
	s.Require().NoError(err)
	waiting, cancel := context.WithTimeout(s.T().Context(), 20*time.Millisecond)
	defer cancel()

	_, err = s.slots.take(waiting, "conn", 1)
	s.Require().ErrorIs(err, context.DeadlineExceeded)
	held()

	s.Empty(s.slots.byConnection)
}
