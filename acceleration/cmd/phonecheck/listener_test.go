package main

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

type ListenerSuite struct {
	suite.Suite
	l *sdkListener
}

func TestListenerSuite(t *testing.T) { suite.Run(t, new(ListenerSuite)) }

func (s *ListenerSuite) SetupTest() {
	s.l = &sdkListener{cancel: func() {}, arrived: make(chan stream.InboundCall, 1), ended: make(chan struct{})}
}

func (s *ListenerSuite) TestNoCallWithinTheWaitIsNoArrival() {
	_, err := s.l.arrival(context.Background(), 20*time.Millisecond)

	s.ErrorIs(err, errNoCallArrived)
}

func (s *ListenerSuite) TestAnArrivedCallIsTaken() {
	s.l.arrived <- stream.InboundCall{CalledNumber: agentNumber}

	call, err := s.l.arrival(context.Background(), time.Hour)

	s.Require().NoError(err)
	s.Equal(agentNumber, call.CalledNumber)
}

func (s *ListenerSuite) TestAStoppedRunIsNotANoArrival() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := s.l.arrival(ctx, time.Hour)

	s.ErrorIs(err, context.Canceled)
}

func (s *ListenerSuite) TestStopGivesUpOnAWorkerThatDoesNotLeave() {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	err := s.l.Stop(ctx)

	s.ErrorContains(err, "the worker did not stop")
}

func (s *ListenerSuite) TestStopReturnsOnceTheWorkerLeft() {
	close(s.l.ended)

	s.NoError(s.l.Stop(context.Background()))
}
