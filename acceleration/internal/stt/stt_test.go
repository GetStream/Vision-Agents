package stt

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"
)

type STTSuite struct {
	suite.Suite
}

func TestSTTSuite(t *testing.T) {
	suite.Run(t, new(STTSuite))
}

func (s *STTSuite) TestOnlyFinalModeIsFinal() {
	s.True(Transcript{Mode: ModeFinal}.Final())
	s.False(Transcript{Mode: ModeReplacement}.Final())
	s.False(Transcript{Mode: ModeDelta}.Final())
}

func (s *STTSuite) TestEmitterDeliversEventsInOrder() {
	e := NewEmitter(4)
	defer e.Close()

	e.Send(Transcript{Text: "one", Mode: ModeReplacement})
	e.Send(Transcript{Text: "two", Mode: ModeFinal})

	first, ok := (<-e.Events()).(Transcript)
	s.Require().True(ok)
	s.Equal("one", first.Text)

	second, ok := (<-e.Events()).(Transcript)
	s.Require().True(ok)
	s.Equal("two", second.Text)
	s.True(second.Final())
}

func (s *STTSuite) TestEmitterCloseEndsConsumption() {
	e := NewEmitter(1)
	e.Send(Connected{Provider: "test"})
	e.Close()

	// Buffered events still drain, then the channel reports closed.
	_, ok := <-e.Events()
	s.True(ok)
	_, ok = <-e.Events()
	s.False(ok)
}

func (s *STTSuite) TestEmitterCloseIsIdempotent() {
	e := NewEmitter(1)
	e.Close()
	s.NotPanics(e.Close)
}

func (s *STTSuite) TestSendAfterCloseDoesNotPanic() {
	e := NewEmitter(1)
	e.Close()
	s.NotPanics(func() { e.Send(Connected{Provider: "test"}) })
}

func (s *STTSuite) TestClosingWhileProvidersAreEmittingIsSafe() {
	// A provider emits from its own goroutine while the owner closes the session, so
	// Close must not race the sends onto the channel.
	e := NewEmitter(1)

	var senders sync.WaitGroup
	for range 8 {
		senders.Add(1)
		go func() {
			defer senders.Done()
			for range 100 {
				e.Send(Connected{Provider: "test"})
			}
		}()
	}

	drained := make(chan struct{})
	go func() {
		defer close(drained)
		for range e.Events() {
		}
	}()

	e.Close()
	s.NotPanics(senders.Wait)
	<-drained
}
