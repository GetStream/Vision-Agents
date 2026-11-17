// Package emit fans provider events out to a single consumer channel.
//
// Providers hold an Emitter rather than managing a channel and its close semantics
// themselves, which is the same problem for every modality: events arrive on the
// provider's own read goroutine while the owner may close the session at any moment.
package emit

import (
	"context"
	"sync"
)

// Emitter delivers events of one type to a single consumer.
//
// The read lock is what makes closing the channel safe: Close cannot run while a Send is
// in flight, and a Send after Close is dropped instead of panicking.
type Emitter[E any] struct {
	ch     chan E
	ctx    context.Context
	cancel context.CancelFunc

	mu     sync.RWMutex
	closed bool
}

// New returns an Emitter with the given channel buffer.
func New[E any](buffer int) *Emitter[E] {
	ctx, cancel := context.WithCancel(context.Background())
	return &Emitter[E]{ch: make(chan E, buffer), ctx: ctx, cancel: cancel}
}

// Send blocks until the consumer takes the event or the Emitter is closed. Events are
// too valuable to drop on a slow reader, so this deliberately applies backpressure.
func (e *Emitter[E]) Send(event E) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return
	}

	select {
	case e.ch <- event:
	case <-e.ctx.Done():
	}
}

// Events returns the consumer channel. It is closed by Close, and any events already
// buffered can still be drained.
func (e *Emitter[E]) Events() <-chan E { return e.ch }

// Close stops further sends and closes the channel exactly once.
func (e *Emitter[E]) Close() {
	// Cancel first so a Send blocked on a full channel lets go of the read lock.
	e.cancel()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	close(e.ch)
}
