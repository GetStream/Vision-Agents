package session

import (
	"context"
	"sync"
)

// callSlots caps the connector calls in flight at once on each connection, in this process,
// as the connection's manifest says (rate_limit.max_concurrent_calls). The zero value is ready
// to use, and a nil one caps nothing.
//
// Per connection, so one person's or one app's runaway loop never takes another's share: a
// session's tool calls run at once (agent.go starts one goroutine per harness.ToolRequested).
// Per process, so a call waits on no Redis: N routers send at most N times the cap, and a
// provider's own limit across every caller is the provider's to keep.
type callSlots struct {
	mu sync.Mutex
	// byConnection holds the slots of each connection with a call waiting or in flight, so a
	// connection no call uses holds no memory.
	byConnection map[string]*slots
}

// slots are one connection's: held has a value per call in flight, and users counts the calls
// waiting or in flight.
type slots struct {
	held  chan struct{}
	users int
}

// take waits until fewer than limit calls on connection are in flight, or ctx ends, and
// returns what ends this call's turn. A limit of 0 waits for nothing. The limit of the first
// call stands until every call on the connection has ended: a connection's calls all read the
// one revision it pinned (dispatcher.recheck refuses a call when it moved).
func (c *callSlots) take(ctx context.Context, connection string, limit int) (func(), error) {
	if c == nil || limit <= 0 {
		return func() {}, nil
	}
	c.mu.Lock()
	if c.byConnection == nil {
		c.byConnection = map[string]*slots{}
	}
	s := c.byConnection[connection]
	if s == nil {
		s = &slots{held: make(chan struct{}, limit)}
		c.byConnection[connection] = s
	}
	s.users++
	c.mu.Unlock()
	select {
	case s.held <- struct{}{}:
		return func() {
			<-s.held
			c.leave(connection, s)
		}, nil
	case <-ctx.Done():
		c.leave(connection, s)
		return nil, ctx.Err()
	}
}

// leave counts one call on connection out, and forgets the connection's slots after its last.
func (c *callSlots) leave(connection string, s *slots) {
	c.mu.Lock()
	defer c.mu.Unlock()
	s.users--
	if s.users == 0 {
		delete(c.byConnection, connection)
	}
}
