package sipbridge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

var errCallEnded = errors.New("call ended")

// errNotEstablished refuses a request that arrives while Dial is still setting the call up:
// the other leg may not have a dialog yet to pass it on in.
var errNotEstablished = errors.New("call not set up yet")

// Call is an established outbound call: two dialogs, and whatever one side sends in its
// dialog is passed on to the other.
type Call struct {
	legs [2]leg
	log  *slog.Logger

	established atomic.Bool
	ending      atomic.Bool
	done        chan struct{}
	endOnce     sync.Once
	err         error
}

func newCall(customer, stream leg, log *slog.Logger) *Call {
	return &Call{legs: [2]leg{customer, stream}, log: log, done: make(chan struct{})}
}

// Done is closed when the call has ended.
func (c *Call) Done() <-chan struct{} { return c.done }

// Err says why the call ended, nil for a normal hangup. Read it after Done is closed.
func (c *Call) Err() error { return c.err }

// Hangup sends BYE on both legs at once, each bounded by its own cleanup timeout, so a
// customer trunk that does not answer cannot hold up the BYE to Stream, and ends the call.
func (c *Call) Hangup(ctx context.Context) error {
	if !c.claim() {
		return nil
	}
	var errs [2]error
	var wg sync.WaitGroup
	for s, l := range c.legs {
		wg.Go(func() {
			byeCtx, cancel := context.WithTimeout(ctx, cleanupTimeout)
			defer cancel()
			errs[s] = l.Bye(byeCtx)
		})
	}
	wg.Wait()
	c.end(nil)
	return errors.Join(errs[:]...)
}

// establish lets requests from one side through to the other. Dial calls it once both
// legs are up.
func (c *Call) establish() { c.established.Store(true) }

func (c *Call) claim() bool {
	return c.ending.CompareAndSwap(false, true)
}

func (c *Call) ended() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

func (c *Call) end(err error) {
	c.endOnce.Do(func() {
		c.err = err
		close(c.done)
	})
}

func (c *Call) other(s side) leg { return c.legs[1-s] }

func (c *Call) onBye(ctx context.Context, from side) {
	if !c.claim() {
		return
	}
	c.log.Info("BYE received, hanging up the other leg", "from", from)
	if err := c.other(from).Bye(ctx); err != nil {
		c.log.Warn("BYE to the other leg failed", "leg", 1-from, "error", err)
	}
	c.end(nil)
}

func (c *Call) onReinvite(ctx context.Context, from side, offer []byte) ([]byte, error) {
	if c.ending.Load() {
		return nil, stack.Wrap(errCallEnded)
	}
	if !c.established.Load() {
		return nil, stack.Wrap(errNotEstablished)
	}
	c.log.Info("re-INVITE received, passing it on", "from", from)
	return c.other(from).Reinvite(ctx, offer)
}

func (c *Call) onInfo(ctx context.Context, from side, contentType string, body []byte) error {
	if c.ending.Load() {
		return stack.Wrap(errCallEnded)
	}
	if !c.established.Load() {
		return stack.Wrap(errNotEstablished)
	}
	c.log.Info("INFO received, passing it on", "from", from, "content_type", contentType)
	return c.other(from).Info(ctx, contentType, body)
}

// lost ends the call when one leg can no longer be reached, so the other is not left open.
func (c *Call) lost(ctx context.Context, s side, err error) {
	if !c.claim() {
		return
	}
	c.log.Warn("leg lost, hanging up the other", "leg", s, "error", err)
	if byeErr := c.other(s).Bye(ctx); byeErr != nil {
		c.log.Warn("BYE to the other leg failed", "leg", 1-s, "error", byeErr)
	}
	c.end(stack.Wrap(fmt.Errorf("%s leg lost: %w", s, err)))
}
