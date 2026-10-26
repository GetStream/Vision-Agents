package stream

import (
	"context"
	"errors"
	"sync"
)

// errNoSocket is why a running call could not reach its session: the socket it came on is
// gone.
var errNoSocket = errors.New("stream: the socket the call came on is closed")

// toolLine is how a call a socket delivered reaches the session it runs for: send writes a
// frame on that socket, and holds are the calls waiting to hear whether they were held.
type toolLine struct {
	send  func(Frame) error
	holds *holds
}

func (l toolLine) Update(_ context.Context, id, text string, verbatim bool) error {
	return l.send(Frame{"type": "tool_update", "tool_call_id": id, "text": text, "verbatim": verbatim})
}

func (l toolLine) Hold(ctx context.Context, id string) (bool, error) {
	return l.holds.ask(ctx, id, func() error { return l.send(Frame{"type": "tool_hold", "tool_call_id": id}) })
}

// holds are the calls that asked to be held and wait on the router's tool_held or
// tool_cancel.
type holds struct {
	mu      sync.Mutex
	waiting map[string]chan bool
}

// ask sends the hold and waits for the router to say whether the call was.
func (h *holds) ask(ctx context.Context, id string, send func() error) (bool, error) {
	answer := make(chan bool, 1)
	h.mu.Lock()
	if h.waiting == nil {
		h.waiting = map[string]chan bool{}
	}
	h.waiting[id] = answer
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		delete(h.waiting, id)
		h.mu.Unlock()
	}()

	if err := send(); err != nil {
		return false, err
	}
	select {
	case held := <-answer:
		return held, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// answer hands the router's answer to the call waiting on it, if one is.
func (h *holds) answer(id string, held bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if waiting, ok := h.waiting[id]; ok {
		select {
		case waiting <- held:
		default:
		}
	}
}
