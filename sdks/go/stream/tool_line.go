package stream

import (
	"context"
	"errors"
	"sync"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// errNoSocket is why a running call could not reach its session: the socket it came on is
// gone.
var errNoSocket = errors.New("stream: the socket the call came on is closed")

// toolFailed is all the model is told of a failure that was not written for it.
const toolFailed = "the tool failed"

// toolDeclined is what the model is told of a call the person asked about said no to.
const toolDeclined = "The person declined this call."

// failureSaid is what the model is told about a call that failed: a tools.UserError's
// message, which was written for it, and otherwise only that the tool failed. Whatever else
// a function returns may name a host, an account or a query, so it is logged here instead.
func failureSaid(err error) string {
	var said tools.UserError
	if errors.As(err, &said) {
		return string(said)
	}
	return toolFailed
}

// asksFirst is the question a function has a person answer before it runs
// (tools.NeedsApproval), or nothing for one that asks none.
func asksFirst(functions *tools.Registry, name string) string {
	for _, function := range functions.List() {
		if function.Name == name {
			return function.Approval
		}
	}
	return ""
}

// heldCall is a call waiting for a person's answer: its frame, and what runs it.
type heldCall struct {
	frame Frame
	run   func()
}

// approvals are the calls of functions that ask first, held until a person answers through
// Pipeline.AnswerApproval. They last as long as the session.
type approvals struct {
	mu      sync.Mutex
	waiting map[string]heldCall
	// allowed are the calls a person said yes to, so one the router sends again after a
	// reconnect joins the run it started instead of asking them twice.
	allowed map[string]bool
}

// hold keeps a call until it is answered, and reports whether it was: a call a person
// already allowed is not held again.
func (a *approvals) hold(id string, call heldCall) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.allowed[id] {
		return false
	}
	if a.waiting == nil {
		a.waiting = map[string]heldCall{}
	}
	a.waiting[id] = call
	return true
}

// asking is the call waiting under id, if one is.
func (a *approvals) asking(id string) (heldCall, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	call, waiting := a.waiting[id]
	return call, waiting
}

// settle stops the call under id waiting, and reports whether it was: only the first of two
// answers finds it. Allowed, it is remembered as such.
func (a *approvals) settle(id string, allowed bool) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, waiting := a.waiting[id]; !waiting {
		return false
	}
	delete(a.waiting, id)
	if allowed {
		if a.allowed == nil {
			a.allowed = map[string]bool{}
		}
		a.allowed[id] = true
	}
	return true
}

// clear forgets every call, once the session they belong to is over.
func (a *approvals) clear() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.waiting, a.allowed = nil, nil
}

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
