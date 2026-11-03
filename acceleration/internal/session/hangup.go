package session

import (
	"context"
	"sync"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// hangUp is how the agent on a session hangs up (agent.Ender): it ends the call for
// everybody on it, when the call is the session's to end, then closes the session.
type hangUp struct {
	manager *Manager
	session *Session
	// end ends the call at Stream. Nil when the agent only leaves: for a call the caller
	// named, where other people may still be on it, and for the session's own call with no
	// Stream client to end it.
	end func(ctx context.Context) error

	mu   sync.Mutex
	done bool
}

// ender is what the agent built for spec hangs up with, nil when it may not: in writing, or
// when the session's config does not turn end_call on.
func (m *Manager) ender(created *Session, spec Spec, stream streamapp.Bound) agent.Ender {
	if spec.Text || !spec.endsCalls() {
		return nil
	}
	hang := &hangUp{manager: m, session: created}
	switch {
	// Stream's SIP bridge holds the phone leg, and ending the call drops it on every
	// vendor and trunk.
	case spec.Phone != nil && spec.Phone.Number != "":
		hang.end = func(ctx context.Context) error {
			return m.options.Phone.EndCall(ctx, spec.CustomerID, spec.StreamApp, spec.CallType, spec.CallID)
		}
	case spec.ownCall && stream.Client != nil:
		hang.end = func(ctx context.Context) error {
			_, err := stream.Client.Video().EndCall(ctx, spec.CallType, spec.CallID, &getstream.EndCallRequest{})
			return stack.Wrap(err)
		}
	}
	return hang
}

// EndCall ends the call and the session once. A call that could not be ended is not
// marked done, so the model may try again.
func (h *hangUp) EndCall(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.done {
		return nil
	}
	if h.end != nil {
		if err := h.end(ctx); err != nil {
			return err
		}
	}
	h.done = true
	h.manager.hangUp(h.session, EndAgentHangup)
	return nil
}
