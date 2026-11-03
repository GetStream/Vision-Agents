package session

import (
	"context"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
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
	if h.session.holdHangUp(h.later) {
		return &agent.EndDeferred{Reason: machineChecking}
	}
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

// later carries out a hang-up held while the call was checked for an answering machine. The
// agent's own context is long gone by then.
func (h *hangUp) later() {
	if h.session.closingNow() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), hangUpTimeout)
	defer cancel()
	if err := h.EndCall(ctx); err != nil {
		h.manager.logger.Warn("could not hang up once the call was checked for an answering machine",
			"session", h.session.id, "error", err)
	}
}

// machineChecking is what the model is told of a hang-up held for the answering machine check.
const machineChecking = "The call is still being checked for an answering machine, and ends by itself " +
	"once that is done. Say nothing more. If the caller speaks before then, it stays open: " +
	"answer them, and call end_call again once they are done."

// machineWait bounds how long the agent's own hang-up is held on a placed call that leaves a
// voicemail, from when its phone leg answered. A vendor's report on a mobile comes about 30 s
// after the answer. A variable so tests can shorten it.
var machineWait = 40 * time.Second

// machineCheck holds the agent's own hang-up on a placed call that leaves a voicemail until
// the vendor says what picked up. A recording says nothing the agent takes as meant for it,
// and an agent that hangs up on the silence is gone before the report, leaving no message.
type machineCheck struct {
	mu sync.Mutex
	// over is set once the report came, the caller spoke to the agent or machineWait passed.
	over bool
	// held is the hang-up the agent asked for meanwhile.
	held  func()
	timer *time.Timer
}

// holdHangUp keeps hangUp for when the check is over, reporting whether it did: only on a
// call that leaves a voicemail and is still being checked.
func (s *Session) holdHangUp(hangUp func()) bool {
	if s.Spec().Voicemail == nil {
		return false
	}
	s.machine.mu.Lock()
	defer s.machine.mu.Unlock()
	if s.machine.over {
		return false
	}
	s.machine.held = hangUp
	return true
}

// answered starts the bound on the check once the phone leg picked up.
func (s *Session) answered() {
	s.machine.mu.Lock()
	defer s.machine.mu.Unlock()
	if s.machine.over || s.machine.timer != nil {
		return
	}
	s.machine.timer = time.AfterFunc(machineWait, func() { s.checked(true) })
}

// checked ends the check. With carryOut a hang-up held meanwhile goes through; without, a
// machine is being left a message or the caller spoke, and the call carries on.
func (s *Session) checked(carryOut bool) {
	s.machine.mu.Lock()
	if s.machine.over {
		s.machine.mu.Unlock()
		return
	}
	s.machine.over = true
	held := s.machine.held
	s.machine.held = nil
	if s.machine.timer != nil {
		s.machine.timer.Stop()
	}
	s.machine.mu.Unlock()
	// Not on s.running: the hang-up closes the session, and closing waits on s.running.
	// It returns at once on a session that began closing meanwhile.
	if carryOut && held != nil {
		go held()
	}
}

// heardCaller ends the check once the phone caller said something the agent took as meant
// for it, which no recording does.
func (s *Session) heardCaller(event Event) {
	if heard, ok := event.(agent.Heard); ok {
		if _, onPhone := phone.CallerNumber(heard.Participant.UserID); onPhone {
			s.checked(false)
		}
	}
}
