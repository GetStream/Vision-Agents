package session

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// A phone session ends hangupTimeout after a phone caller has been and the call is left
// empty, and at a start timeout when the agent joins a call nobody is in yet: the outbound
// one counts from about when the call was placed, so it allows for the ringing. Variables so
// tests can shorten them.
var (
	hangupTimeout        = 10 * time.Second
	outboundStartTimeout = 90 * time.Second
	inboundStartTimeout  = 30 * time.Second
)

// presence is who is in the call one agent run is on. Its lock is taken before s.mu, which
// the timeout calls take, and never while holding s.mu.
type presence struct {
	mu sync.Mutex
	// people are keyed by connection id: two phone legs can share a user id, and a leg that
	// drops comes back with a new id.
	people map[string]struct{}
	// settled is set at Joined, once everyone already in the call has been reported: an
	// empty call before then means nothing.
	settled bool
	// retired is set once the session moved off this run, whose late leaves are stale.
	retired bool
	// pendingGreeting says the greeting when the first participant joins. It is nil after
	// the agent says the greeting, or when there is no greeting.
	pendingGreeting func()
}

// holdGreeting keeps the greeting until the first participant joins. Call it before the
// agent joins the call, so that the agent also greets a participant who is already in the
// call.
func (run *presence) holdGreeting(greet func()) {
	run.mu.Lock()
	defer run.mu.Unlock()
	run.pendingGreeting = greet
}

// sayGreeting returns a function that says the greeting in spec. The function logs an error
// and does not return it, because no caller waits for it.
func (s *Session) sayGreeting(ctx context.Context, held *agent.Agent, spec Spec) func() {
	return func() {
		if err := held.Greet(context.WithoutCancel(ctx), spec.Greeting, spec.VaryGreeting); err != nil {
			s.logger.Warn("could not greet the call", "session", s.id, "error", err)
		}
	}
}

func newPresence() *presence {
	return &presence{people: map[string]struct{}{}}
}

// track starts keeping who is in held's call. Nil for a run that cannot tell: one in
// writing, or on an edge that does not say who comes and goes.
func (s *Session) track(held *agent.Agent) *presence {
	if !held.Attends() {
		return nil
	}
	run := newPresence()
	s.presenceMu.Lock()
	defer s.presenceMu.Unlock()
	if s.runs == nil {
		s.runs = map[*agent.Agent]*presence{}
	}
	s.runs[held] = run
	return run
}

func (s *Session) untrack(held *agent.Agent) {
	s.presenceMu.Lock()
	defer s.presenceMu.Unlock()
	delete(s.runs, held)
}

// retire stops held's run from deciding the session's timeout. Called holding commandMu
// once the session has moved off it.
func (s *Session) retire(held *agent.Agent) {
	s.presenceMu.Lock()
	run := s.runs[held]
	s.presenceMu.Unlock()
	if run == nil {
		return
	}
	run.mu.Lock()
	run.retired = true
	run.mu.Unlock()
}

// phoneSession is whether the router keeps the session's timeout: it was created with a
// phone, or a phone caller has been in its call.
func (s *Session) phoneSession(spec Spec) bool {
	return spec.Phone != nil || s.callerSeen.Load()
}

// observe sets or clears the session's timeout from who comes and goes on one run's call.
func (s *Session) observe(run *presence, event Event) {
	if run == nil {
		return
	}
	switch event.(type) {
	case agent.ParticipantJoined, agent.ParticipantLeft, agent.Joined:
	default:
		return
	}
	spec := s.Spec()

	// Held through the timeout call, so a retire either comes first and drops this, or
	// comes after and the ClearTimeout that follows it wins.
	run.mu.Lock()
	defer run.mu.Unlock()
	if run.retired {
		return
	}
	switch reported := event.(type) {
	case agent.ParticipantJoined:
		if isAgent(spec.UserID, reported.Participant) {
			return
		}
		run.people[reported.Participant.ID] = struct{}{}
		// Say the greeting in a different goroutine, because the events of the greeting come
		// back through this event loop.
		if greet := run.pendingGreeting; greet != nil {
			run.pendingGreeting = nil
			s.running.Go(greet)
		}
		if _, ok := phone.CallerNumber(reported.Participant.UserID); ok {
			s.callerSeen.Store(true)
		}
		if run.settled && s.phoneSession(spec) {
			s.ClearTimeout()
		}
	case agent.ParticipantLeft:
		if _, ok := run.people[reported.Participant.ID]; !ok {
			return
		}
		delete(run.people, reported.Participant.ID)
		if run.settled && len(run.people) == 0 && s.callerSeen.Load() {
			s.SetTimeout(hangupTimeout, TimeoutHangup)
		}
	case agent.Joined:
		run.settled = true
		first := !s.settledOnce.Swap(true)
		if !s.phoneSession(spec) {
			return
		}
		switch {
		case len(run.people) > 0:
			s.ClearTimeout()
		case s.callerSeen.Load():
			// Voice was stopped and started again after the caller hung up.
			s.SetTimeout(hangupTimeout, TimeoutHangup)
		case first && spec.Phone != nil:
			if spec.Phone.VendorCallID != "" {
				s.SetTimeout(outboundStartTimeout, TimeoutStart)
			} else {
				s.SetTimeout(inboundStartTimeout, TimeoutStart)
			}
		}
	}
}

// isAgent reports whether a participant is the agent itself under another user, such as
// the "-video" one a Python agent with video processors joins: they are nobody to talk to.
func isAgent(agentUserID string, p stt.Participant) bool {
	return p.UserID == agentUserID || strings.HasPrefix(p.UserID, agentUserID+"-")
}

// CallerJoined reports whether a phone caller has been in the session's call: on a call the
// router placed, the person rung has picked up.
func (s *Session) CallerJoined() bool {
	return s.callerSeen.Load()
}
