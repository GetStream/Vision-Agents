package session

import (
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
			s.SetTimeout(hangupTimeout)
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
			s.SetTimeout(hangupTimeout)
		case first && spec.Phone != nil:
			if spec.Phone.VendorCallID != "" {
				s.SetTimeout(outboundStartTimeout)
			} else {
				s.SetTimeout(inboundStartTimeout)
			}
		}
	}
}

// isAgent reports whether a participant is the agent itself under another user, such as
// the "-video" one a Python agent with video processors joins: they are nobody to talk to.
func isAgent(agentUserID string, p stt.Participant) bool {
	return p.UserID == agentUserID || strings.HasPrefix(p.UserID, agentUserID+"-")
}
