package session

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

var (
	caller  = stt.Participant{ID: "caller-1", UserID: "sip-caller", Name: "Caller"}
	visitor = stt.Participant{ID: "visitor-1", UserID: "visitor", Name: "Visitor"}
)

// shortensPhoneTimeouts makes the phone timeouts short enough to wait out, each a different
// length so a test can tell which one ended a session.
func (s *SessionSuite) shortensPhoneTimeouts() {
	hangup, answered, placed := hangupTimeout, inboundStartTimeout, outboundStartTimeout
	hangupTimeout = 4 * testTimeout
	inboundStartTimeout = 6 * testTimeout
	outboundStartTimeout = 8 * testTimeout
	s.T().Cleanup(func() {
		hangupTimeout, inboundStartTimeout, outboundStartTimeout = hangup, answered, placed
	})
}

// inbound and outbound leave the number out: a number needs telephony the suite does not
// have, and only whether the session has a phone, and a vendor call, decides its timeout.
func inbound() Spec {
	return Spec{Phone: &PhoneSpec{}}
}

func outbound() Spec {
	return Spec{Phone: &PhoneSpec{VendorCallID: "vendor-call-1"}}
}

// rings creates a voice session on a call the people present are already in.
func (s *SessionSuite) rings(spec Spec, present ...stt.Participant) *Session {
	s.present = present
	return s.joins(spec)
}

// call is the edge the session is on now.
func (s *SessionSuite) call() *quietEdge {
	return s.edges[len(s.edges)-1]
}

// watches attaches a watcher, so a test can wait for the session to have taken an event.
func (s *SessionSuite) watches(created *Session) <-chan Event {
	events, detach := created.Watch()
	s.T().Cleanup(detach)
	return events
}

// saw waits for the session to have taken somebody coming or going.
func (s *SessionSuite) saw(events <-chan Event, p stt.Participant, joined bool) {
	deadline := time.After(settleFor)
	for {
		select {
		case event := <-events:
			switch reported := event.(type) {
			case agent.ParticipantJoined:
				if joined && reported.Participant.ID == p.ID {
					return
				}
			case agent.ParticipantLeft:
				if !joined && reported.Participant.ID == p.ID {
					return
				}
			}
		case <-deadline:
			s.FailNow("the session never heard of them", "%s joined=%v", p.ID, joined)
		}
	}
}

// settles waits for the session to have taken its run's Joined, which decides the timeout
// from who was already in the call.
func (s *SessionSuite) settles(created *Session) {
	s.eventually(func() bool {
		created.presenceMu.Lock()
		run := created.runs[created.current()]
		created.presenceMu.Unlock()
		if run == nil {
			return false
		}
		run.mu.Lock()
		defer run.mu.Unlock()
		return run.settled
	}, "the run never settled")
}

// outlivesPhoneTimeouts checks the session is still running after the longest of them.
func (s *SessionSuite) outlivesPhoneTimeouts(created *Session) {
	time.Sleep(outboundStartTimeout + testTimeout)
	s.Equal(Live, created.State(), "the session ended")
	s.True(s.manager.Running(created.ID()), "the session was taken off")
}

func (s *SessionSuite) TestAPhoneSessionEndsAfterTheLastPersonLeaves() {
	s.manages()
	created := s.rings(inbound(), caller)

	left := time.Now()
	s.call().leaves(caller)

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(left), hangupTimeout, "the session ended before its timeout")
	s.True(s.edges[0].gone(), "the agent is still in the call")
}

func (s *SessionSuite) TestALeaveFromSomebodyNeverInTheCallDoesNotMoveTheHangup() {
	s.manages()
	created := s.rings(inbound(), caller)
	events := s.watches(created)

	s.call().leaves(caller)
	s.saw(events, caller, false)
	created.mu.Lock()
	armed := created.timeoutArmed
	created.mu.Unlock()

	s.call().leaves(visitor)
	s.saw(events, visitor, false)

	created.mu.Lock()
	defer created.mu.Unlock()
	s.Equal(armed, created.timeoutArmed, "the stray leave touched the timeout")
}

func (s *SessionSuite) TestAPhoneCallWithAnotherPersonKeepsRunning() {
	s.manages()
	created := s.rings(inbound(), caller, visitor)
	events := s.watches(created)

	s.call().leaves(caller)
	s.saw(events, caller, false)

	s.outlivesPhoneTimeouts(created)
	s.call().leaves(visitor)
	s.timedOut(created)
}

func (s *SessionSuite) TestAPhoneCallerRejoiningInTimeKeepsTheSession() {
	s.manages()
	created := s.rings(inbound(), caller)
	events := s.watches(created)

	s.call().leaves(caller)
	s.saw(events, caller, false)
	rejoined := stt.Participant{ID: "caller-2", UserID: caller.UserID}
	s.call().arrives(rejoined)

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestAnInboundCallerAlreadyInTheCallIsNeverEndedByTheStartTimeout() {
	s.manages()
	created := s.rings(inbound(), caller)

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestAnOutboundSessionNobodyJoinsEndsAtTheStartTimeout() {
	s.manages()
	started := time.Now()
	created := s.rings(outbound())

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(started), outboundStartTimeout, "the session ended before the outbound start timeout")
}

func (s *SessionSuite) TestAnInboundSessionWhoseCallerIsGoneEndsAtTheStartTimeout() {
	s.manages()
	started := time.Now()
	created := s.rings(inbound())

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(started), inboundStartTimeout, "the session ended before the inbound start timeout")
}

// endingReason takes the place of the session's ending, to read which timeout reached it.
func (s *SessionSuite) endingReason(created *Session) <-chan TimeoutReason {
	reasons := make(chan TimeoutReason, 1)
	created.mu.Lock()
	created.end = func(reason TimeoutReason) { reasons <- reason }
	created.mu.Unlock()
	return reasons
}

// endedBy waits for the session's timeout and checks which one it was.
func (s *SessionSuite) endedBy(reasons <-chan TimeoutReason, want TimeoutReason) {
	select {
	case reason := <-reasons:
		s.Equal(want, reason)
	case <-time.After(settleFor):
		s.FailNow("no timeout ended the session")
	}
}

func (s *SessionSuite) TestTheHangupTimeoutSaysItIsTheHangup() {
	s.manages()
	created := s.rings(inbound(), caller)
	reasons := s.endingReason(created)

	s.call().leaves(caller)

	s.endedBy(reasons, TimeoutHangup)
}

func (s *SessionSuite) TestTheStartTimeoutSaysItIsTheStart() {
	s.manages()
	created := s.rings(outbound())
	reasons := s.endingReason(created)

	s.endedBy(reasons, TimeoutStart)
}

func (s *SessionSuite) TestASessionWithoutPhoneGetsNoStartTimeout() {
	s.manages()
	created := s.rings(Spec{})

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestASessionWithoutPhoneBecomesAPhoneSessionWhenACallerJoins() {
	s.manages()
	created := s.rings(Spec{})
	events := s.watches(created)

	s.call().arrives(caller)
	s.saw(events, caller, true)
	left := time.Now()
	s.call().leaves(caller)

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(left), hangupTimeout, "the session ended before its timeout")
}

func (s *SessionSuite) TestTheRouterNeverTouchesANonPhoneSessionsTimeout() {
	s.manages()
	created := s.rings(Spec{})
	events := s.watches(created)

	set := time.Now()
	created.SetTimeout(2*hangupTimeout, TimeoutHangup)
	s.call().arrives(visitor)
	s.saw(events, visitor, true)
	s.call().leaves(visitor)
	s.saw(events, visitor, false)

	s.timedOut(created)
	s.GreaterOrEqual(time.Since(set), 2*hangupTimeout, "the timeout set on the session was moved")
}

func (s *SessionSuite) TestTheAgentsOwnVideoUserDoesNotKeepTheSessionAlive() {
	s.manages()
	video := stt.Participant{ID: "video-1", UserID: defaultUserID + "-video"}
	created := s.rings(inbound(), caller, video)

	s.call().leaves(caller)

	s.timedOut(created)
}

func (s *SessionSuite) TestTwoPhoneLegsWithOneNumberKeepTheSession() {
	s.manages()
	other := stt.Participant{ID: "caller-2", UserID: caller.UserID}
	created := s.rings(inbound(), caller, other)
	events := s.watches(created)

	s.call().leaves(caller)
	s.saw(events, caller, false)

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestRestartingVoiceAfterTheCallerLeftStillEndsIt() {
	s.manages()
	created := s.rings(inbound(), caller)

	s.Require().NoError(created.StopVoice(s.ctx))
	s.present = nil
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))

	s.timedOut(created)
}

func (s *SessionSuite) TestStoppingVoiceDuringTheHangupTimeoutDoesNotEndIt() {
	s.manages()
	created := s.rings(inbound(), caller)
	events := s.watches(created)

	s.call().leaves(caller)
	s.saw(events, caller, false)
	s.Require().NoError(created.StopVoice(s.ctx))

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestALeaveQueuedOnTheOldRunAfterVoiceStopsSetsNoTimeout() {
	s.manages()
	created := s.rings(inbound(), caller)
	events := s.watches(created)

	// The edge reports the caller leaving as the agent leaves the call.
	s.Require().NoError(created.StopVoice(s.ctx))
	s.saw(events, caller, false)

	s.outlivesPhoneTimeouts(created)
}

func (s *SessionSuite) TestRestartingVoiceWithTheCallerStillThereThenTheyLeaveEndsIt() {
	s.manages()
	created := s.rings(inbound(), caller)
	s.Require().NoError(created.StopVoice(s.ctx))
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))

	s.outlivesPhoneTimeouts(created)
	s.call().leaves(caller)

	s.timedOut(created)
}

func (s *SessionSuite) TestNothingIsSetFromAnUnsettledRun() {
	s.manages()

	timed := s.rings(inbound(), caller)
	s.settles(timed)
	timed.SetTimeout(hangupTimeout, TimeoutHangup)
	unsettled := newPresence()
	timed.observe(unsettled, agent.ParticipantJoined{Participant: visitor})
	s.timedOut(timed)

	untimed := s.rings(inbound(), caller)
	unsettled = newPresence()
	untimed.observe(unsettled, agent.ParticipantJoined{Participant: caller})
	untimed.observe(unsettled, agent.ParticipantLeft{Participant: caller})
	s.outlivesPhoneTimeouts(untimed)
}

func (s *SessionSuite) TestASocketSessionWithPhoneGetsNoTimeout() {
	// A socket edge says nobody comes or goes, which this edge stands in for.
	s.unattended = true
	s.manages()
	created := s.rings(inbound())

	s.outlivesPhoneTimeouts(created)
}
