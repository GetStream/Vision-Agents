//go:build integration

package phone

import (
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The person a placed call rings, as Stream names their phone, and the user session it joins with.
const (
	person        = "sip-+15125550123"
	personSession = "phone-session-1"
)

const (
	noAgent     = "no agent joined the call; ending it"
	phoneGone   = "the phone left the call before the agent check"
	someoneElse = "someone besides the phone is in the call"
	notListed   = "could not ask Stream who is in the call; the call continues"
)

// agentWaits turns the agent check on and gives the test its timer: each value sent ends one wait.
func (s *MeteringSuite) agentWaits() chan<- time.Time {
	waits := make(chan time.Time)
	s.service.agentJoinTimeout = 10 * time.Second
	s.service.agentWait = func(time.Duration) <-chan time.Time { return waits }
	return waits
}

// waitEnds ends the wait of an agent check that is waiting, and fails when none is.
func (s *MeteringSuite) waitEnds(waits chan<- time.Time) {
	select {
	case waits <- time.Now():
	case <-time.After(5 * time.Second):
		s.FailNow("no agent check is waiting")
	}
}

// checked waits until the agent check logged message at level.
func (s *MeteringSuite) checked(level, message string) {
	s.Eventually(func() bool { return s.logs.count(level, message) == 1 }, 5*time.Second, 10*time.Millisecond,
		"the agent check logged %q at %s", message, level)
}

// endedAtStream reports whether Stream was asked to end the call.
func (s *MeteringSuite) endedAtStream(callType, callID string) bool {
	for _, request := range s.deployment.Requests(s.deployment.Key) {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/video/call/"+callType+"/"+callID+"/mark_ended") {
			return true
		}
	}
	return false
}

func (s *MeteringSuite) TestACallAnsweredWithNoAgentInItIsEndedAfterTheWait() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})

	s.answered(placed.CallType, placed.CallID, personSession)
	s.False(s.endedAtStream(placed.CallType, placed.CallID), "the call is not ended before the wait")
	s.waitEnds(waits)

	s.checked("INFO", noAgent)
	s.Eventually(func() bool { return s.endedAtStream(placed.CallType, placed.CallID) }, 5*time.Second, 10*time.Millisecond)
	s.Contains(s.logs.String(), "session_id="+placed.CallID)
	s.Contains(s.logs.String(), "vendor_call_id=telnyx-call")
	s.Contains(s.logs.String(), "waited=10s")
}

// The agent was in the call while it rang, and left before the person answered.
func (s *MeteringSuite) TestACallWhoseAgentLeftBeforeTheAnswerIsEndedAfterTheWait() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: "agent-1", UserSessionID: "agent-session"})
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})

	s.answered(placed.CallType, placed.CallID, personSession)
	s.waitEnds(waits)

	s.checked("INFO", noAgent)
	s.Eventually(func() bool { return s.endedAtStream(placed.CallType, placed.CallID) }, 5*time.Second, 10*time.Millisecond)
}

// A campaign opens the agent's session while the phone rings, so the person can answer first.
func (s *MeteringSuite) TestAnAgentThatJoinsWithinTheWaitKeepsTheCall() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})
	s.answered(placed.CallType, placed.CallID, personSession)

	s.deployment.PutCallParticipants(placed.CallType, placed.CallID,
		chattest.Participant{UserID: person, UserSessionID: personSession}, chattest.Participant{UserID: "agent-1", UserSessionID: "agent-session"})
	s.waitEnds(waits)

	s.checked("DEBUG", someoneElse)
	s.False(s.endedAtStream(placed.CallType, placed.CallID))
	s.Zero(s.logs.count("INFO", noAgent))
}

// keptWith answers a placed call with other in it beside the phone, and checks that the call
// continues.
func (s *MeteringSuite) keptWith(other chattest.Participant) {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID,
		chattest.Participant{UserID: person, UserSessionID: personSession}, other)

	s.answered(placed.CallType, placed.CallID, personSession)
	s.waitEnds(waits)

	s.checked("DEBUG", someoneElse)
	s.False(s.endedAtStream(placed.CallType, placed.CallID))
}

func (s *MeteringSuite) TestACallWithARouterAgentInItContinues() {
	s.keptWith(chattest.Participant{UserID: "agent-1", UserSessionID: "agent-session"})
}

func (s *MeteringSuite) TestACallWithAnAppUserInItContinues() {
	s.keptWith(chattest.Participant{UserID: "user-42", UserSessionID: "user-session"})
}

func (s *MeteringSuite) TestACallWithAnotherPhoneInItContinues() {
	s.keptWith(chattest.Participant{UserID: "sip-+15125550199", UserSessionID: "phone-session-2"})
}

// Participants are told apart by user session, so the same number on a second session is
// someone else.
func (s *MeteringSuite) TestACallWithTheSamePhoneUserOnAnotherSessionContinues() {
	s.keptWith(chattest.Participant{UserID: person, UserSessionID: "phone-session-2"})
}

func (s *MeteringSuite) TestACallThePersonLeftDuringTheWaitIsLeftAlone() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})
	s.answered(placed.CallType, placed.CallID, personSession)

	s.deployment.PutCallParticipants(placed.CallType, placed.CallID)
	s.waitEnds(waits)

	s.checked("DEBUG", phoneGone)
	s.False(s.endedAtStream(placed.CallType, placed.CallID))
	s.Zero(s.logs.count("INFO", noAgent))
	s.Zero(s.logs.count("WARN", ""))
}

// The router does not end a call it could not check.
func (s *MeteringSuite) TestACallStreamCouldNotListIsLeftAlone() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})
	s.deployment.Fail(http.MethodGet, "/video/call/"+placed.CallType+"/"+placed.CallID)

	s.answered(placed.CallType, placed.CallID, personSession)
	s.waitEnds(waits)

	s.checked("WARN", notListed)
	s.False(s.endedAtStream(placed.CallType, placed.CallID))
}

// Stream does not order its events: the phone of a transfer can join before its leg is written.
func (s *MeteringSuite) TestAPhoneThatJoinedBeforeItsLegWasWrittenIsCheckedToo() {
	waits := s.agentWaits()
	callID := "call-" + uuid.NewString()
	s.deployment.PutCallParticipants(defaultCallType, callID, chattest.Participant{UserID: "sip-+15125550124", UserSessionID: "human-1"})
	s.joined(defaultCallType, callID, "human-1", "+15125550124")

	s.transfer(callID, "+15125550124")
	s.waitEnds(waits)

	s.checked("INFO", noAgent)
	s.Eventually(func() bool { return s.endedAtStream(defaultCallType, callID) }, 5*time.Second, 10*time.Millisecond)
}

// noWait has every agent check run at once, so a check that should not start ends the call
// before neverEnded looks.
func (s *MeteringSuite) noWait() {
	done := make(chan time.Time)
	close(done)
	s.service.agentJoinTimeout = 10 * time.Second
	s.service.agentWait = func(time.Duration) <-chan time.Time { return done }
}

// neverEnded checks, over a bounded time, that the call is not ended and that no end is logged.
func (s *MeteringSuite) neverEnded(callType, callID string) {
	s.Never(func() bool { return s.endedAtStream(callType, callID) || s.logs.count("INFO", noAgent) > 0 },
		500*time.Millisecond, 10*time.Millisecond)
}

// A call that arrived is not one the router placed: the inbound side is checked when it arrives.
func (s *MeteringSuite) TestACallThatArrivedIsNotCheckedWhenItsCallerJoins() {
	s.noWait()
	callID := "call-" + uuid.NewString()
	s.deployment.PutCallParticipants(defaultCallType, callID, chattest.Participant{UserID: "sip-+15125550125", UserSessionID: "caller-1"})
	s.arrived(callID)

	s.joined(defaultCallType, callID, "caller-1", "+15125550125")

	s.neverEnded(defaultCallType, callID)
}

// Most calls that arrive open their leg with the caller already in it.
func (s *MeteringSuite) TestACallThatArrivedWithItsCallerIsNotChecked() {
	s.noWait()
	callID := "call-" + uuid.NewString()
	s.deployment.PutCallParticipants(defaultCallType, callID, chattest.Participant{UserID: "sip-+15125550125", UserSessionID: "caller-1"})

	s.service.OpenLeg(s.ctx, routing.Owner{CustomerID: s.customer, Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound}},
		store.CallLeg{CallType: defaultCallType, CallID: callID, StreamSessionID: "session-1", ParticipantSession: "caller-1",
			Vendor: s.vendor.vendor, Direction: store.Inbound, LocalNumber: s.e164, RemoteNumber: "+15125550125", Country: "US"}, true)

	s.neverEnded(defaultCallType, callID)
}

// The setting reaches a real timer: a service built from its options, with no test timer, ends
// an empty call.
func (s *MeteringSuite) TestTheWaitSetInTheOptionsEndsAnEmptyCall() {
	service, err := NewService(ServiceOptions{
		Registry: s.service.registry, Store: s.store, Recorder: s.recorder, Now: s.clock.now,
		Apps: s.service.apps, Gate: s.service.gate, Logger: s.service.logger,
		AgentJoinTimeout: 20 * time.Millisecond,
	})
	s.Require().NoError(err)
	s.service = service
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})

	s.answered(placed.CallType, placed.CallID, personSession)

	s.Eventually(func() bool { return s.endedAtStream(placed.CallType, placed.CallID) }, 5*time.Second, 10*time.Millisecond)
}

// A router that stops does not end a call after it said it stopped.
func (s *MeteringSuite) TestCloseStopsAWaitingCheck() {
	waits := s.agentWaits()
	placed := s.place()
	s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})
	s.answered(placed.CallType, placed.CallID, personSession)

	s.Require().NoError(s.service.Close(s.ctx))

	select {
	case waits <- time.Now():
	case <-time.After(500 * time.Millisecond):
	}
	s.Never(func() bool { return s.endedAtStream(placed.CallType, placed.CallID) }, 500*time.Millisecond, 10*time.Millisecond,
		"a router that stopped ended the call")
}

// A wait that ends as the router stops does not start a check. Go picks a ready case of a
// select at random, so each call has an even chance to show a check that starts anyway.
func (s *MeteringSuite) TestAWaitThatEndsAsTheRouterStopsStartsNoCheck() {
	s.noWait()
	s.Require().NoError(s.service.Close(s.ctx))
	var calls []Placed
	for range 16 {
		placed := s.place()
		s.deployment.PutCallParticipants(placed.CallType, placed.CallID, chattest.Participant{UserID: person, UserSessionID: personSession})
		s.answered(placed.CallType, placed.CallID, personSession)
		calls = append(calls, placed)
	}

	s.Never(func() bool {
		for _, placed := range calls {
			if s.endedAtStream(placed.CallType, placed.CallID) {
				return true
			}
		}
		return false
	}, 500*time.Millisecond, 10*time.Millisecond, "a check started after the router stopped")
}

// Stream answers a call whose session has not started with no session at all. The check must
// not crash the router on it.
func (s *MeteringSuite) TestACallWithNoSessionIsLeftAlone() {
	waits := s.agentWaits()
	placed := s.place()
	s.answered(placed.CallType, placed.CallID, personSession)

	s.waitEnds(waits)

	s.checked("DEBUG", phoneGone)
	s.False(s.endedAtStream(placed.CallType, placed.CallID))
}
