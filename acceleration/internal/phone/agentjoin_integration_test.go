//go:build integration

package phone

import (
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
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

// Stream lists every kind of participant alike. A second phone of the same number is another
// user session, so it is someone else too.
func (s *MeteringSuite) TestACallWithSomeoneBesidesThePhoneIsNeverEnded() {
	for name, other := range map[string]chattest.Participant{
		"router agent":             {UserID: "agent-1", UserSessionID: "agent-session"},
		"app user":                 {UserID: "user-42", UserSessionID: "user-session"},
		"a phone of another":       {UserID: "sip-+15125550199", UserSessionID: "phone-session-2"},
		"a phone of the same user": {UserID: person, UserSessionID: "phone-session-2"},
	} {
		s.Run(name, func() {
			s.SetupTest()
			waits := s.agentWaits()
			placed := s.place()
			s.deployment.PutCallParticipants(placed.CallType, placed.CallID,
				chattest.Participant{UserID: person, UserSessionID: personSession}, other)

			s.answered(placed.CallType, placed.CallID, personSession)
			s.waitEnds(waits)

			s.checked("DEBUG", someoneElse)
			s.False(s.endedAtStream(placed.CallType, placed.CallID))
		})
	}
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

// A call that arrived is not one the router placed: the inbound side is checked when it arrives.
func (s *MeteringSuite) TestACallThatArrivedIsNotChecked() {
	waits := 0
	s.service.agentJoinTimeout = 10 * time.Second
	s.service.agentWait = func(time.Duration) <-chan time.Time {
		waits++
		return make(chan time.Time)
	}
	callID := "call-" + uuid.NewString()
	s.arrived(callID)

	s.joined(defaultCallType, callID, "caller-1", "+15125550125")

	s.Zero(waits)
}
