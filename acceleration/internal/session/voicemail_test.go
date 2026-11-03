package session

import (
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// leavesVoicemail is a placed call's session whose plan is to leave message on a machine,
// or to hang up when message is empty.
func leavesVoicemail(message string) Spec {
	spec := outbound()
	spec.Voicemail = &VoicemailPlan{Message: message}
	return spec
}

// endedForVoicemail waits for the session to have ended for an answering machine.
func (s *SessionSuite) endedForVoicemail(created *Session) {
	s.eventually(func() bool {
		return created.State() == Ended && !s.manager.Running(created.ID())
	}, "the session did not end for the answering machine")
	s.Equal(EndVoicemail, created.endReason())
}

// saidText is every line the voice was asked to say.
func (s *SessionSuite) saidText() []string {
	var said []string
	for _, request := range s.voice.spoken() {
		said = append(said, request.Text)
	}
	return said
}

// carriesOn checks a machine report left the session running and said nothing for it.
func (s *SessionSuite) carriesOn(created *Session, said int) {
	s.Never(func() bool {
		return created.State() == Ended || !s.manager.Running(created.ID()) || len(s.voice.spoken()) != said
	}, 4*testTimeout, 5*time.Millisecond, "the report changed the conversation")
	s.Empty(created.endReason())
}

func (s *SessionSuite) TestAMachineWhoseGreetingEndedHearsTheMessageThenTheSessionEnds() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))

	s.endedForVoicemail(created)
	s.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.")
}

func (s *SessionSuite) TestTheSessionEndsOnlyOnceTheMessageHasBeenSpoken() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	release := s.voice.holds("Hi Ali, this is Northwind Dental.")
	s.T().Cleanup(release)

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))

	s.eventually(func() bool { return slices.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.") },
		"the message was never said")
	s.Never(func() bool { return created.State() == Ended || !s.manager.Running(created.ID()) }, 4*testTimeout, 5*time.Millisecond,
		"the session ended while the message was being spoken")
	release()
	s.endedForVoicemail(created)
}

func (s *SessionSuite) TestAMachineIsHungUpOnWithNothingSaidWhenThereIsNoMessage() {
	s.manages()
	created := s.rings(leavesVoicemail(""), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))

	s.endedForVoicemail(created)
	s.Len(s.voice.spoken(), said, "something was said to the machine")
}

func (s *SessionSuite) TestTheMessageIsLeftOnceWhenTheReportComesTwice() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)

	s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded)
	s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded)

	s.endedForVoicemail(created)
	left := 0
	for _, text := range s.saidText() {
		if text == "Hi Ali, this is Northwind Dental." {
			left++
		}
	}
	s.Equal(1, left, "the message was left more than once")
}

func (s *SessionSuite) TestAPersonPickingUpChangesNothing() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineHuman))

	s.carriesOn(created, said)
}

func (s *SessionSuite) TestAnUnknownPickUpChangesNothing() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineUnknown))

	s.carriesOn(created, said)
}

func (s *SessionSuite) TestAMachineChangesNothingForASessionWithNoVoicemailPlan() {
	s.manages()
	created := s.rings(outbound(), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))

	s.carriesOn(created, said)
}

func (s *SessionSuite) TestAMachineReportForAnotherCustomersSessionChangesNothing() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.manager.Machine("another", created.ID(), phone.MachineGreetingEnded)

	s.carriesOn(created, said)
}

func (s *SessionSuite) TestASpeechToSpeechSessionCannotBeGivenAMessageToLeave() {
	s.manages()
	spec := leavesVoicemail("Hi Ali, this is Northwind Dental.")
	spec.CustomerID, spec.CallID, spec.STSTarget = "acme", "call-1", "en-low-latency"

	_, err := s.manager.Create(s.ctx, spec)

	s.Require().Error(err)
	s.Contains(err.Error(), "cannot say a fixed voicemail message")
}

func (s *SessionSuite) TestAMachineReportForASessionHeldElsewhereIsNotHere() {
	s.manages()

	s.False(s.manager.Machine("acme", "held-elsewhere", phone.MachineGreetingEnded))
}
