package session

import (
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
	time.Sleep(4 * testTimeout)
	s.stillRuns(created)
	s.Len(s.voice.spoken(), said, "something was said for the report")
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

func (s *SessionSuite) TestAMachineReportForASessionHeldElsewhereIsNotHere() {
	s.manages()

	s.False(s.manager.Machine("acme", "held-elsewhere", phone.MachineGreetingEnded))
}
