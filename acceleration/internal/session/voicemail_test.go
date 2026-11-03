package session

import (
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
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
	never(s.T(), func() bool {
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
	never(s.T(), func() bool { return created.State() == Ended || !s.manager.Running(created.ID()) }, 4*testTimeout, 5*time.Millisecond,
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

func (s *SessionSuite) TestAnUnknownPickUpNobodySpokeOnHearsTheMessage() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineUnknown))

	s.endedForVoicemail(created)
	s.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.")
}

func (s *SessionSuite) TestAnUnknownPickUpAfterTheCallerSpokeToTheAgentChangesNothing() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	s.callerSays("who is this?")
	s.eventually(func() bool { return len(s.model.requests()) == 1 && !created.Busy() }, "the caller was never answered")
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineUnknown))

	s.carriesOn(created, said)
}

func (s *SessionSuite) TestAFaxIsHungUpOnWithNothingSaid() {
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)
	s.settles(created)
	said := len(s.voice.spoken())

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineFax))

	s.endedForVoicemail(created)
	s.Len(s.voice.spoken(), said, "something was said to the fax")
}

func (s *SessionSuite) TestACallNobodySpokeOnWithNoReportByTheBoundHearsTheMessage() {
	wait := machineWait
	machineWait = 4 * testTimeout
	s.T().Cleanup(func() { machineWait = wait })
	s.manages()
	created := s.rings(leavesVoicemail("Hi Ali, this is Northwind Dental."), caller)

	s.endedForVoicemail(created)
	s.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.")
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

// checksForVoicemail has a placed call that leaves a voicemail ring a caller, with a model
// that says goodbye and calls end_call on its first turn.
func (s *SessionSuite) checksForVoicemail() *Session {
	s.manages()
	s.model.reply = "Goodbye."
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: "end_call", Arguments: "{}"}}
	spec := leavesVoicemail("Hi Ali, this is Northwind Dental.")
	spec.ConfigID, spec.Calls = "cfg-1", endsCalls()
	created := s.rings(spec, caller)
	s.settles(created)
	s.hears(created)
	return created
}

// asksToHangUp has the model call end_call on a turn nobody on the phone said, and returns
// what it was told came of it.
func (s *SessionSuite) asksToHangUp(created *Session) string {
	s.says(created, "nobody has said anything for a while")
	var told string
	s.eventually(func() bool {
		for _, request := range s.model.requests() {
			for _, message := range request.Input {
				if message.ToolCallID == "call-1" {
					told = message.Content + llm.TextOf(message.Parts)
					return true
				}
			}
		}
		return created.State() == Ended
	}, "the model was never told what end_call did")
	s.Require().NotEqual(Ended, created.State(), "the agent hung up before the machine report")
	return told
}

func (s *SessionSuite) TestAnAgentHangingUpBeforeTheMachineReportIsHeldAndToldToWait() {
	created := s.checksForVoicemail()

	told := s.asksToHangUp(created)

	s.Contains(told, "answering machine")
	s.NotContains(told, "did not work")
	never(s.T(), func() bool { return created.State() == Ended || !s.manager.Running(created.ID()) },
		4*testTimeout, 5*time.Millisecond, "the agent hung up before the machine report")
}

func (s *SessionSuite) TestAMachineReportAfterAHeldHangUpLeavesTheMessage() {
	created := s.checksForVoicemail()
	s.asksToHangUp(created)

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineGreetingEnded))

	s.endedForVoicemail(created)
	s.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.")
}

func (s *SessionSuite) TestAHeldHangUpLeavesTheMessageOnceTheCheckRunsOut() {
	wait := machineWait
	machineWait = 4 * testTimeout
	s.T().Cleanup(func() { machineWait = wait })
	created := s.checksForVoicemail()

	s.asksToHangUp(created)

	s.endedForVoicemail(created)
	s.Contains(s.saidText(), "Hi Ali, this is Northwind Dental.")
}

func (s *SessionSuite) TestAHeldHangUpGoesThroughWhenAPersonPicksUp() {
	created := s.checksForVoicemail()
	s.asksToHangUp(created)

	s.True(s.manager.Machine("acme", created.ID(), phone.MachineHuman))

	s.eventually(func() bool { return created.State() == Ended && !s.manager.Running(created.ID()) },
		"the held hang-up never went through")
	s.Equal(EndAgentHangup, created.endReason())
}

func (s *SessionSuite) TestTheCallerSpeakingToTheAgentLetsItHangUp() {
	created := s.checksForVoicemail()

	s.callerSays("who is this?")

	s.eventually(func() bool { return created.State() == Ended && !s.manager.Running(created.ID()) },
		"the agent could not hang up on a caller who spoke to it")
	s.Equal(EndAgentHangup, created.endReason())
}

// callerSays has the phone caller say text to the agent.
func (s *SessionSuite) callerSays(text string) {
	s.call().inbound <- agent.InboundAudio{Participant: caller, Audio: audio.PcmData{Samples: make([]int16, 320), SampleRate: 16_000, Channels: 1}}
	s.ears.emitter.Send(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: text, Language: "en", Confidence: 1})
}
