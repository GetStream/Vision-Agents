package agent

import (
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// heardLine is the message logged with what the agent heard from a participant.
const heardLine = "what the agent heard from a participant"

// joinHeard joins with the agent's log lines kept.
func (s *AgentSuite) joinHeard() {
	s.logs = &logLines{}
	s.join(true)
}

// speakFor pushes n chunks of a participant's audio and waits until the transcriber has
// all of them, transcribed counting from what it had before.
func (s *AgentSuite) speakFor(participant stt.Participant, n int) {
	had := len(s.ears.transcribed())
	for range n {
		s.speak(participant)
	}
	s.eventually(func() bool { return len(s.ears.transcribed()) == had+n },
		"the audio never reached the transcriber")
}

// heardAt waits for the line logged at moment for a participant and returns it.
func (s *AgentSuite) heardAt(moment, participantID string) map[string]any {
	var found map[string]any
	s.eventually(func() bool {
		for _, record := range s.logs.records(heardLine) {
			if record["moment"] == moment && record["participant"] == participantID {
				found = record
				return true
			}
		}
		return false
	}, "nothing was said of what was heard at "+moment)
	return found
}

func (s *AgentSuite) TestASettledTurnSaysWhatWasHeardFromItsCaller() {
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 3)

	s.mutters(alice, "book a")
	s.says(alice, "book a table")

	line := s.heardAt("turn", "alice")
	s.Equal(float64(3), line["frames"])
	s.Equal(float64(60), line["audio_ms"], "three chunks of 20 ms")
	s.Equal(float64(1), line["partials"])
	s.Equal(float64(1), line["finals"])
	s.GreaterOrEqual(line["since_frame_ms"], float64(0))
	s.GreaterOrEqual(line["since_words_ms"], float64(0))
	s.Contains(line["turn"], "turn-")
	s.NotContains(line, "text", "what was said stays out of the log")
}

func (s *AgentSuite) TestATranscriptWithNoWordsIsNotCountedAsHeard() {
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 1)

	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: " "})
	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: ""})
	s.says(alice, "hello")

	line := s.heardAt("turn", "alice")
	s.Equal(float64(0), line["partials"])
	s.Equal(float64(1), line["finals"])
}

func (s *AgentSuite) TestACheckInSaysTheCallerWasHeardButNotTranscribed() {
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 2)

	s.agent.checkIn(alice, idlePurpose)

	line := s.heardAt("checkin", "alice")
	s.Equal(idlePurpose, line["purpose"])
	s.Equal(float64(2), line["frames"])
	s.Equal(float64(40), line["audio_ms"])
	s.Equal(float64(0), line["partials"])
	s.Equal(float64(0), line["finals"])
	s.GreaterOrEqual(line["since_frame_ms"], float64(0))
	s.Equal(float64(-1), line["since_words_ms"], "no words were ever heard")
}

func (s *AgentSuite) TestACheckInToSomebodyNeverHeardSaysNothingArrived() {
	s.joinHeard()

	s.agent.checkIn(stt.Participant{ID: "bob"}, idlePurpose)

	line := s.heardAt("checkin", "bob")
	s.Equal(float64(0), line["frames"])
	s.Equal(float64(0), line["audio_ms"])
	s.Equal(float64(-1), line["since_frame_ms"])
	s.Equal(float64(-1), line["since_words_ms"])
}

func (s *AgentSuite) TestTheEndOfTheCallSaysWhatWasHeardFromEachParticipant() {
	s.joinHeard()
	alice, bob := stt.Participant{ID: "alice"}, stt.Participant{ID: "bob"}
	s.speakFor(alice, 2)
	s.speakFor(bob, 1)

	s.Require().NoError(s.agent.Close())

	var ended []map[string]any
	for _, record := range s.logs.records(heardLine) {
		if record["moment"] == "end" {
			ended = append(ended, record)
		}
	}
	s.Require().Len(ended, 2)
	s.Equal("alice", ended[0]["participant"])
	s.Equal(float64(2), ended[0]["frames"])
	s.Equal("bob", ended[1]["participant"])
	s.Equal(float64(1), ended[1]["frames"])
}

func (s *AgentSuite) TestAudioDroppedDuringAModelSwitchStillCountsAsReceived() {
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 1)

	received := testutil.ToFloat64(inboundFrames)
	s.agent.switching.Store(true)
	s.speak(alice)
	s.eventually(func() bool { return testutil.ToFloat64(inboundFrames) == received+1 },
		"the frame was never taken from the call")
	s.agent.switching.Store(false)

	s.agent.checkIn(alice, idlePurpose)
	s.Equal(float64(2), s.heardAt("checkin", "alice")["frames"])
}

func (s *AgentSuite) TestInboundAudioAndTranscriptsAreCountedForTheRouter() {
	frames := testutil.ToFloat64(inboundFrames)
	partials := testutil.ToFloat64(sttPartials.WithLabelValues("stub"))
	finals := testutil.ToFloat64(sttFinals.WithLabelValues("stub"))
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 2)

	s.mutters(alice, "book")
	s.mutters(alice, "book a")
	s.says(alice, "book a table")
	s.heardAt("turn", "alice")

	s.Equal(frames+2, testutil.ToFloat64(inboundFrames))
	s.Equal(partials+2, testutil.ToFloat64(sttPartials.WithLabelValues("stub")))
	s.Equal(finals+1, testutil.ToFloat64(sttFinals.WithLabelValues("stub")))
}

// The expected values were read from a probe of this test on the base commit, before
// anything was counted.
func (s *AgentSuite) TestCountingWhatIsHeardChangesNothingTheCallerHearsOrIsAnswered() {
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	for range 3 {
		s.speak(alice)
	}
	s.eventually(func() bool { return len(s.ears.transcribed()) == 3 }, "the audio never reached the transcriber")

	s.mutters(alice, "book a")
	s.says(alice, "book a table")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the turn was never answered")
	s.eventually(func() bool { return len(s.edge.heard()) == 2 }, "the reply never reached the call")
	s.eventually(func() bool { return len(s.voice.spoken()) == 3 }, "the reply never reached the voice")
	transcribed := s.ears.transcribed()
	s.Require().Len(transcribed, 3)
	for _, pcm := range transcribed {
		s.Len(pcm.Samples, 320)
	}
	s.Require().Len(s.model.requests(), 1)
	s.Equal("book a table", s.model.requests()[0].Input[0].Content)
	var spoken []string
	for _, request := range s.voice.spoken() {
		spoken = append(spoken, request.Text)
	}
	s.Equal([]string{"Hello there.", "How are you?", ""}, spoken)
	s.Equal([]llm.Message{
		{Role: llm.User, Content: "book a table"},
		{Role: llm.Assistant, Content: "Hello there. How are you?"},
	}, s.agent.History())
}

func (s *AgentSuite) TestATurnIsReportedForItsOwnCallerWhenSomebodyElseSpokeLast() {
	s.joinHeard()
	alice, bob := stt.Participant{ID: "alice"}, stt.Participant{ID: "bob"}
	s.speakFor(alice, 2)
	s.speakFor(bob, 1)

	s.says(alice, "book a table")
	s.mutters(bob, "and")
	s.eventually(func() bool { return s.agent.heardFrom("bob").partials.Load() == 1 },
		"bob's words never arrived")
	s.Equal("bob", s.agent.floor().LastParticipant.ID)

	line := s.heardAt("turn", "alice")
	s.Equal(float64(2), line["frames"], "alice's frames, not bob's")
	for _, record := range s.logs.records(heardLine) {
		s.NotEqual("bob", record["participant"], "bob's turn never settled")
	}
}

func (s *AgentSuite) TestACheckInBeforeAnyWordsSaysWhatWasHeardFromEveryone() {
	s.joinHeard()
	alice, bob := stt.Participant{ID: "alice"}, stt.Participant{ID: "bob"}
	s.speakFor(alice, 3)
	s.speakFor(bob, 1)

	s.agent.checkIn(stt.Participant{}, idlePurpose)

	line := s.heardAt("checkin", "alice")
	s.Equal(idlePurpose, line["purpose"])
	s.Equal(float64(3), line["frames"])
	s.Equal(float64(1), s.heardAt("checkin", "bob")["frames"])
	for _, record := range s.logs.records(heardLine) {
		s.NotEqual("", record["participant"], "nobody is not who was heard")
	}
}

func (s *AgentSuite) TestAPartialAloneCountsAsWordsHeardAtACheckIn() {
	s.joinHeard()
	alice := stt.Participant{ID: "alice"}
	s.speakFor(alice, 1)

	s.mutters(alice, "book")
	s.eventually(func() bool { return s.agent.heardFrom("alice").partials.Load() == 1 },
		"the partial never arrived")
	s.agent.checkIn(alice, idlePurpose)

	line := s.heardAt("checkin", "alice")
	s.Equal(float64(0), line["finals"])
	s.GreaterOrEqual(line["since_words_ms"], float64(0), "a partial is words heard")
}

func (s *AgentSuite) TestHowLongAgoIsInMilliseconds() {
	s.Equal(int64(-1), sinceMs(0), "never")
	ago := sinceMs(time.Now().Add(-1500 * time.Millisecond).UnixNano())
	s.GreaterOrEqual(ago, int64(1500))
	s.Less(ago, int64(1900))
}
