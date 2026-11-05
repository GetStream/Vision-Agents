package agent

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// refusingMarkedEdge is a markedEdge whose call has been left, so none of the speech goes out.
type refusingMarkedEdge struct{ *loopbackEdge }

func (refusingMarkedEdge) PublishAudioMarked(audio.PcmData, PlayoutMarks) error { return ErrCallLeft }

// frameAt is 20 ms of a participant's audio with every sample at level.
func frameAt(level int16) audio.PcmData {
	samples := make([]int16, 320)
	for i := range samples {
		samples[i] = level
	}
	return audio.PcmData{Samples: samples, SampleRate: 16_000, Channels: 1}
}

// sends pushes frames of a participant's audio, one per level, and waits for the agent to
// have counted them.
func (s *AgentSuite) sends(participant stt.Participant, levels ...int16) {
	before := s.agent.CallAudio().Frames
	for _, level := range levels {
		s.edge.inbound <- InboundAudio{Participant: participant, Audio: frameAt(level)}
	}
	s.eventually(func() bool { return s.agent.CallAudio().Frames == before+int64(len(levels)) },
		"the agent never counted the frames")
}

// arrives has somebody join the call and waits for the agent to have taken it.
func (s *AgentSuite) arrives(participant stt.Participant) {
	s.edge.attending <- Attendance{Participant: participant, Joined: true}
	s.eventually(func() bool { return countOf[ParticipantJoined](s.reported()) > 0 }, "the arrival was never taken")
}

// talksTo has a participant say something audible that the agent answers, and waits until
// the answer was handed to the edge.
func (s *AgentSuite) talksTo(participant stt.Participant) {
	s.sends(participant, 0, audio.AudibleSample, 0)
	s.says(participant, "hello")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the agent never answered")
}

func (s *AgentSuite) TestACallerInTheCallWhoSentNoAudioIsNoInboundAudio() {
	// Catches a muted caller, or a media path that carried nothing, left uncounted.
	s.join(true)

	s.arrives(stt.Participant{ID: "alice"})

	s.Equal([]Silence{SilenceNoInbound}, s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestACallNobodyJoinedIsNoSilentCall() {
	// Catches a session nobody came to, which ends as a failed start, counted as silent too.
	s.join(true)

	s.Nil(s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestAudioJustBelowTheAudibleLevelIsSilentInboundAudio() {
	// Catches a line of digital silence or the faintest hiss taken as somebody heard.
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.arrives(alice)

	s.sends(alice, 0, audio.AudibleSample-1, -(audio.AudibleSample - 1))

	s.Equal([]Silence{SilenceSilentInbound}, s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestOneAudibleFrameAmongSilentOnesIsAudioTheTranscriberWroteNoWordsFor() {
	// Catches an audible frame after a silent one missed, a quiet negative swing not taken as
	// sound, or a transcript of nothing but silence counted as words.
	s.join(true)
	alice := stt.Participant{ID: "alice"}

	s.sends(alice, 0, -audio.AudibleSample, 0)
	s.says(alice, "  ")
	s.eventually(func() bool { return len(s.ears.transcribed()) == 3 }, "the transcriber never heard the frames")

	s.Equal([]Silence{SilenceNoWords}, s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestACallWithWordsHeardAndSpeechTakenByTheTrackIsNotSilent() {
	// Catches a normal call counted as silent or one-way.
	s.edgeFactory = func(base *loopbackEdge) Edge { return &markedEdge{loopbackEdge: base} }
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.arrives(alice)

	s.talksTo(alice)

	s.eventually(func() bool { return s.agent.CallAudio().Pulled }, "the track's audible frame was never noted")
	heard := s.agent.CallAudio()
	s.True(heard.Published, "the speech handed to the edge was not noted")
	s.Nil(heard.Silences())
}

func (s *AgentSuite) TestSpeechTheTrackNeverTookIsNoAudibleOutbound() {
	// Catches a call whose agent was never heard, because the track never took its speech,
	// left uncounted.
	s.edgeFactory = func(base *loopbackEdge) Edge { return &markedEdge{loopbackEdge: base, neverTakes: true} }
	s.join(true)
	alice := stt.Participant{ID: "alice"}

	s.talksTo(alice)

	s.Equal([]Silence{SilenceNoAudibleOutbound}, s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestSpeechAnEdgeCannotReportOnIsNotJudged() {
	// Catches speech to an edge without a track of its own, which never reports a frame
	// taken, counted as never heard.
	s.join(true)
	alice := stt.Participant{ID: "alice"}

	s.talksTo(alice)

	s.Nil(s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestSpeechForACallAlreadyLeftIsNotSpeechPublished() {
	// Catches speech the edge refused counted as handed over, and then as never heard.
	s.edgeFactory = func(base *loopbackEdge) Edge { return refusingMarkedEdge{base} }
	s.join(true)
	alice := stt.Participant{ID: "alice"}
	s.sends(alice, audio.AudibleSample)
	s.says(alice, "hello")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) > 0 }, "the agent never answered")

	s.Nil(s.agent.CallAudio().Silences())
}

func (s *AgentSuite) TestAudioANativeModelHeardIsNotJudgedForWords() {
	// Catches a speech-to-speech call, whose model writes no transcript here, counted as one
	// whose caller was never transcribed.
	s.joinNativeDelegating()

	s.sends(stt.Participant{ID: "alice"}, audio.AudibleSample)

	heard := s.agent.CallAudio()
	s.True(heard.Native)
	s.Nil(heard.Silences())
}

// speaksNativelyIntoMarked has a speech-to-speech reply spoken into edge, and returns the agent
// once it finished the reply.
func (s *AgentSuite) speaksNativelyIntoMarked(edge Edge) *Agent {
	agent, err := New(Options{CustomerID: "acme", Edge: edge, STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2"})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	seen := collect(agent)
	source := make(chan sts.Event, 8)
	ordered := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(2)
	go agent.receiveSTS(agent.pipe, source, ordered)
	go agent.consumeSTS(agent.pipe, nil, ordered)
	s.T().Cleanup(func() { close(source) })

	pcm := audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}
	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1}
	s.Eventually(func() bool { return countOf[Spoke](seen.seen()) == 1 }, time.Second, time.Millisecond)
	return agent
}

func (s *AgentSuite) TestANativeReplyTheTrackTookIsSpeechPublishedAndHeard() {
	// Catches the speech-to-speech path, which publishes on its own, noting neither.
	agent := s.speaksNativelyIntoMarked(&markedEdge{loopbackEdge: newLoopbackEdge(), trackDelay: 10 * time.Millisecond})

	s.Eventually(func() bool { return agent.CallAudio().Pulled }, settleFor, time.Millisecond,
		"the track's audible frame was never noted")
	s.True(agent.CallAudio().Published)
}

func (s *AgentSuite) TestANativeReplyForACallAlreadyLeftIsNotSpeechPublished() {
	agent := s.speaksNativelyIntoMarked(refusingMarkedEdge{newLoopbackEdge()})

	s.False(agent.CallAudio().Published)
}

func (s *AgentSuite) TestTheRunsOfOneCallAreJudgedTogether() {
	// Catches a session judged on one of its runs: voice stopped and started again, or a run
	// with no roster, whose audio says somebody was there.
	attended := CallAudio{Attended: true}
	spoken := CallAudio{Frames: 2, Audible: true, Words: 1}
	native := CallAudio{Frames: 1, Audible: true, Native: true}
	unheard := CallAudio{Attended: true, Frames: 1, Audible: true, Words: 1, Published: true}
	pulled := CallAudio{Pulled: true}

	s.Nil(attended.Merge(spoken).Silences())
	s.Nil(spoken.Merge(attended).Silences())
	s.Nil(attended.Merge(native).Silences())
	s.Nil(native.Merge(attended).Silences())
	s.Nil(unheard.Merge(pulled).Silences())
	s.Nil(pulled.Merge(unheard).Silences())
	s.Equal([]Silence{SilenceNoInbound}, CallAudio{}.Merge(attended).Silences())
	s.Equal([]Silence{SilenceNoInbound}, attended.Merge(CallAudio{}).Silences())
	s.Equal([]Silence{SilenceNoAudibleOutbound}, CallAudio{}.Merge(unheard).Silences())
	s.Equal([]Silence{SilenceNoInbound, SilenceNoAudibleOutbound},
		attended.Merge(CallAudio{Published: true}).Silences(), "no audio either way is in both cases")
}
