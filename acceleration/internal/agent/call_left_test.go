package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// errLeftByTheCaller is how an edge says the call is over: socketedge's ErrLeft wraps
// ErrCallLeft the same way.
var errLeftByTheCaller = fmt.Errorf("socketedge: %w", ErrCallLeft)

// errTrackBroke is a publish that failed while the call went on.
var errTrackBroke = errors.New("edge: the track is gone")

// refusingEdge is a loopback edge that publishes nothing, failing every chunk as given.
type refusingEdge struct {
	*loopbackEdge
	refusal error
}

func (e *refusingEdge) PublishAudio(audio.PcmData) error { return e.refusal }

// speaksInto has the voice speak one utterance into an edge that refuses its audio, and
// waits until the agent finished it.
func (s *AgentSuite) speaksInto(refusal error) {
	s.edgeFactory = func(base *loopbackEdge) Edge { return &refusingEdge{loopbackEdge: base, refusal: refusal} }
	s.logs = &logLines{}
	s.join(true)

	s.voiceSends(1,
		tts.SynthesisStarted{SynthesisID: "turn-a", Provider: "stub"},
		chunkOf("turn-a", 160, 1000),
		chunkOf("turn-a", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "turn-a"},
	)
}

func (s *AgentSuite) TestSpeechForACallTheCallerLeftIsNoFailure() {
	s.speaksInto(errLeftByTheCaller)

	s.Empty(s.logs.records("agent failure"))
	s.Zero(countOf[Error](s.reported()))
}

// The control: an edge failing while the call goes on is still a failure, once per chunk.
func (s *AgentSuite) TestSpeechAnEdgeCouldNotPublishIsAFailure() {
	s.speaksInto(errTrackBroke)

	s.Len(s.logs.records("agent failure"), 2)
	s.Equal(2, countOf[Error](s.reported()))
}

// speaksNativelyInto has a speech-to-speech reply spoken into an edge that refuses its audio,
// and returns what the agent logged and reported once it finished the reply.
func (s *AgentSuite) speaksNativelyInto(refusal error) (*logLines, *collector) {
	logs := &logLines{}
	edge := &refusingEdge{loopbackEdge: newLoopbackEdge(), refusal: refusal}
	agent, err := New(Options{
		CustomerID: "acme", Edge: edge, STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2",
		Logger: slog.New(slog.NewJSONHandler(logs, nil)),
	})
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
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1}
	s.Eventually(func() bool { return countOf[Spoke](seen.seen()) == 1 }, time.Second, time.Millisecond)
	return logs, seen
}

func (s *AgentSuite) TestNativeSpeechForACallTheCallerLeftIsNoFailure() {
	logs, seen := s.speaksNativelyInto(errLeftByTheCaller)

	s.Empty(logs.records("agent failure"))
	s.Zero(countOf[Error](seen.seen()))
}

// The control for the speech-to-speech reply.
func (s *AgentSuite) TestNativeSpeechAnEdgeCouldNotPublishIsAFailure() {
	logs, seen := s.speaksNativelyInto(errTrackBroke)

	s.Len(logs.records("agent failure"), 2)
	s.Equal(2, countOf[Error](seen.seen()))
}
