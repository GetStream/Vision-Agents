package agent

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// markedEdge is a loopback edge with a track of its own: publishing queues the speech, and
// the track takes it a moment later, or never.
type markedEdge struct {
	*loopbackEdge
	trackDelay time.Duration
	neverTakes bool

	mu     sync.Mutex
	marked int
}

func (e *markedEdge) PublishAudioMarked(_ context.Context, pcm audio.PcmData, marks PlayoutMarks) error {
	if err := e.loopbackEdge.PublishAudio(pcm); err != nil {
		return err
	}
	if marks == nil {
		return nil
	}
	e.mu.Lock()
	e.marked++
	e.mu.Unlock()
	marks.FirstFrameQueued(time.Now())
	if !e.neverTakes {
		time.AfterFunc(e.trackDelay, func() { marks.FirstAudiblePulled(time.Now()) })
	}
	return nil
}

func (e *markedEdge) markedWrites() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.marked
}

func (s *AgentSuite) TestATurnIsMeasuredToWhenTheTrackTookTheFirstFrame() {
	var edge *markedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &markedEdge{loopbackEdge: base, trackDelay: 120 * time.Millisecond}
		return edge
	}
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Positive(edge.markedWrites(), "the edge was never asked to report the speech")
	s.Positive(turn.FirstFrameQueuedMs, "the first frame was never reported as queued")
	s.Greater(turn.FirstAudibleFrameMs, turn.FirstFrameQueuedMs+60,
		"the turn closed before the track took its speech, so it lost the moment")
	s.LessOrEqual(turn.FirstFrameQueuedMs, turn.RoundtripMs, "queued before publishing returned")
	s.Positive(turn.SpeechEndToAudibleMs)
	s.Positive(turn.RoundtripMs, "the figure taken at the return is kept beside them")
}

func (s *AgentSuite) TestATurnIsStillReportedWhenTheTrackNeverTakesItsSpeech() {
	s.edgeFactory = func(base *loopbackEdge) Edge {
		return &markedEdge{loopbackEdge: base, neverTakes: true}
	}
	s.join(true)
	s.agent.turns.mu.Lock()
	s.agent.turns.grace = 50 * time.Millisecond
	s.agent.turns.mu.Unlock()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Positive(turn.FirstFrameQueuedMs)
	s.Zero(turn.FirstAudibleFrameMs, "the moment that never happened is left out")
}

func (s *AgentSuite) TestAnEdgeThatReportsNothingStillGetsItsTurnsMeasured() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Positive(turn.RoundtripMs)
	s.Zero(turn.FirstFrameQueuedMs)
	s.Zero(turn.FirstAudibleFrameMs)
}

func (s *AgentSuite) TestANativeTurnIsMeasuredToWhenTheTrackTookTheFirstFrame() {
	edge := &markedEdge{loopbackEdge: newLoopbackEdge(), trackDelay: 80 * time.Millisecond}
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
	go agent.consumeSTS(agent.pipe, ordered)
	s.T().Cleanup(func() { close(source) })
	// Somebody stopped talking just now, so the reply that follows answers their silence.
	agent.mu.Lock()
	agent.waitingSince = time.Now()
	agent.mu.Unlock()

	pcm := audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}
	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1}

	s.Eventually(func() bool { return countOf[Turn](seen.seen()) == 1 }, 2*time.Second, time.Millisecond)
	turn, _ := firstOf[Turn](seen.seen())
	s.Positive(turn.FirstFrameQueuedMs)
	s.Greater(turn.FirstAudibleFrameMs, turn.FirstFrameQueuedMs+40,
		"the turn closed before the track took its speech")
}
