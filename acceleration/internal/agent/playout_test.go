package agent

import (
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// finishedSpeaking is the message logged when an utterance completes.
const finishedSpeaking = "finished speaking"

// countedEdge is a loopback edge with an outgoing queue a test drains by hand: publishing
// queues a frame for every 20 ms, and nothing leaves until the test takes or drops it.
type countedEdge struct {
	*loopbackEdge

	mu      sync.Mutex
	queued  uint64
	sent    uint64
	dropped uint64
	waiting int
	peak    int
	reads   int
}

func (e *countedEdge) PublishAudio(pcm audio.PcmData) error {
	e.mu.Lock()
	frames := int(pcm.DurationMs() / 20)
	e.queued += uint64(frames)
	e.waiting += frames
	e.peak = max(e.peak, e.waiting)
	e.mu.Unlock()
	return e.loopbackEdge.PublishAudio(pcm)
}

func (e *countedEdge) PlayoutStats() PlayoutStats {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.reads++
	stats := PlayoutStats{
		Queued: e.queued, Sent: e.sent, Dropped: e.dropped,
		Waiting: time.Duration(e.waiting) * 20 * time.Millisecond,
		Peak:    time.Duration(e.peak) * 20 * time.Millisecond,
	}
	e.peak = e.waiting
	return stats
}

// take has the track take n frames.
func (e *countedEdge) take(n int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.sent += uint64(n)
	e.waiting -= n
}

// discard throws away every frame still waiting.
func (e *countedEdge) discard() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.dropped += uint64(e.waiting)
	e.waiting = 0
}

func (e *countedEdge) readings() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.reads
}

// joinCounted joins with a counted edge whose queue has already thrown dropped frames away,
// and waits for the voice's consumer to take its first reading.
func (s *AgentSuite) joinCounted(dropped uint64) *countedEdge {
	var edge *countedEdge
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge = &countedEdge{loopbackEdge: base, dropped: dropped}
		return edge
	}
	s.logs = &logLines{}
	s.join(true)
	s.eventually(func() bool { return edge.readings() == 1 }, "the queue was never read")
	return edge
}

// started has the voice start an utterance and waits until the queue was read for it.
func (s *AgentSuite) started(edge *countedEdge, synthesisID string) {
	reads := edge.readings()
	s.voice.emitter.Send(tts.SynthesisStarted{SynthesisID: synthesisID, Provider: "stub"})
	s.eventually(func() bool { return edge.readings() == reads+1 }, "the start was never read")
}

// published has the voice send a chunk of ms and waits until the edge has it.
func (s *AgentSuite) published(synthesisID string, ms int) {
	heard := len(s.edge.heard())
	s.voice.emitter.Send(chunkOf(synthesisID, 16*ms, 1000))
	s.eventually(func() bool { return len(s.edge.heard()) == heard+1 }, "the chunk was never published")
}

// finishLine is the finish line logged for a synthesis.
func (s *AgentSuite) finishLine(synthesisID string) map[string]any {
	for _, record := range s.logs.records(finishedSpeaking) {
		if record["synthesis"] == synthesisID {
			return record
		}
	}
	s.FailNow("no finish line", synthesisID)
	return nil
}

// peaksObserved is how many queue peaks were observed for the stub voice, and their sum.
func peaksObserved() (uint64, float64) {
	families, _ := prometheus.DefaultGatherer.Gather()
	for _, family := range families {
		if family.GetName() != "router_agent_playout_queue_peak_seconds" {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metric.GetLabel()[0].GetValue() == "stub" {
				return metric.GetHistogram().GetSampleCount(), metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return 0, 0
}

func (s *AgentSuite) TestTheFinishLineSaysWhatTheEdgeQueueDidWithTheUtterance() {
	edge := s.joinCounted(0)
	drops := testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))
	peaks, peakSum := peaksObserved()

	s.started(edge, "turn-a")
	s.published("turn-a", 100)
	edge.take(2)
	edge.discard()
	s.published("turn-a", 40)
	s.voiceSends(1, tts.SynthesisComplete{SynthesisID: "turn-a"})

	line := s.finishLine("turn-a")
	s.EqualValues(7, line["frames_queued"])
	s.EqualValues(2, line["frames_sent"])
	s.EqualValues(3, line["frames_dropped"])
	s.EqualValues(100, line["peak_queued_ms"], "five frames waited at once before any went")
	s.EqualValues(40, line["queued_ms"], "the last chunk was still waiting when the voice finished")
	s.Equal(3.0, testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))-drops)
	count, sum := peaksObserved()
	s.Equal(peaks+1, count)
	s.InDelta(0.1, sum-peakSum, 1e-9)
}

func (s *AgentSuite) TestThePeakReadAtAnotherUtterancesStartCountsForTheOneOpen() {
	edge := s.joinCounted(0)

	s.started(edge, "turn-a")
	s.published("turn-a", 120)
	edge.take(5)
	// The six frames turn-a queued are read here, and the queue is two frames deep at most
	// from now on.
	s.started(edge, "tool-b")
	s.published("tool-b", 20)
	s.voiceSends(1, tts.SynthesisComplete{SynthesisID: "turn-a"})
	s.voiceSends(2, tts.SynthesisComplete{SynthesisID: "tool-b"})

	first := s.finishLine("turn-a")
	s.EqualValues(7, first["frames_queued"], "the queue's counts include tool-b's frame")
	s.EqualValues(5, first["frames_sent"])
	s.EqualValues(120, first["peak_queued_ms"])
	s.EqualValues(40, first["queued_ms"])
	second := s.finishLine("tool-b")
	s.EqualValues(1, second["frames_queued"])
	s.EqualValues(0, second["frames_sent"])
	s.EqualValues(40, second["peak_queued_ms"], "tool-b queued behind the frame turn-a left")
}

func (s *AgentSuite) TestDropsCountedBeforeTheVoiceWasTakenAreNotCountedAgain() {
	edge := s.joinCounted(5)
	drops := testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))

	s.started(edge, "turn-a")
	s.published("turn-a", 20)
	s.voiceSends(1, tts.SynthesisComplete{SynthesisID: "turn-a"})

	s.EqualValues(0, s.finishLine("turn-a")["frames_dropped"])
	s.Equal(0.0, testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))-drops,
		"a voice swapped mid-call would count the last one's drops again")
}

func (s *AgentSuite) TestADropIsCountedOnceHoweverOftenTheQueueIsReadAfter() {
	edge := s.joinCounted(0)
	drops := testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))

	s.started(edge, "turn-a")
	s.published("turn-a", 60)
	edge.discard()
	s.voiceSends(1, tts.SynthesisComplete{SynthesisID: "turn-a"})
	s.started(edge, "turn-b")
	s.published("turn-b", 20)
	s.voiceSends(2, tts.SynthesisComplete{SynthesisID: "turn-b"})

	s.Equal(3.0, testutil.ToFloat64(playoutDroppedFrames.WithLabelValues("stub"))-drops)
}

func (s *AgentSuite) TestAnUtteranceNeverSeenToStartSaysNothingOfTheQueue() {
	s.joinCounted(0)

	s.published("turn-a", 20)
	s.voiceSends(1, tts.SynthesisComplete{SynthesisID: "turn-a"})

	s.NotContains(s.finishLine("turn-a"), "frames_queued", "there is no reading to count from")
}

func (s *AgentSuite) TestAnEdgeWithNoQueueSaysNothingOfIt() {
	s.logs = &logLines{}
	s.join(true)

	s.voiceSends(1,
		tts.SynthesisStarted{SynthesisID: "turn-a", Provider: "stub"},
		chunkOf("turn-a", 320, 1000),
		tts.SynthesisComplete{SynthesisID: "turn-a"},
	)

	line := s.finishLine("turn-a")
	s.NotContains(line, "frames_queued", "zeros would read as an empty queue")
	s.NotContains(line, "queued_ms")
}

// PlayoutSuite covers what reading the queue keeps between utterances.
type PlayoutSuite struct {
	suite.Suite
}

func TestPlayoutSuite(t *testing.T) {
	suite.Run(t, new(PlayoutSuite))
}

func (s *PlayoutSuite) TestAFinishedUtteranceIsForgotten() {
	queue := newPlayout(&countedEdge{loopbackEdge: newLoopbackEdge()}, "stub")
	queue.started("turn-a")

	s.NotEmpty(queue.finished("turn-a"))

	s.Empty(queue.open)
	s.Empty(queue.finished("turn-a"), "a second completion has no start to count from")
}
