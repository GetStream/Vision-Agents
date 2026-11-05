package agent

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// turnLatencyLegs are the legs of router_agent_turn_latency_seconds, each with the Turn field
// it is read from.
var turnLatencyLegs = map[string]func(Turn) float64{
	"roundtrip":             func(turn Turn) float64 { return turn.RoundtripMs },
	"speech_end_to_audio":   func(turn Turn) float64 { return turn.SpeechEndToAudioMs },
	"speech_end_to_audible": func(turn Turn) float64 { return turn.SpeechEndToAudibleMs },
}

// latencySeries is one series of router_agent_turn_latency_seconds: how many turns it has seen
// and the seconds they add up to.
type latencySeries struct {
	count uint64
	sum   float64
}

// turnLatencies reads every leg of one pipeline's turn latency. The labels are a fixed set
// shared by every test of the package, so a test compares what it reads before and after
// its own turn rather than reading a total.
func turnLatencies(pipeline string) map[string]latencySeries {
	read := map[string]latencySeries{}
	for leg := range turnLatencyLegs {
		var metric dto.Metric
		if err := turnLatency.WithLabelValues(leg, pipeline).(prometheus.Histogram).Write(&metric); err != nil {
			panic(err)
		}
		read[leg] = latencySeries{count: metric.GetHistogram().GetSampleCount(), sum: metric.GetHistogram().GetSampleSum()}
	}
	return read
}

// observedSince is what each leg of one pipeline gained since before was read.
func observedSince(pipeline string, before map[string]latencySeries) map[string]latencySeries {
	gained := map[string]latencySeries{}
	for leg, now := range turnLatencies(pipeline) {
		gained[leg] = latencySeries{count: now.count - before[leg].count, sum: now.sum - before[leg].sum}
	}
	return gained
}

// Catches a leg read from the wrong field of the turn, or a turn never observed.
func (s *AgentSuite) TestACallerTurnIsObservedAtEachOfItsLatencies() {
	s.edgeFactory = func(base *loopbackEdge) Edge {
		return &markedEdge{loopbackEdge: base, trackDelay: 40 * time.Millisecond}
	}
	s.join(true)
	before, native := turnLatencies(pipelineCascade), turnLatencies(pipelineNative)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.saysAfter(participant, "hello", 300)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Require().Greater(turn.SpeechEndToAudibleMs, turn.SpeechEndToAudioMs, "the legs must differ to tell them apart")
	s.Require().Greater(turn.SpeechEndToAudioMs, turn.RoundtripMs)
	gained := observedSince(pipelineCascade, before)
	for leg, field := range turnLatencyLegs {
		s.Equal(uint64(1), gained[leg].count, leg)
		s.InDelta(field(turn)/1000, gained[leg].sum, 1e-9, leg)
	}
	for leg, series := range observedSince(pipelineNative, native) {
		s.Zero(series.count, "a cascade's turn was counted as native: "+leg)
	}
	s.Contains(turnLatencyBounds(), 1.2, "turns slower than Twilio's slow-turn line cannot be counted")
}

// turnLatencyBounds are the upper bounds of the histogram's buckets, as /metrics shows them.
func turnLatencyBounds() []float64 {
	var metric dto.Metric
	if err := turnLatency.WithLabelValues("roundtrip", pipelineCascade).(prometheus.Histogram).Write(&metric); err != nil {
		panic(err)
	}
	var bounds []float64
	for _, bucket := range metric.GetHistogram().GetBucket() {
		bounds = append(bounds, bucket.GetUpperBound())
	}
	return bounds
}

// Catches a leg the turn never reached counted as an instant reply.
func (s *AgentSuite) TestALegATurnNeverReachedIsNotObserved() {
	s.join(true)
	before := turnLatencies(pipelineCascade)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.saysAfter(participant, "hello", 30)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Require().Zero(turn.SpeechEndToAudibleMs, "this edge never says when the track took the speech")
	gained := observedSince(pipelineCascade, before)
	s.Zero(gained["speech_end_to_audible"].count)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches a reply interrupted before any of it went out counted as an instant reply.
func (s *AgentSuite) TestAReplyInterruptedBeforeItsAudioIsNotObserved() {
	s.join(true)
	before := turnLatencies(pipelineCascade)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "no turn was started")

	s.says(participant, "actually, stop")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the interrupted turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Require().True(turn.Interrupted)
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// Catches a tool's answer timed as if the caller had waited on it since they stopped: its
// clock starts when the tool comes back.
func (s *AgentSuite) TestATurnTheAgentStartedIsNotObserved() {
	before := turnLatencies(pipelineCascade)
	s.waitsOnATool()

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 2 }, "both turns were never measured")
	caller := s.callerTurn()
	var callerWait, toolWait float64
	for _, event := range s.reported() {
		if turn, ok := event.(Turn); ok && turn.TurnID == caller {
			callerWait = turn.RoundtripMs
		} else if ok {
			toolWait = turn.RoundtripMs
		}
	}
	s.Require().Positive(toolWait, "the tool's answer was heard, so it has a wait to leave out")
	gained := observedSince(pipelineCascade, before)
	var callerTurns uint64
	if callerWait > 0 {
		callerTurns = 1
	}
	s.Equal(callerTurns, gained["roundtrip"].count)
	s.InDelta(callerWait/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches a speech-to-speech model's turn labelled as a cascade's.
func (s *AgentSuite) TestANativeTurnIsObservedAsNative() {
	before, cascade := turnLatencies(pipelineNative), turnLatencies(pipelineCascade)
	stopped := time.Now().Add(-50 * time.Millisecond)

	events := s.replayNative(
		sts.SpeechStarted{At: stopped.Add(-time.Second)},
		sts.SpeechStopped{At: stopped},
		sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()},
		sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: nativePCM},
		sts.ResponseComplete{ResponseID: "r1", Generation: 1},
	)

	turn, ok := turnFor(events, "r1")
	s.Require().True(ok, "the caller was waiting on this reply")
	gained := observedSince(pipelineNative, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
	for leg, series := range observedSince(pipelineCascade, cascade) {
		s.Zero(series.count, "a native turn was counted as a cascade's: "+leg)
	}
}
