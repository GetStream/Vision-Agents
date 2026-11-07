package agent

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
		if err := turnLatencySeconds.WithLabelValues(leg, pipeline).(prometheus.Histogram).Write(&metric); err != nil {
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
	if err := turnLatencySeconds.WithLabelValues("roundtrip", pipelineCascade).(prometheus.Histogram).Write(&metric); err != nil {
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

// Catches a reply interrupted after its first audio went out dropped from the metric: the
// caller did wait that long for it.
func (s *AgentSuite) TestAReplyInterruptedAfterItsAudioIsStillObserved() {
	s.join(true)
	s.edge.holdSpeech(true)
	s.model.reply = nil
	s.model.then = []string{"I heard you."}
	before := turnLatencies(pipelineCascade)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool {
		requests := s.model.requests()
		return len(requests) == 1 && s.model.script(requests[0].ID) != nil
	}, "the first reply never started")
	turnID := s.model.requests()[0].ID
	s.model.writes(turnID, "The old reply is still in flight.")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the first audio never went out")

	s.says(participant, "actually, stop")

	// The caller's second words start a reply of their own, which is observed too.
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 2 }, "both turns were never reported")
	var interrupted Turn
	var seconds float64
	for _, event := range s.reported() {
		if turn, ok := event.(Turn); ok {
			seconds += turn.RoundtripMs / 1000
			if turn.TurnID == turnID {
				interrupted = turn
			}
		}
	}
	s.Require().True(interrupted.Interrupted)
	s.Require().Positive(interrupted.RoundtripMs, "the first audio went out before the interruption")
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(2), gained["roundtrip"].count)
	s.InDelta(seconds, gained["roundtrip"].sum, 1e-9)
}

// Catches a turn labelled with the pipeline it finished on, not the one it started on: a
// settings change moves the agent onto a speech-to-speech model while the turn waits out its
// playout grace.
func (s *AgentSuite) TestATurnIsObservedOnThePipelineItStartedOn() {
	s.edgeFactory = func(base *loopbackEdge) Edge {
		return &markedEdge{loopbackEdge: base, neverTakes: true}
	}
	w := s.joinSwappable()
	before, native := turnLatencies(pipelineCascade), turnLatencies(pipelineNative)
	alice := stt.Participant{ID: "alice"}
	s.hears(w, alice, "hello", 1)
	s.Require().Zero(countOf[Turn](s.reported()), "the turn must still be open when the pipeline moves")

	moved := cascadeOnStub
	moved.STSTarget = "openai/gpt-realtime-2"
	s.Require().NoError(s.agent.SetSettings(s.ctx, moved))

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
	for leg, series := range observedSince(pipelineNative, native) {
		s.Zero(series.count, "a cascade's turn was counted as native after the move: "+leg)
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
	s.Require().Positive(callerWait, "the hold line said for the caller's turn ended their wait")
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
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

// subagentHeld is how long a test keeps the subagent's answer back once the reply that handed
// the work over has finished, so a turn timed to that answer is at least this slow.
const subagentHeld = 150 * time.Millisecond

// handsOverInSilence has the caller ask, the model hand the work to the subagent without a
// word, and the subagent kept from answering until the test releases it.
func (s *AgentSuite) handsOverInSilence() (participant stt.Participant, release func()) {
	s.delegates()
	held := make(chan struct{})
	s.subagent.holdCreate = held
	s.join(true)
	s.model.reply = []string{`<ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	participant = stt.Participant{ID: "alice"}
	s.speak(participant)
	release = sync.OnceFunc(func() { close(held) })
	s.T().Cleanup(release)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply that handed it over never finished")
	return participant, release
}

// Catches a caller turn answered through the subagent left out of the metric: its own reply
// said nothing, so the caller waited until the answer the subagent came back with began.
func (s *AgentSuite) TestACallerTurnAnsweredThroughADelegateIsObservedAtTheAnswer() {
	before := turnLatencies(pipelineCascade)
	_, release := s.handsOverInSilence()
	s.Require().Zero(countOf[Turn](s.reported()), "the turn closed before the answer it waits on")

	time.AfterFunc(subagentHeld, release)
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the answer was never spoken")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the caller's turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(s.callerTurn(), turn.TurnID)
	s.GreaterOrEqual(turn.RoundtripMs, float64(subagentHeld.Milliseconds()), "timed before the answer began")
	s.InDelta(turn.RoundtripMs+turn.STTLatencyMs, turn.SpeechEndToAudioMs, 1e-9)
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
	s.Equal(uint64(1), gained["speech_end_to_audio"].count)
	s.InDelta(turn.SpeechEndToAudioMs/1000, gained["speech_end_to_audio"].sum, 1e-9)
}

// Catches the delegated answer counted a second time: a filler in the turn's own reply ended
// the caller's wait, so the turn is observed once, at the filler, before the answer arrives.
func (s *AgentSuite) TestADelegatedTurnWithAFillerIsObservedOnceAtTheFiller() {
	s.delegates()
	held := make(chan struct{})
	s.subagent.holdCreate = held
	s.join(true)
	s.model.reply = []string{`One moment. <ask skill="think">15% of 84.20</ask>`}
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	before := turnLatencies(pipelineCascade)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "what is 15% of 84.20")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn the filler answered was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Require().Positive(turn.RoundtripMs)
	close(held)
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the delegated answer was never spoken")
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["speech_end_to_audio"].count)
	s.InDelta(turn.SpeechEndToAudioMs/1000, gained["speech_end_to_audio"].sum, 1e-9)
	s.Equal(uint64(1), gained["roundtrip"].count)
}

// Control: a greeting the agent says on its own answers nobody's silence, so it adds no
// observation.
func (s *AgentSuite) TestAGreetingAddsNoObservation() {
	s.join(true)
	before := turnLatencies(pipelineCascade)

	s.Require().NoError(s.agent.Say(s.ctx, "Hello, how can I help?"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the greeting was never spoken")
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// Catches a line the agent says on its own taken for the answer a caller turn waits on.
func (s *AgentSuite) TestALineSaidOnItsOwnDoesNotEndTheWaitForADelegatedAnswer() {
	before := turnLatencies(pipelineCascade)
	_, release := s.handsOverInSilence()

	s.Require().NoError(s.agent.Say(s.ctx, "Bear with me."))
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the line was never spoken")
	s.Zero(countOf[Turn](s.reported()), "the caller's turn was timed to a line nobody said for it")
	time.AfterFunc(subagentHeld, release)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the caller's turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.GreaterOrEqual(turn.RoundtripMs, float64(subagentHeld.Milliseconds()), "timed to the line said on its own")
	s.Equal(uint64(1), observedSince(pipelineCascade, before)["roundtrip"].count)
}

// Catches a caller turn left waiting, or timed to an answer, once the caller spoke again
// before hearing one: what is said next answers the newer turn.
func (s *AgentSuite) TestACallerTurnSpokenOverBeforeItsAnswerIsNotObserved() {
	before := turnLatencies(pipelineCascade)
	participant, _ := s.handsOverInSilence()
	first := s.callerTurn()

	s.says(participant, "and what is the time")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 2 }, "both turns were never reported")
	var unanswered, answered Turn
	for _, event := range s.reported() {
		if turn, ok := event.(Turn); ok && turn.TurnID == first {
			unanswered = turn
		} else if ok {
			answered = turn
		}
	}
	s.Equal(first, unanswered.TurnID)
	s.Zero(unanswered.RoundtripMs)
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(answered.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches a caller turn still waiting for its answer lost when the call ends: it is reported,
// without a wait, as the base reported it when its reply finished.
func (s *AgentSuite) TestACallerTurnStillWaitingWhenTheCallEndsIsReportedUnobserved() {
	before := turnLatencies(pipelineCascade)
	s.handsOverInSilence()
	turnID := s.callerTurn()

	s.Require().NoError(s.agent.Close())

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was lost with the call")
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(turnID, turn.TurnID)
	s.Zero(turn.RoundtripMs)
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// Catches delegated work that came back before the reply that handed it over finished: the
// caller still waits for the answer, which is said once the reply is done.
func (s *AgentSuite) TestATurnWhoseDelegateAnsweredFirstIsObservedAtTheAnswer() {
	s.delegates()
	s.join(true)
	s.model.reply = nil
	s.model.then = []string{"That comes to 12.63."}
	s.subagent.reply = []string{"It is 12.63."}
	before := turnLatencies(pipelineCascade)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "what is 15% of 84.20")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply never started")
	turnID := s.model.requests()[0].ID
	s.model.writes(turnID, `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return countOf[TaskSettled](s.reported()) == 1 }, "the task never settled")

	s.model.finishes(turnID)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the caller's turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Require().Positive(turn.RoundtripMs, "the answer said after the reply ended the wait")
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches a written turn that handed work over held back for an answer it can never hear:
// in writing it is reported when its reply finishes, as before.
func (s *AgentSuite) TestAWrittenTurnThatHandsWorkOverIsReportedAtOnce() {
	s.delegates()
	held := make(chan struct{})
	s.subagent.holdCreate = held
	defer close(held)
	s.joinText()
	s.model.reply = []string{`<ask skill="think">15% of 84.20</ask>`}

	_, err := s.agent.RespondTo(s.ctx, "what is 15% of 84.20", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "nothing was handed over")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the written turn waited on the subagent")
}

// Catches a caller turn that said nothing and handed nothing on left waiting for a line that
// is not coming.
func (s *AgentSuite) TestACallerTurnThatSaidNothingAndHandedNothingOnIsReportedAtOnce() {
	s.join(true)
	s.model.reply = []string{""}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hmm")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.Zero(turn.RoundtripMs)
}

// toolHeldInSilence has the caller ask for an order and the model reach for a tool without a
// word, with the tool kept from answering until the test releases it. hold is the line the
// model writes while the tool runs, none when empty.
func (s *AgentSuite) toolHeldInSilence(hold string) (saidAt time.Time, release func()) {
	s.ownsTools("")
	s.join(false)
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	release = sync.OnceFunc(func() { close(runner.release) })
	s.T().Cleanup(release)
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	if hold != "" {
		s.model.composes = []string{hold}
	}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	saidAt = time.Now()
	s.says(participant, "where is my order")

	<-runner.began
	return saidAt, release
}

// Catches a caller turn that only called a tool timed past the hold line said while the tool
// ran: that line is the first the caller heard.
func (s *AgentSuite) TestACallerTurnThatOnlyCalledAToolIsObservedAtTheHoldLine() {
	before := turnLatencies(pipelineCascade)
	saidAt, release := s.toolHeldInSilence("Pulling up order twelve now.")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the hold line was never said")
	heldAt := time.Now()

	time.AfterFunc(subagentHeld, release)

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the answer never followed the hold line")
	turn, ok := turnFor(s.reported(), s.callerTurn())
	s.Require().True(ok, "the caller's turn was never reported")
	s.Positive(turn.RoundtripMs)
	s.LessOrEqual(turn.RoundtripMs, msBetween(saidAt, heldAt), "timed past the hold line")
	s.Equal(uint64(1), observedSince(pipelineCascade, before)["roundtrip"].count)
}

// Catches a caller turn that only called a tool, with no hold line, left out of the metric:
// the tool's answer is the first the caller heard.
func (s *AgentSuite) TestACallerTurnThatOnlyCalledAToolIsObservedAtTheToolsAnswer() {
	before := turnLatencies(pipelineCascade)
	_, release := s.toolHeldInSilence("")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply that called the tool never finished")
	s.Require().Zero(countOf[Turn](s.reported()), "the turn closed before the answer it waits on")

	time.AfterFunc(subagentHeld, release)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 2 }, "both turns were never reported")
	turn, ok := turnFor(s.reported(), s.callerTurn())
	s.Require().True(ok)
	s.GreaterOrEqual(turn.RoundtripMs, float64(subagentHeld.Milliseconds()), "timed before the tool's answer began")
	gained := observedSince(pipelineCascade, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches an update said while delegated work runs not taken for the end of the caller's
// wait: it is the first line the caller heard.
func (s *AgentSuite) TestADelegatedTurnIsObservedAtTheUpdateSaidWhileItRuns() {
	s.duplex = DuplexOptions{updateGaps: []time.Duration{10 * time.Millisecond}}
	before := turnLatencies(pipelineCascade)
	s.delegates()
	held := make(chan struct{})
	s.subagent.holdCreate = held
	defer close(held)
	s.join(true)
	s.model.reply = []string{`<ask skill="think">work through the itinerary</ask>`}
	s.model.composes = []string{"Still mapping out your itinerary."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please work through the itinerary")

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the update never ended the wait")
	s.Zero(countOf[TaskSettled](s.reported()), "the work was still running")
	turn, _ := firstOf[Turn](s.reported())
	s.Positive(turn.RoundtripMs)
	s.Equal(uint64(1), observedSince(pipelineCascade, before)["roundtrip"].count)
}

// Catches a reply read from a tool that calls another without a word held back like a caller's
// turn: only a caller waits from when they stopped, so it is reported when it is done.
func (s *AgentSuite) TestAToolsReplyThatCallsAnotherToolIsReportedWhenDone() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{}
	s.model.replyOn = map[int][]string{2: {}}
	s.model.then = []string{"It ships tomorrow."}
	s.model.keepCalling = true
	s.model.callsUntil = 3
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the second tool's answer was never read")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 3 }, "a turn of the chain was held back")
}

// Catches a scheduled task's turn held back for the line said for it, like a caller's: nobody
// in the call waited on it, so it is reported when its reply is done, as before (AI-2969 R1.1).
func (s *AgentSuite) TestAScheduledTaskThatOnlyCallsAToolIsNotObserved() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.replyOn = map[int][]string{1: {"Hello."}, 2: {}}
	s.model.then = []string{"Your order ships tomorrow."}
	s.model.callsOn = 2
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "hello")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the caller's turn was never reported")
	before := turnLatencies(pipelineCascade)

	scheduled, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Tell the caller where order 12 is", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return countOf[Turn](s.reported()) == 3 }, "the task's turn and the tool's answer were never reported")
	s.eventually(func() bool { return s.spokenText("Your order ships tomorrow.") }, "the tool's answer was never said")
	turn, ok := turnFor(s.reported(), scheduled)
	s.Require().True(ok, "the task's turn was never reported")
	s.Zero(turn.RoundtripMs, "the task's turn was timed to the tool's answer")
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// Catches a scheduled task's turn that hands work to the subagent timed to the answer it came
// back with (AI-2969 R1.1).
func (s *AgentSuite) TestAScheduledTaskThatHandsWorkToTheSubagentIsNotObserved() {
	s.delegates()
	s.join(true)
	s.model.reply = []string{`<ask skill="think">order 12</ask>`}
	s.model.then = []string{"Your order ships tomorrow."}
	s.subagent.reply = []string{"It ships tomorrow."}
	before := turnLatencies(pipelineCascade)

	scheduled, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind the caller about order 12", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return s.spokenText("Your order ships tomorrow.") }, "the subagent's answer was never said")
	s.eventually(func() bool { _, ok := turnFor(s.reported(), scheduled); return ok }, "the task's turn was never reported")
	turn, _ := turnFor(s.reported(), scheduled)
	s.Zero(turn.RoundtripMs, "the task's turn was timed to the subagent's answer")
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// Catches a scheduled task's turn answered with words observed: a turn the agent starts on its
// own answers nobody's silence. Its turn event keeps its timings (AI-2969 R1.5).
func (s *AgentSuite) TestAScheduledTaskAnsweredWithWordsIsNotObserved() {
	s.join(true)
	s.model.reply = []string{"Your order ships tomorrow."}
	before := turnLatencies(pipelineCascade)

	scheduled, err := s.agent.RespondTo(routing.WithScheduledTurn(s.ctx), "Remind the caller about order 12", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { _, ok := turnFor(s.reported(), scheduled); return ok }, "the task's turn was never reported")
	turn, _ := turnFor(s.reported(), scheduled)
	s.Positive(turn.RoundtripMs, "the task's turn event lost its wait")
	for leg, series := range observedSince(pipelineCascade, before) {
		s.Zero(series.count, leg)
	}
}

// How far the work a native caller turn hands to the subagent has gone when its reply completes.
const (
	handOverRunning  = iota // the delegate_skill call has not returned
	handOverReturned        // the call returned and the subagent is working
	handOverFinished        // the subagent finished and its findings wait to be said
)

// delegatesInSilence has a native caller turn r1 hand work to the subagent, saying something
// first when spoken is set, and has the reply r2 after the hand-over say a line.
func (s *AgentSuite) delegatesInSilence(spoken bool, handOver int) []Event {
	peer := s.joinNativeDelegating()
	s.nativeFrame(peer, "session.update")
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	if spoken {
		s.nativeSend(peer, `{"type":"response.output_audio.delta","response_id":"r1","delta":"AQACAA=="}`)
	}
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	if handOver != handOverRunning {
		s.nativeFrame(peer, "conversation.item.create")
		s.eventually(func() bool {
			s.agent.mu.Lock()
			defer s.agent.mu.Unlock()
			return s.agent.pendingTools == 0
		}, "the delegate_skill call never returned")
	}
	if handOver == handOverFinished {
		s.eventually(func() bool { return len(s.subagent.requests()) == 1 }, "the subagent was never asked")
		request := s.subagent.requests()[0]
		s.subagent.writes(request.ID, "The total is 42 dollars.")
		s.subagent.finishes(request.ID)
		s.eventually(func() bool { return s.agent.harness.Pending() && !s.agent.harness.Delegating() }, "the findings never came back")
	}
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.output_audio.delta","response_id":"r2","delta":"AQACAA=="}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "both replies were never spoken")
	s.eventually(func() bool { _, ok := turnFor(s.reported(), "r1"); return ok }, "the caller's turn was never reported")
	return s.reported()
}

// Catches a native caller turn that handed work to the subagent without a word reported at once
// with no wait, and the reply after the hand-over left without a turn to time, whether the
// hand-over is still running, returned or finished when the turn's reply completes (AI-2969 R1.2).
func (s *AgentSuite) TestANativeCallerTurnAnsweredThroughADelegateIsObservedAtTheReplyAfter() {
	for name, handOver := range map[string]int{"running": handOverRunning, "returned": handOverReturned, "finished": handOverFinished} {
		s.Run(name, func() {
			before := turnLatencies(pipelineNative)

			events := s.delegatesInSilence(false, handOver)

			turn, _ := turnFor(events, "r1")
			s.Positive(turn.RoundtripMs, "the caller's turn was never timed to the reply after the hand-over")
			gained := observedSince(pipelineNative, before)
			s.Equal(uint64(1), gained["roundtrip"].count)
			s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
			_, measured := turnFor(events, "r2")
			s.False(measured, "the reply after the hand-over answers nobody's silence of its own")
			s.Equal(kindFollowUp, s.agent.turns.originOf("r2").kind, "the reply delivering the subagent's work is a follow-up, as on the cascade")
		})
	}
}

// typedWhileOwed has a native caller turn r1 hand work to the subagent without a word, then a
// turn typed in with ctx, before r1 completes when early is set and after it otherwise, and the
// reply r2 to the typed turn say a line while the work still runs.
func (s *AgentSuite) typedWhileOwed(ctx context.Context, early bool) {
	peer := s.joinNativeDelegating()
	s.nativeFrame(peer, "session.update")
	s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
	s.heardSpeechStart()
	s.stoppedSpeaking(peer)
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
	s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"delegate_skill","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
	typed := func() {
		_, err := s.agent.RespondTo(ctx, "Remind me to call the venue", nil, options.LLM{})
		s.Require().NoError(err)
		s.nativeFrame(peer, "response.create")
	}
	if early {
		s.eventually(func() bool { return s.agent.turns.originOf("r1").kind == kindCaller }, "the caller's turn never started")
		typed()
	}
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller's reply never completed")
	if !early {
		typed()
	}
	s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
	s.nativeSend(peer, `{"type":"response.output_audio.delta","response_id":"r2","delta":"AQACAA=="}`)
	s.nativeSend(peer, `{"type":"response.done","response":{"id":"r2","status":"completed"}}`)
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the typed turn's reply was never spoken")
}

// Catches the reply to a scheduled reminder, or to a message typed in, ending the wait of a
// native caller turn still waiting on the subagent: it answers the typed turn, so the caller
// turn is reported at once without a wait, as the cascade reports it (AI-2969 R2.1).
func (s *AgentSuite) TestANativeCallerTurnWaitingOnTheSubagentIsNotAnsweredByATypedTurnsReply() {
	for name, ctx := range map[string]context.Context{"scheduled": routing.WithScheduledTurn(s.ctx), "typed": s.ctx} {
		s.Run(name, func() {
			before := turnLatencies(pipelineNative)

			s.typedWhileOwed(ctx, false)

			s.eventually(func() bool { _, ok := turnFor(s.reported(), "r1"); return ok }, "the caller's turn was held past the typed turn")
			turn, _ := turnFor(s.reported(), "r1")
			s.Zero(turn.RoundtripMs, "the caller's turn was timed to the typed turn's reply")
			for leg, series := range observedSince(pipelineNative, before) {
				s.Zero(series.count, leg)
			}
		})
	}
}

// Catches the reply to a turn typed in while a native caller turn was still being said ending
// that turn's wait once it hands work on without a word (AI-2969 R2.1).
func (s *AgentSuite) TestANativeCallerTurnTypedOverBeforeItHandsWorkOnIsNotAnsweredByTheTypedTurnsReply() {
	before := turnLatencies(pipelineNative)

	s.typedWhileOwed(s.ctx, true)

	for leg, series := range observedSince(pipelineNative, before) {
		s.Zero(series.count, leg)
	}
}

// Catches the reply that answers a native caller turn waiting on a tool recorded as the wrong
// work: a tool's answer for a tool, a follow-up for a delegate_skill call still running, as the
// cascade names the same lines (AI-2969 R2.2).
func (s *AgentSuite) TestANativeReplyAnsweringACallStillRunningIsRecordedAsThatCallsWork() {
	for tool, want := range map[string]string{"lookup_order": kindTool, delegateSkill: kindFollowUp} {
		s.Run(tool, func() {
			held := newHeldStart()
			defer held.let()
			peer := s.joinNativeWith(func(options *Options) { options.OnToolStarted = held.hold })
			s.nativeFrame(peer, "session.update")
			s.nativeSend(peer, `{"type":"input_audio_buffer.speech_started"}`)
			s.heardSpeechStart()
			s.stoppedSpeaking(peer)
			s.nativeSend(peer, `{"type":"response.created","response":{"id":"r1"}}`)
			s.nativeSend(peer, `{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"call1","name":"`+tool+`","arguments":"{\"skill\":\"think\",\"prompt\":\"Work out the booking total\"}"}`)
			select {
			case <-held.started:
			case <-time.After(settleFor):
				s.FailNow("the tool call never started")
			}
			s.nativeSend(peer, `{"type":"response.done","response":{"id":"r1","status":"completed"}}`)
			s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the caller's reply never completed")
			s.nativeSend(peer, `{"type":"response.created","response":{"id":"r2"}}`)
			s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 }, "the reply after the call never began")

			s.Equal(want, s.agent.turns.originOf("r2").kind)
		})
	}
}

// Control: a native caller turn that said something before it handed work on is observed once,
// at its own audio, as on the base.
func (s *AgentSuite) TestANativeCallerTurnThatSpokeBeforeItDelegatedIsObservedOnceAtItsOwnAudio() {
	before := turnLatencies(pipelineNative)

	events := s.delegatesInSilence(true, handOverRunning)

	turn, _ := turnFor(events, "r1")
	s.Require().Positive(turn.RoundtripMs)
	gained := observedSince(pipelineNative, before)
	s.Equal(uint64(1), gained["roundtrip"].count)
	s.InDelta(turn.RoundtripMs/1000, gained["roundtrip"].sum, 1e-9)
}

// Catches an update said while the turn waits out its playout grace after a filler moving the
// turn's first audio from the filler to the update (AI-2969 R1.4).
func (s *AgentSuite) TestAnUpdateSaidWhileTheFillerWaitsOnTheTrackKeepsTheFillersWait() {
	s.edgeFactory = func(base *loopbackEdge) Edge { return &markedEdge{loopbackEdge: base, neverTakes: true} }
	s.duplex = DuplexOptions{updateGaps: []time.Duration{10 * time.Millisecond}}
	s.delegates()
	held := make(chan struct{})
	s.subagent.holdCreate = held
	defer close(held)
	s.join(true)
	s.model.reply = []string{`One moment. <ask skill="think">work through the itinerary</ask>`}
	s.model.composes = []string{"Still mapping out your itinerary."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	saidAt := time.Now()
	before := turnLatencies(pipelineCascade)

	s.says(participant, "please work through the itinerary")

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the filler never went out")
	fillerAt := time.Now()
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the update never went out")
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn never reported")
	turn, _ := firstOf[Turn](s.reported())
	s.LessOrEqual(turn.RoundtripMs, msBetween(saidAt, fillerAt), "timed to the update, past the filler")
	s.Equal(uint64(1), observedSince(pipelineCascade, before)["roundtrip"].count)
}
