package score

import (
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/world"
)

func TestPercentileAndPassK(t *testing.T) {
	m := Metrics{V2V: []Timing{{V2VMS: 400}, {V2VMS: 500}, {V2VMS: 2000}}}
	SummarizeTiming(&m)
	if m.V2VP50 != 500 {
		t.Fatalf("p50 %d", m.V2VP50)
	}
	if m.SpikeCount != 1 {
		t.Fatalf("spikes %d", m.SpikeCount)
	}
	if !PassAtK([]bool{false, true, false}) {
		t.Fatal("pass@k")
	}
	if PassHatK([]bool{true, false, true}) {
		t.Fatal("pass^k should fail")
	}
}

func TestEntityInSpeech(t *testing.T) {
	fails := EntityInSpeech("booked Saturday at seven thirty for Alvarez party of six", []scenario.Entity{
		{Name: "name", Value: "Alvarez", InSpeech: true},
		{Name: "time", Value: "7:30", InSpeech: true},
		{Name: "party", Value: "6", InSpeech: true},
		{Name: "missing", Value: "xyz", InSpeech: true},
	})
	if len(fails) != 1 {
		t.Fatalf("fails %v", fails)
	}
}

func TestBargeInStopMSUnmeasured(t *testing.T) {
	rec := caller.Result{Rate: audio.Rate, Agent: audio.Silence(audio.Rate)}
	got, reason := BargeInStopMS(rec)
	if got != -1 || reason != BargeNoEvent {
		t.Fatalf("no barge event: %d %q", got, reason)
	}
	rec.Events = []caller.Event{{
		BargeIn:    true,
		Kind:       scenario.TriggerBargeIn,
		RecStartMs: 400,
	}}
	got, reason = BargeInStopMS(rec)
	if got != -1 || reason != BargeAlreadyQuiet {
		t.Fatalf("no straddle: %d %q", got, reason)
	}
}

func TestBargeInStopMSNamesAReplyThatStartedAfterTheBarge(t *testing.T) {
	rate := audio.Rate
	agent := audio.Concat(audio.Silence(1500*rate/1000), audio.Tone(600*rate/1000, 220, 12000))
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{{
			BargeIn:    true,
			Kind:       scenario.TriggerBargeIn,
			RecStartMs: 400,
		}},
	}
	got, reason := BargeInStopMS(rec)
	if got != -1 || reason != BargeStartedAfter {
		t.Fatalf("an agent that answered late is not an agent with nothing to cut off: %d %q", got, reason)
	}
}

// A reply that stops promptly and is followed by the answer to the correction is two
// utterances. Reading the stop edge off spans merged at UtteranceMergeGapMS would join
// them and report the prompt stop as a reply that ran on past the limit.
func TestBargeInStopMSDoesNotRunOnIntoTheNextReply(t *testing.T) {
	rate := audio.Rate
	ms := func(n int) int { return n * rate / 1000 }
	agent := audio.Concat(
		audio.Silence(ms(200)),
		audio.Tone(ms(400), 220, 12000),
		audio.Silence(ms(600)),
		audio.Tone(ms(1500), 220, 12000),
	)
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{{
			BargeIn:    true,
			Kind:       scenario.TriggerBargeIn,
			RecStartMs: 400,
		}},
	}
	got, reason := BargeInStopMS(rec)
	if reason != BargeMeasured {
		t.Fatalf("reason %q", reason)
	}
	if got < 150 || got > 350 {
		t.Fatalf("stop %dms: the reply stopped 200ms after the barge", got)
	}
}

func TestBargeInStopMSMeasured(t *testing.T) {
	rate := audio.Rate
	agent := audio.Concat(audio.Silence(rate/5), audio.Tone(rate, 220, 12000))
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{{
			BargeIn:    true,
			Kind:       scenario.TriggerBargeIn,
			RecStartMs: 400,
		}},
	}
	got, reason := BargeInStopMS(rec)
	if got < 0 {
		t.Fatalf("want measured stop, got %d (%s)", got, reason)
	}
}

func TestBargeInStopMSToleratesFrameAlignment(t *testing.T) {
	rate := audio.Rate
	agent := audio.Concat(audio.Silence(450*rate/1000), audio.Tone(300*rate/1000, 220, 12000))
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{{
			BargeIn:    true,
			Kind:       scenario.TriggerBargeIn,
			RecStartMs: 400,
		}},
	}
	if got, _ := BargeInStopMS(rec); got < 300 || got > 400 {
		t.Fatalf("stop %dms", got)
	}
}

func TestApplyGates(t *testing.T) {
	m := Metrics{SelectivityHold: true, HoldThroughOverlap: true}
	ApplyGates(&m, scenario.Scenario{})
	if !m.Passed {
		t.Fatalf("notes %v", m.GateNotes)
	}
	m.EndStateFail = []string{"reservation.allergen"}
	ApplyGates(&m, scenario.Scenario{})
	if m.Passed {
		t.Fatal("expected fail")
	}
}

func TestApplyGatesToolOrderAndFiller(t *testing.T) {
	m := Metrics{SelectivityHold: true, HoldThroughOverlap: true, ToolOrderFail: []string{"walk_reboot"}, FillerFail: []string{"no filler"}}
	ApplyGates(&m, scenario.Scenario{})
	if m.Passed {
		t.Fatal("expected fail")
	}
	got := strings.Join(m.GateNotes, ",")
	if !strings.Contains(got, "tool_order") || !strings.Contains(got, "filler") {
		t.Fatalf("notes %v", m.GateNotes)
	}
}

func TestApplyGatesBargeIn(t *testing.T) {
	m := Metrics{SelectivityHold: true, HoldThroughOverlap: true, BargeInStopMS: 1200}
	ApplyGates(&m, scenario.Scenario{})
	if m.Passed {
		t.Fatal("expected barge_in fail")
	}
	m = Metrics{SelectivityHold: true, HoldThroughOverlap: true, BargeInStopMS: -1}
	ApplyGates(&m, scenario.Scenario{Turns: []scenario.Turn{{Trigger: scenario.Trigger{Kind: scenario.TriggerBargeIn}}}})
	if m.Passed {
		t.Fatal("expected unmeasured barge_in fail")
	}
}

func TestApplyGatesExpectedTools(t *testing.T) {
	m := Metrics{SelectivityHold: true, HoldThroughOverlap: true, ExpectedToolFail: []string{"create_order not called"}}
	ApplyGates(&m, scenario.Scenario{})
	if m.Passed || !strings.Contains(strings.Join(m.GateNotes, ","), "expected_tools") {
		t.Fatalf("notes %v", m.GateNotes)
	}
}

func TestSummarizeTimingSkipsToolTurns(t *testing.T) {
	m := Metrics{V2V: []Timing{
		{TurnID: "a", V2VMS: 400},
		{TurnID: "b", V2VMS: 500},
		{TurnID: "c", V2VMS: 600},
		{TurnID: "tool", V2VMS: 8000, Tool: true},
	}}
	SummarizeTiming(&m)
	if m.NonToolP50 != 500 {
		t.Fatalf("non-tool p50 %d", m.NonToolP50)
	}
	if m.SpikeCount != 0 {
		t.Fatalf("tool turn should not count as a spike, got %d", m.SpikeCount)
	}
	if m.InHumanBandPct < 99 {
		t.Fatalf("human band %v", m.InHumanBandPct)
	}
}

func TestContainsFiller(t *testing.T) {
	if !ContainsFiller("One moment, checking the book.") {
		t.Fatal("expected filler")
	}
	if ContainsFiller("Saturday at 7:30 patio is open.") {
		t.Fatal("false filler")
	}
}

func TestDelayedToolNamesSkipsVerify(t *testing.T) {
	sc := scenario.Scenario{
		ToolDelayMS: map[string]int{"lookup_appointment": 3000},
		ExpectedTools: []scenario.ExpectedTool{
			{Name: "verify_identity"},
			{Name: "lookup_appointment"},
		},
	}
	got := DelayedToolNames(sc)
	if len(got) != 1 || got[0] != "lookup_appointment" {
		t.Fatalf("got %v", got)
	}
}

func TestScoreFiller(t *testing.T) {
	rate := audio.Rate
	start := time.Now().Add(-2 * time.Second)
	agent := audio.Concat(audio.Silence(rate/5), audio.Tone(3*rate, 220, 12000))
	rec := caller.Result{Agent: agent, Rate: rate, StartedAt: start}
	sc := scenario.Scenario{ToolDelayMS: map[string]int{"check_availability": 3000}}
	sess := &world.Session{Tools: []world.ToolCall{{
		Name:    "check_availability",
		Started: start.Add(100 * time.Millisecond),
		Ended:   start.Add(3100 * time.Millisecond),
	}}}
	m := &Metrics{}
	ScoreFiller(m, sc, rec, sess, Transcript{
		Text: "One moment, checking the book.",
		Words: []TranscriptWord{
			{Text: "One", StartMS: 200, EndMS: 260},
			{Text: "moment", StartMS: 270, EndMS: 350},
			{Text: "checking", StartMS: 360, EndMS: 450},
		},
	})
	if !m.FillerHeard || !m.FillerNonBlocking {
		t.Fatalf("filler %+v", m)
	}
	if len(m.FillerFail) != 0 {
		t.Fatalf("fails %v", m.FillerFail)
	}
}

// An agent that reads the booking back across the whole lookup never leaves the caller
// waiting, and says something more useful than a stall phrase. Requiring the phrase failed
// it, and a contract rewritten to lead with the phrase passed this gate while losing four
// other scenarios, so what the words were is recorded and what fails is silence.
func TestScoreFillerAcceptsAReadBackThatCoversTheWait(t *testing.T) {
	rate := audio.Rate
	start := time.Now().Add(-4 * time.Second)
	rec := caller.Result{
		Agent:     audio.Concat(audio.Silence(rate/10), audio.Tone(3*rate, 220, 12000)),
		Rate:      rate,
		StartedAt: start,
	}
	sc := scenario.Scenario{ToolDelayMS: map[string]int{"check_availability": 3000}}
	sess := &world.Session{Tools: []world.ToolCall{{
		Name:    "check_availability",
		Started: start.Add(100 * time.Millisecond),
		Ended:   start.Add(3100 * time.Millisecond),
	}}}
	m := &Metrics{}
	ScoreFiller(m, sc, rec, sess, Transcript{
		Text: "That is a party of four at seven thirty on the patio for Alvarez.",
		Words: []TranscriptWord{
			{Text: "party", StartMS: 200, EndMS: 400},
			{Text: "Alvarez", StartMS: 2600, EndMS: 2900},
		},
	})
	if len(m.FillerFail) != 0 {
		t.Fatalf("speech covering the whole wait should pass: %v", m.FillerFail)
	}
	if m.FillerHeard {
		t.Fatal("no stall phrase was spoken, so none should be recorded")
	}
}

// Saying "one moment" and then going quiet is the failure the gate is for. The phrase was
// what it used to look for, so this was the one shape that passed while sounding worst.
func TestScoreFillerFailsAStallPhraseFollowedBySilence(t *testing.T) {
	rate := audio.Rate
	start := time.Now().Add(-4 * time.Second)
	rec := caller.Result{
		Agent:     audio.Concat(audio.Silence(rate/10), audio.Tone(rate/2, 220, 12000), audio.Silence(3*rate)),
		Rate:      rate,
		StartedAt: start,
	}
	sc := scenario.Scenario{ToolDelayMS: map[string]int{"check_availability": 3000}}
	sess := &world.Session{Tools: []world.ToolCall{{
		Name:    "check_availability",
		Started: start.Add(100 * time.Millisecond),
		Ended:   start.Add(3100 * time.Millisecond),
	}}}
	m := &Metrics{}
	ScoreFiller(m, sc, rec, sess, Transcript{
		Text: "One moment.",
		Words: []TranscriptWord{
			{Text: "One", StartMS: 150, EndMS: 300},
			{Text: "moment", StartMS: 310, EndMS: 500},
		},
	})
	if !m.FillerHeard {
		t.Fatal("the phrase was spoken and should be recorded")
	}
	if len(m.FillerFail) == 0 {
		t.Fatal("two seconds of dead air after the phrase should fail")
	}
	if m.FillerSilenceMS < 2000 {
		t.Fatalf("silence measured as %d ms", m.FillerSilenceMS)
	}
}

// The reply that asks for a tool is spoken before the request is handed over, so the
// filler lands just before the window opens. The non-blocking half of the gate already
// looked back for it; the phrase half has to look back the same way.
func TestScoreFillerCountsAFillerSpokenAsTheToolWasRequested(t *testing.T) {
	rate := audio.Rate
	start := time.Now().Add(-4 * time.Second)
	agent := audio.Concat(audio.Silence(rate/5), audio.Tone(3*rate, 220, 12000))
	rec := caller.Result{Agent: agent, Rate: rate, StartedAt: start}
	sc := scenario.Scenario{ToolDelayMS: map[string]int{"check_availability": 3000}}
	sess := &world.Session{Tools: []world.ToolCall{{
		Name:    "check_availability",
		Started: start.Add(900 * time.Millisecond),
		Ended:   start.Add(3900 * time.Millisecond),
	}}}
	m := &Metrics{}
	ScoreFiller(m, sc, rec, sess, Transcript{
		Text: "One moment, checking the book.",
		Words: []TranscriptWord{
			{Text: "One", StartMS: 500, EndMS: 560},
			{Text: "moment", StartMS: 570, EndMS: 650},
		},
	})
	if !m.FillerHeard {
		t.Fatalf("filler spoken 400ms before the tool started was not heard: %v", m.FillerFail)
	}
	if len(m.FillerFail) != 0 {
		t.Fatalf("fails %v", m.FillerFail)
	}
}

func TestHoldThroughOverlap(t *testing.T) {
	rate := audio.Rate
	agent := audio.Concat(audio.Silence(rate/10), audio.Tone(rate, 220, 12000))
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{{
			Kind:         scenario.TriggerDuringAgent,
			RecStartMs:   400,
			RecEndMs:     600,
			OverlapSound: "cough",
		}},
	}
	if !HoldThroughOverlap(ScoreOverlaps(rec)) {
		t.Fatal("agent continued through cough")
	}
	rec.Agent = audio.Concat(audio.Tone(rate/5, 220, 12000), audio.Silence(rate))
	if HoldThroughOverlap(ScoreOverlaps(rec)) {
		t.Fatal("agent stopped at cough")
	}
}

func TestMergeUtterancesJoinsTheSelectivityT1Gaps(t *testing.T) {
	// restaurant.selectivity-t1: three sentences with 260 ms and 620 ms of true silence
	// between them. The cough sits in the first gap at [12600, 12800].
	spans := []audio.Span{
		{StartMs: 12220, EndMs: 12540},
		{StartMs: 12800, EndMs: 13380},
		{StartMs: 14000, EndMs: 14500},
	}
	got := mergeUtterances(spans, UtteranceMergeGapMS)
	if len(got) != 1 || got[0].StartMs != 12220 || got[0].EndMs != 14500 {
		t.Fatalf("merged %+v", got)
	}
}

func TestScoreOverlapsDoesNotTreatAGapCoughAsANewTurn(t *testing.T) {
	rate := audio.Rate
	totalMs := 15000
	n := rate * totalMs / 1000
	agent := make([]int16, n)
	for _, span := range []audio.Span{
		{StartMs: 12220, EndMs: 12540},
		{StartMs: 12800, EndMs: 13380},
		{StartMs: 14000, EndMs: 14500},
	} {
		start := span.StartMs * rate / 1000
		end := span.EndMs * rate / 1000
		for i := start; i < end && i < n; i++ {
			agent[i] = 10000
		}
	}
	rec := caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{
			{TurnID: "cough", Kind: scenario.TriggerDuringAgent, RecStartMs: 12600, RecEndMs: 12800, OverlapSound: "cough"},
			{TurnID: "talker", Kind: scenario.TriggerDuringAgent, RecStartMs: 13300, RecEndMs: 14500, OverlapSound: "talker"},
		},
	}
	checks := ScoreOverlaps(rec)
	if !SelectivityHold(checks) {
		t.Fatalf("cough in a sentence gap read as a new turn: %+v", checks)
	}
	if !HoldThroughOverlap(checks) {
		t.Fatalf("merged reply did not hold through the cough: %+v", checks)
	}
}

func TestScoreOverlapsChecksEverySound(t *testing.T) {
	rate := audio.Rate
	rec := caller.Result{
		Agent: audio.Concat(audio.Silence(rate/10), audio.Tone(rate, 220, 12000)),
		Rate:  rate,
		Events: []caller.Event{
			{TurnID: "cough", Kind: scenario.TriggerDuringAgent, RecStartMs: 300, RecEndMs: 400, OverlapSound: "cough"},
			{TurnID: "talker", Kind: scenario.TriggerDuringAgent, RecStartMs: 600, RecEndMs: 700, OverlapSound: "talker"},
		},
	}
	checks := ScoreOverlaps(rec)
	if len(checks) != 2 {
		t.Fatalf("checks %d", len(checks))
	}
	if !HoldThroughOverlap(checks) || !SelectivityHold(checks) {
		t.Fatalf("checks %+v", checks)
	}
}

func TestFalseCutoffUsesScriptIntervalsNotCallerVAD(t *testing.T) {
	rate := audio.Rate
	rec := caller.Result{
		Caller: audio.Tone(rate, 180, 12000),
		Agent:  audio.Concat(audio.Silence(rate/2), audio.Tone(rate/10, 220, 12000)),
		Rate:   rate,
		Events: []caller.Event{{TurnID: "intro", RecStartMs: 100, RecEndMs: 300}},
	}
	if got := FalseCutoff(rec); got != 0 {
		t.Fatalf("noise outside the scripted interval produced %d false cutoff(s)", got)
	}
	rec.Events[0].RecEndMs = 800
	if got := FalseCutoff(rec); got != 1 {
		t.Fatalf("agent start inside caller interval produced %d false cutoff(s)", got)
	}
}

func TestTimingFromRecording(t *testing.T) {
	rate := audio.Rate
	callerPCM := audio.Concat(audio.Silence(rate/10), audio.Tone(rate/5, 180, 12000), audio.Silence(rate))
	agentPCM := audio.Concat(audio.Silence(rate/2), audio.Tone(rate/10, 220, 12000), audio.Silence(rate/10))
	rec := caller.Result{
		Caller: callerPCM,
		Agent:  agentPCM,
		Rate:   rate,
		Events: []caller.Event{{TurnID: "intro", RecEndMs: 300}},
	}
	got, _ := TimingFromRecording(rec)
	if len(got) != 1 {
		t.Fatalf("got %d samples", len(got))
	}
	if got[0].V2VMS < 100 || got[0].V2VMS > 400 {
		t.Fatalf("v2v %d, want ~200ms", got[0].V2VMS)
	}
}

func TestTimingIgnoresMergedEarlierAgentSpan(t *testing.T) {
	rate := audio.Rate
	agent := audio.Concat(audio.Tone(rate, 220, 12000), audio.Silence(rate/5), audio.Tone(rate/10, 220, 12000))
	rec := caller.Result{
		Agent:  agent,
		Rate:   rate,
		Events: []caller.Event{{TurnID: "intro", RecEndMs: 400}},
	}
	got, _ := TimingFromRecording(rec)
	if len(got) != 1 {
		t.Fatalf("got %d", len(got))
	}
	if got[0].V2VMS < 500 {
		t.Fatalf("v2v %d should skip the in-progress greeting", got[0].V2VMS)
	}
}

func TestWorldGates(t *testing.T) {
	m := &Metrics{}
	sc := scenario.Scenario{
		EndState: []scenario.Assertion{{Path: "reservation.allergen", Eq: "peanut"}},
		Entities: []scenario.Entity{{Name: "allergen", Value: "peanut", InTools: true, InSpeech: true}},
	}
	sess := &world.Session{
		State: map[string]any{"reservation": map[string]any{"allergen": "peanut"}},
		Tools: []world.ToolCall{{Name: "create_reservation", Args: map[string]any{"allergen": "peanut"}}},
	}
	WorldGates(m, sc, sess, "peanut allergy noted")
	if len(m.EndStateFail)+len(m.EntityToolFail)+len(m.EntitySpeechFail) != 0 {
		t.Fatalf("%+v", m)
	}
}

func TestViolationsDropsNonViolations(t *testing.T) {
	got := violations([]finding{
		{Rule: "say_do", Evidence: "claimed the update but no tool call", Violation: true},
		{Rule: "say_do", Evidence: "no attempt to update pharmacy, so this is consistent", Violation: false},
		{Rule: "policy", Violation: true},
	})
	want := []string{"say_do: claimed the update but no tool call", "policy"}
	if len(got) != len(want) {
		t.Fatalf("violations %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("violations[%d] = %q want %q", i, got[i], want[i])
		}
	}
}

func TestStaffNamesFromSeed(t *testing.T) {
	seed := map[string]any{
		"patients": []any{
			map[string]any{"name": "Maya Chen"},
			map[string]any{"name": "Leo Chen"},
		},
		"appointments": []any{
			map[string]any{"clinician": "Dr Chen"},
			map[string]any{"clinician": "Dr Adeyemi"},
			map[string]any{"clinician": "Dr Chen"},
		},
	}
	got := staffNames(seed)
	if len(got) != 2 || got[0] != "Dr Adeyemi" || got[1] != "Dr Chen" {
		t.Fatalf("staffNames %v", got)
	}
}

// Tool timestamps are wall time; RecEndMs is recording time, which advances one 20 ms frame per
// pacer tick and falls behind under load. Subtracting StartedAt compares the two directly, so a
// tool that ran inside a reply gap can land outside it and the turn is scored as a non-tool turn.
func TestMarkToolTurnsMapsWallTimeOntoRecordingTime(t *testing.T) {
	start := time.Now()
	rec := caller.Result{
		Caller:       make([]int16, audio.Rate*10),
		Rate:         audio.Rate,
		Events:       []caller.Event{{TurnID: "a", RecEndMs: 5000}},
		StartedAt:    start,
		FirstFrameAt: start,
		// 10 s of recording took 12 s of wall time.
		LastFrameAt: start.Add(12 * time.Second),
	}
	if got := rec.SampleMs(start.Add(6600 * time.Millisecond)); got != 5500 {
		t.Fatalf("SampleMs %d, want 5500", got)
	}
	if got := rec.ClockDriftMS(); got != 2000 {
		t.Fatalf("clock drift %d, want 2000", got)
	}

	m := &Metrics{V2V: []Timing{{TurnID: "a", V2VMS: 1000}}}
	MarkToolTurns(m, rec, []world.ToolCall{{
		Name:    "lookup_appointment",
		Started: start.Add(6600 * time.Millisecond),
		Ended:   start.Add(6660 * time.Millisecond),
	}})
	if !m.V2V[0].Tool {
		t.Fatal("tool ran inside the reply gap but the turn was scored as a non-tool turn")
	}
}

func TestTimingReportsWhyATurnWasNotMeasured(t *testing.T) {
	rate := audio.Rate
	rec := caller.Result{
		Caller: audio.Silence(rate),
		Agent:  audio.Silence(rate),
		Rate:   rate,
		Events: []caller.Event{
			{TurnID: "barge", RecEndMs: 100, BargeIn: true},
			{TurnID: "over", RecEndMs: 200, Overlap: true},
			{TurnID: "silent", RecEndMs: 300},
		},
	}
	got, dropped := TimingFromRecording(rec)
	if len(got) != 0 {
		t.Fatalf("measured %d turns on a silent agent leg", len(got))
	}
	want := []DroppedTurn{
		{TurnID: "barge", Reason: DropBargeIn},
		{TurnID: "over", Reason: DropOverlap},
		{TurnID: "silent", Reason: DropNoOnset},
	}
	if len(dropped) != len(want) {
		t.Fatalf("dropped %+v", dropped)
	}
	for i := range want {
		if dropped[i] != want[i] {
			t.Errorf("dropped[%d] = %+v want %+v", i, dropped[i], want[i])
		}
	}
}

func TestExtraToolNames(t *testing.T) {
	got := ExtraToolNames([]world.ToolCall{
		{Name: "check_availability"},
		{Name: "invented_refund"},
		{Name: "invented_refund"},
	}, []scenario.ExpectedTool{{Name: "check_availability"}})
	if len(got) != 1 || got[0] != "invented_refund" {
		t.Fatalf("%v", got)
	}
}

func TestCountConversation(t *testing.T) {
	rate := audio.Rate
	agent := audio.Silence(rate / 2)
	speech := make([]int16, rate/5)
	for i := range speech {
		speech[i] = 8000
	}
	agent = append(agent, speech...)
	m := Metrics{}
	CountConversation(&m, caller.Result{
		Agent: agent,
		Rate:  rate,
		Events: []caller.Event{
			{TurnID: "a", Text: true},
			{TurnID: "b", Text: true},
			{TurnID: "cough", OverlapSound: "cough"},
		},
	}, "hello there friend")
	if m.CallerTurns != 2 {
		t.Fatalf("caller turns %d", m.CallerTurns)
	}
	if m.AgentTurns < 1 {
		t.Fatalf("agent turns %d", m.AgentTurns)
	}
	if m.AgentWords != 3 {
		t.Fatalf("words %d", m.AgentWords)
	}
}

// noise_kitchen asks the agent to complete the booking despite the noise and lists nothing
// to refuse. The empty "Must refuse:" heading sat directly above the policy list, and the
// judge read the list as what had to be refused: it failed the agent for booking, which is
// the thing the scenario wanted.
func TestJudgePromptDoesNotHeadThePolicyListWithAnEmptyRefusal(t *testing.T) {
	sc := scenario.Scenario{Policy: []string{"Complete the booking despite kitchen noise."}}
	prompt := judgePrompt(sc, "book me a table", "you are booked", nil)
	if strings.Contains(prompt, "Must refuse") {
		t.Fatalf("nothing to refuse, so no refusal heading:\n%s", prompt)
	}
	if !strings.Contains(prompt, "required to satisfy:\n- Complete the booking despite kitchen noise.") {
		t.Fatalf("the policy list should read as requirements:\n%s", prompt)
	}
}

func TestJudgePromptKeepsRefusalsWhenThereAreSome(t *testing.T) {
	sc := scenario.Scenario{
		Policy: []string{"Do not overbook a full slot."},
		Judge:  scenario.JudgeSpec{MustRefuse: []string{"squeeze in an extra table"}},
	}
	prompt := judgePrompt(sc, "squeeze us in", "I cannot do that", nil)
	if !strings.Contains(prompt, "Must refuse: squeeze in an extra table") {
		t.Fatalf("refusals should still be named:\n%s", prompt)
	}
}
