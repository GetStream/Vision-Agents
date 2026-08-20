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
	if got := BargeInStopMS(rec); got != -1 {
		t.Fatalf("no barge event: %d", got)
	}
	rec.Events = []caller.Event{{
		BargeIn:    true,
		Kind:       scenario.TriggerBargeIn,
		RecStartMs: 400,
	}}
	if got := BargeInStopMS(rec); got != -1 {
		t.Fatalf("no straddle: %d", got)
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
	got := BargeInStopMS(rec)
	if got < 0 {
		t.Fatalf("want measured stop, got %d", got)
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
	agent := audio.Concat(audio.Silence(rate/5), audio.Tone(rate, 220, 12000))
	rec := caller.Result{Agent: agent, Rate: rate, StartedAt: start}
	sc := scenario.Scenario{ToolDelayMS: map[string]int{"check_availability": 3000}}
	sess := &world.Session{Tools: []world.ToolCall{{
		Name:    "check_availability",
		Started: start.Add(100 * time.Millisecond),
		Ended:   start.Add(3100 * time.Millisecond),
	}}}
	m := &Metrics{}
	ScoreFiller(m, sc, rec, sess, "One moment, checking the book.")
	if !m.FillerHeard || !m.FillerNonBlocking {
		t.Fatalf("filler %+v", m)
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
			Kind:       scenario.TriggerDuringAgent,
			RecStartMs: 400,
			RecEndMs:   600,
		}},
	}
	if !HoldThroughOverlap(rec) {
		t.Fatal("agent continued through cough")
	}
	rec.Agent = audio.Concat(audio.Tone(rate/5, 220, 12000), audio.Silence(rate))
	if HoldThroughOverlap(rec) {
		t.Fatal("agent stopped at cough")
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
	got := TimingFromRecording(rec)
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
	got := TimingFromRecording(rec)
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
