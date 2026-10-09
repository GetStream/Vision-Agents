package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

func TestMarkdownAndSummary(t *testing.T) {
	calls := []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 420}}, Passed: true}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: false, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 900}}, GateNotes: []string{"end_state"}}},
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 500}}, Passed: true}},
	}
	sum := BuildSummary("vision-agents", "run1", 3, calls)
	if len(sum.Packs) != 2 {
		t.Fatalf("packs %d", len(sum.Packs))
	}
	md := Markdown(sum)
	if !strings.Contains(md, "restaurant") || !strings.Contains(md, "pass@k") {
		t.Fatalf("markdown:\n%s", md)
	}
	if !strings.Contains(md, "Task completion") || !strings.Contains(md, "Reply gap") {
		t.Fatalf("scorecard missing:\n%s", md)
	}
	dir := t.TempDir()
	if err := Write(dir, sum); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "summary.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err != nil {
		t.Fatal(err)
	}
}

func TestMarkdownSurfacesAssertionFailures(t *testing.T) {
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{{
		ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1,
		Outcome: OutcomeFail, Passed: false,
		Metrics: score.Metrics{
			EndStateFail:     []string{"identity_verified want true got false"},
			ExpectedToolFail: []string{"verify_identity.member_id want ABC123456 got XYZ987654", "lookup_appointment not called", "reschedule_appointment not called"},
			GateNotes:        []string{"end_state", "expected_tools"},
		},
	}})
	md := Markdown(sum)
	if !strings.Contains(md, "identity_verified want true got false") {
		t.Fatalf("end-state detail missing:\n%s", md)
	}
	if !strings.Contains(md, "verify_identity.member_id want ABC123456 got XYZ987654") {
		t.Fatalf("root tool failure missing:\n%s", md)
	}
	if !strings.Contains(md, "lookup_appointment not called") {
		t.Fatalf("cascade detail missing:\n%s", md)
	}
	if strings.Contains(md, "| expected_tools | lookup_appointment not called |") {
		t.Fatalf("cascade counted as a root failure:\n%s", md)
	}
	if !strings.Contains(md, "## Failed trials") {
		t.Fatalf("failed trials section missing:\n%s", md)
	}
}

func TestMarkdownSurfacesHoldAndBargeIn(t *testing.T) {
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{{
		ScenarioID: "restaurant.selectivity", Pack: "restaurant", Category: "checklist", Trial: 1,
		Outcome: OutcomeFail, Passed: false,
		Metrics: score.Metrics{
			SelectivityHold:    false,
			HoldThroughOverlap: false,
			BargeInStopMS:      940,
			GateNotes:          []string{"selectivity", "hold", "barge_in"},
		},
	}})
	md := Markdown(sum)
	if !strings.Contains(md, "agent started a turn on non-directed overlap") {
		t.Fatalf("selectivity detail missing:\n%s", md)
	}
	if !strings.Contains(md, "stop 940 ms exceeds") {
		t.Fatalf("barge-in detail missing:\n%s", md)
	}
}

func TestScorecardTargetVsOurs(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 2, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "a", V2VMS: 420}, {TurnID: "b", V2VMS: 450, Tool: true}}}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: false, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "a", V2VMS: 900}}, SpikeCount: 1}},
		{ScenarioID: "restaurant.interrupt", Pack: "restaurant", Category: "interrupt", Trial: 1, Passed: true, Metrics: score.Metrics{BargeInStopMS: 400, V2V: []score.Timing{{TurnID: "a", V2VMS: 500}}}},
		{ScenarioID: "restaurant.noise_kitchen", Pack: "restaurant", Category: "checklist", Trial: 1, Passed: false, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "a", V2VMS: 600}}}},
	})
	rows := Scorecard(sum)
	byName := map[string]Row{}
	for _, r := range rows {
		byName[r.Name] = r
	}
	task := byName["Task completion"]
	if task.Ours != "1/2 pass" || task.Verdict != VerdictWarn {
		t.Fatalf("task %+v", task)
	}
	noise := byName["Noise"]
	if noise.Gap != "incomplete" || noise.Verdict != VerdictWarn {
		t.Fatalf("noise %+v", noise)
	}
	coh := byName["Coherence (2 min)"]
	if coh.Verdict != VerdictSkip {
		t.Fatalf("coherence %+v", coh)
	}
	barge := byName["Barge-in stop"]
	if barge.Ours != "400 ms" || barge.Verdict != VerdictOK {
		t.Fatalf("barge %+v", barge)
	}
	gap := byName["Reply gap (non-tool P50)"]
	if gap.Verdict != VerdictOK && gap.Verdict != VerdictMiss && gap.Verdict != VerdictWarn {
		t.Fatalf("reply gap %+v", gap)
	}
	if !strings.Contains(Table(sum), "BENCHMARK") {
		t.Fatal("table missing header")
	}
}

func TestInvalidTrialsAreNotScored(t *testing.T) {
	sum := BuildSummary("livekit", "run1", 2, []CallResult{
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 500}}, Passed: true}},
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 2, Error: "spawn livekit worker: did not become ready"},
	})
	cell := sum.Packs[0].Cells[0]
	if cell.Trials != 2 || cell.Passed != 1 || cell.Invalid != 1 {
		t.Fatalf("trials %d passed %d invalid %d", cell.Trials, cell.Passed, cell.Invalid)
	}
	if cell.Complete || cell.PassAtK || cell.PassHatK {
		t.Fatalf("an invalid trial must make reliability incomplete: %+v", cell)
	}
	if sum.InvalidTrials() != 1 {
		t.Fatalf("invalid trials %d", sum.InvalidTrials())
	}
	if md := Markdown(sum); !strings.Contains(md, "reliability incomplete") {
		t.Fatalf("markdown does not surface the invalid trial:\n%s", md)
	}
}

func TestTargetFailureIsAValidFailedTrial(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 1, []CallResult{{
		ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1,
		Outcome: OutcomeFail, Error: "agent published no audio track",
	}})
	scenario := sum.Packs[0].Scenarios[0]
	if !scenario.Complete || scenario.Valid != 1 || scenario.Invalid != 0 || scenario.PassAtK {
		t.Fatalf("target failure classification: %+v", scenario)
	}
	if sum.InvalidTrials() != 0 {
		t.Fatal("target failure counted as evaluator invalid")
	}
}

func TestChecklistReliabilityIsComputedPerScenario(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 2, []CallResult{
		{ScenarioID: "restaurant.noise", Pack: "restaurant", Category: "checklist", Trial: 1, Passed: true},
		{ScenarioID: "restaurant.noise", Pack: "restaurant", Category: "checklist", Trial: 2, Passed: true},
		{ScenarioID: "restaurant.selectivity", Pack: "restaurant", Category: "checklist", Trial: 1, Passed: false},
		{ScenarioID: "restaurant.selectivity", Pack: "restaurant", Category: "checklist", Trial: 2, Passed: false},
	})
	pack := sum.Packs[0]
	if len(pack.Scenarios) != 2 {
		t.Fatalf("scenarios %d", len(pack.Scenarios))
	}
	cell := pack.Cells[0]
	if cell.PassAtK || cell.PassHatK {
		t.Fatalf("one passing checklist scenario hid another failure: %+v", cell)
	}
}

func TestWorldContactWarningIsReported(t *testing.T) {
	sum := BuildSummary("livekit", "run1", 1, []CallResult{{
		ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1,
		Warnings: []string{"target never contacted the world server"},
	}})
	md := Markdown(sum)
	if !strings.Contains(md, "## Warnings") || !strings.Contains(md, "never contacted the world server") {
		t.Fatalf("markdown missing warning:\n%s", md)
	}
}

// The pack figure used to be a median of per-call medians, and score.percentile returns the
// minimum of a two-sample set. Together those turned LiveKit's [2880, 2040] into 2040 and made
// it look faster than a Stream run whose samples were mostly lower.
func TestPackP50PoolsRawSamplesNotPerCallMedians(t *testing.T) {
	sum := BuildSummary("livekit", "run1", 2, []CallResult{
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true,
			Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 2880}, {TurnID: "identity", V2VMS: 2040}}}},
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 2, Passed: true,
			Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "intro", V2VMS: 6080}, {TurnID: "identity", V2VMS: 2380}, {TurnID: "insurance", V2VMS: 2480}}}},
	})
	pack := sum.Packs[0]
	if pack.V2VSamples != 5 {
		t.Fatalf("pooled %d samples, want 5", pack.V2VSamples)
	}
	// Pooled and sorted: 2040 2380 2480 2880 6080. Nearest-rank P50 is 2480.
	if pack.V2VP50 != 2480 {
		t.Fatalf("pack P50 %d, want 2480 over the pooled samples", pack.V2VP50)
	}
	if !strings.Contains(Markdown(sum), "2480 ms (n=5)") {
		t.Fatal("the latency table does not carry the sample count")
	}
}

func TestScenarioSummaryIncludesWilsonInterval(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 3, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: true},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 3, Passed: false},
	})
	scenario := sum.Packs[0].Scenarios[0]
	if scenario.PassRate < 0.66 || scenario.PassRate > 0.67 {
		t.Fatalf("pass rate %f", scenario.PassRate)
	}
	if scenario.CI95Low <= 0 || scenario.CI95High >= 1 || scenario.CI95Low >= scenario.CI95High {
		t.Fatalf("interval %.3f–%.3f", scenario.CI95Low, scenario.CI95High)
	}
}

func TestDroppedTurnsAreReported(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 1, []CallResult{{
		ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true,
		Metrics: score.Metrics{
			V2V: []score.Timing{{TurnID: "identity", V2VMS: 3300}},
			Dropped: []score.DroppedTurn{
				{TurnID: "intro", Reason: score.DropOverlap},
				{TurnID: "insurance", Reason: score.DropOverlap},
			},
		},
	}})
	if sum.Packs[0].DroppedTurns != 2 {
		t.Fatalf("dropped turns %d, want 2", sum.Packs[0].DroppedTurns)
	}
	if md := Markdown(sum); !strings.Contains(md, "3300 ms (n=1)") {
		t.Fatalf("a one-sample P50 is not marked as one:\n%s", md)
	}
}

func TestSummaryReportsTimeToFirstResponse(t *testing.T) {
	sum := BuildSummary("accelerated", "run1", 3, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true, Metrics: score.Metrics{FirstResponse: &score.Timing{TurnID: "t1", V2VMS: 800}}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Outcome: OutcomePass, Passed: true, Metrics: score.Metrics{FirstResponse: &score.Timing{TurnID: "t1", V2VMS: 1200, Tool: true}}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 3, Outcome: OutcomeFail, Metrics: score.Metrics{}},
		{ScenarioID: "restaurant.noise", Pack: "restaurant", Category: "checklist", Trial: 1, Outcome: OutcomeInvalid, InvalidReason: []string{"agent stt skipped"}, Metrics: score.Metrics{FirstResponse: &score.Timing{TurnID: "t1", V2VMS: 50}}},
	})
	pack := sum.Packs[0]
	if pack.FirstResponseP50 != 800 || pack.FirstResponseP95 != 1200 || pack.FirstResponseSamples != 2 || pack.FirstResponseTool != 1 {
		t.Fatalf("pack %+v", pack)
	}
	md := Markdown(sum)
	if !strings.Contains(md, "| restaurant | 800 ms | 1200 ms | 2 | 1 |") {
		t.Fatalf("time to first response table missing:\n%s", md)
	}
	if !strings.Contains(md, "| 1200 ms (tool) |") {
		t.Fatalf("per-call first response missing:\n%s", md)
	}
	empty := Markdown(BuildSummary("livekit", "run2", 1, []CallResult{
		{ScenarioID: "telecom.golden", Pack: "telecom", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true},
	}))
	if !strings.Contains(empty, "| telecom | — | — | 0 | 0 |") {
		t.Fatalf("a pack with no first response must not read as 0 ms:\n%s", empty)
	}
}

func TestSummaryReportsReplyTimeOnNonToolTurnsWithToolTurnsApart(t *testing.T) {
	turns := func(ms ...int) []score.Timing {
		var out []score.Timing
		for i, v := range ms {
			out = append(out, score.Timing{TurnID: fmt.Sprintf("t%d", i), V2VMS: v})
		}
		return out
	}
	withTool := turns(1000, 2000, 3000, 10000)
	withTool[3].Tool = true
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true, Metrics: score.Metrics{V2V: withTool}},
	})
	pack := sum.Packs[0]
	if pack.NonToolP50 != 2000 || pack.NonToolP95 != 3000 || pack.NonToolMean != 2000 || pack.NonToolSamples != 3 {
		t.Fatalf("non-tool reply time %+v", pack)
	}
	if pack.ToolP50 != 10000 || pack.ToolSamples != 1 || pack.V2VMean != 4000 {
		t.Fatalf("a tool turn is reported apart and still counted in all turns: %+v", pack)
	}
	md := Markdown(sum)
	if !strings.Contains(md, "| restaurant | 2000 ms (n=3) | 3000 ms (n=3) | 2000 ms (n=3) | 10000 ms (n=1) | 2000 ms (n=4) | 4000 ms (n=4) |") {
		t.Fatalf("reply time table missing:\n%s", md)
	}
}

func TestSummaryPoolsTheRoutersStagesOverEveryTurn(t *testing.T) {
	stage := func(decision int) score.StageTiming {
		return score.StageTiming{TurnID: "t", CadenceMs: 350, DecisionMs: decision, RoundtripMs: 2000 + decision}
	}
	sum := BuildSummary("accelerated", "run1", 2, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{Stages: []score.StageTiming{stage(400), stage(500)}, AgentMetrics: map[string]float64{"stt_latency_ms__avg": 100}}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{Stages: []score.StageTiming{stage(1200)}, AgentMetrics: map[string]float64{"stt_latency_ms__avg": 300}}},
	})
	pack := sum.Packs[0]
	if pack.StageSamples != 3 || pack.StageP50["decision_ms"] != 500 || pack.StageP90["decision_ms"] != 1200 || pack.StageP50["cadence_ms"] != 350 {
		t.Fatalf("stages are pooled over turns, not calls: %+v", pack)
	}
	if pack.AgentMetricsP50["stt_latency_ms__avg"] != 100 {
		t.Fatalf("agent metrics %+v", pack.AgentMetricsP50)
	}
	md := Markdown(sum)
	if !strings.Contains(md, "## Where the router's time goes") || !strings.Contains(md, "| restaurant | 0 / 0 ms | 350 / 350 ms | 500 / 1200 ms | 0 / 0 ms | 0 / 0 ms | 0 / 0 ms | 2500 / 3200 ms | 3 |") {
		t.Fatalf("stage table missing:\n%s", md)
	}
	if !strings.Contains(md, "## What the agent measured") || !strings.Contains(md, "| restaurant | 100 ms | — |") {
		t.Fatalf("agent metrics table missing:\n%s", md)
	}
}

func TestStageP90IsTheNearestRankTailNotTheSlowestTurn(t *testing.T) {
	var stages []score.StageTiming
	for decision := 100; decision <= 1000; decision += 100 {
		stages = append(stages, score.StageTiming{TurnID: "t", DecisionMs: decision, RoundtripMs: 2000 + decision})
	}
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{Stages: stages}},
	})
	pack := sum.Packs[0]
	if pack.StageP50["decision_ms"] != 500 || pack.StageP90["decision_ms"] != 900 || pack.StageP90["roundtrip_ms"] != 2900 {
		t.Fatalf("p90 is taken as the median is, over the same pooled turns: %+v", pack)
	}
	data, err := json.Marshal(pack)
	if err != nil || !strings.Contains(string(data), `"stage_p90_ms":{`) {
		t.Fatalf("summary.json should carry the p90 beside the median: %s %v", data, err)
	}
}

func TestSummaryReportsRepliesAfterAToolOnTheirOwnRow(t *testing.T) {
	caller := score.StageTiming{TurnID: "turn-1", STTMs: 40, CadenceMs: 350, DecisionMs: 400, ModelToTextMs: 600, TextToTTSMs: 50, TTSToAudioMs: 500, RoundtripMs: 1900}
	afterTool := func(model, audio int) score.StageTiming {
		return score.StageTiming{TurnID: "tool-1", Tool: true, ModelToTextMs: model, TextToTTSMs: 50, TTSToAudioMs: audio}
	}
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{Stages: []score.StageTiming{caller, afterTool(600, 400), afterTool(700, 900), afterTool(1500, 3000)}}},
	})
	pack := sum.Packs[0]
	if pack.StageSamples != 1 || pack.ToolStageSamples != 3 {
		t.Fatalf("a reply after a tool is not a caller turn: %+v", pack)
	}
	if pack.ToolStageP50["tts_to_audio_ms"] != 900 || pack.ToolStageP90["tts_to_audio_ms"] != 3000 {
		t.Fatalf("replies after a tool are pooled apart: %+v", pack.ToolStageP50)
	}
	if _, ok := pack.ToolStageP50["roundtrip_ms"]; ok {
		t.Fatalf("a reply after a tool has no roundtrip to report: %+v", pack.ToolStageP50)
	}
	md := Markdown(sum)
	for _, want := range []string{
		"| Pack (median / p90) |",
		"| restaurant | 40 / 40 ms | 350 / 350 ms | 400 / 400 ms | 600 / 600 ms | 50 / 50 ms | 500 / 500 ms | 1900 / 1900 ms | 1 |",
		"| restaurant after a tool | — | — | — | 700 / 1500 ms | 50 / 50 ms | 900 / 3000 ms | — | 3 |",
		"The row after a tool is the replies the agent starts when a tool returns.",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("report is missing %q:\n%s", want, md)
		}
	}
}

func TestAPackWithOnlyRepliesAfterAToolStillGetsItsRow(t *testing.T) {
	sum := BuildSummary("accelerated", "run1", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{Stages: []score.StageTiming{{TurnID: "tool-1", Tool: true, ModelToTextMs: 600, TextToTTSMs: 50, TTSToAudioMs: 400}}}},
	})
	md := Markdown(sum)
	if !strings.Contains(md, "| restaurant after a tool | — | — | — | 600 / 600 ms | 50 / 50 ms | 400 / 400 ms | — | 1 |") {
		t.Fatalf("replies after a tool should be reported on their own:\n%s", md)
	}
	if strings.Contains(md, "| restaurant | 0 / 0 ms") {
		t.Fatalf("a pack with no caller turns timed has no row for them:\n%s", md)
	}
}

func TestAReportWithoutStagesLeavesTheirTablesOut(t *testing.T) {
	// The accelerated target's Python agent measures none of its own stages, because the
	// router runs them, and reports only counters such as the time to join.
	md := Markdown(BuildSummary("accelerated", "run1", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Outcome: OutcomePass, Passed: true,
			Metrics: score.Metrics{AgentMetrics: map[string]float64{"call_join_ms__avg": 2800, "llm_tool_calls__total": 0}}},
	}))
	if strings.Contains(md, "Where the router's time goes") || strings.Contains(md, "What the agent measured") {
		t.Fatalf("a run with nothing to break down should not print an empty breakdown:\n%s", md)
	}
}
