package report

import (
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
