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
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2VP50: 420, Passed: true}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: false, Metrics: score.Metrics{V2VP50: 900, GateNotes: []string{"end_state"}}},
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{V2VP50: 500, Passed: true}},
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
}

func TestScorecardTargetVsOurs(t *testing.T) {
	sum := BuildSummary("vision-agents", "run1", 2, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Metrics: score.Metrics{NonToolP50: 420, V2VP50: 450}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: false, Metrics: score.Metrics{NonToolP50: 900, V2VP50: 900, SpikeCount: 1}},
		{ScenarioID: "restaurant.interrupt", Pack: "restaurant", Category: "interrupt", Trial: 1, Passed: true, Metrics: score.Metrics{BargeInStopMS: 400, NonToolP50: 500, V2VP50: 500}},
		{ScenarioID: "restaurant.noise_kitchen", Pack: "restaurant", Category: "checklist", Trial: 1, Passed: false, Metrics: score.Metrics{NonToolP50: 600, V2VP50: 600}},
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
	if noise.Gap != "miss" || noise.Verdict != VerdictMiss {
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
