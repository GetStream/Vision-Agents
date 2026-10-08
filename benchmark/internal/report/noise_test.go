package report

import (
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

func noiseRun(runID string, nonToolMS int, passed bool) LabeledRun {
	outcome := OutcomePass
	if !passed {
		outcome = OutcomeFail
	}
	sum := BuildSummary("accelerated", runID, 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "t1", V2VMS: nonToolMS}}}},
		{ScenarioID: "restaurant.order", Pack: "restaurant", Category: "task", Trial: 1, Passed: passed, Outcome: outcome, Metrics: score.Metrics{V2V: []score.Timing{{TurnID: "t1", V2VMS: nonToolMS}}}},
	})
	sum.Manifest = RunManifest{Target: "accelerated", GitCommit: "abc", NetworkProfile: "us-east", ScenarioHash: "s", ContractHash: "c"}
	return LabeledRun{Label: runID, Summary: sum}
}

func spreadOf(t *testing.T, n NoiseFloor, name string) MetricSpread {
	t.Helper()
	for _, m := range n.Metrics {
		if m.Name == name {
			return m
		}
	}
	t.Fatalf("no %s in %+v", name, n.Metrics)
	return MetricSpread{}
}

func TestMeasureNoiseTakesTheLargestGapBetweenRuns(t *testing.T) {
	n, err := MeasureNoise([]LabeledRun{noiseRun("r1", 500, true), noiseRun("r2", 540, true), noiseRun("r3", 470, false)})
	if err != nil {
		t.Fatal(err)
	}
	reply := spreadOf(t, n, "non_tool_p50_ms")
	if reply.MDE != 70 || reply.Mean < 503 || reply.Mean > 504 {
		t.Fatalf("non-tool P50 spread = %+v, want MDE 70 and mean 503.3", reply)
	}
	if rate := spreadOf(t, n, "pass_rate"); rate.MDE != 50 {
		t.Fatalf("pass rate MDE = %v pp, want 50", rate.MDE)
	}
	for _, m := range n.Metrics {
		if m.Name == "tool_p50_ms" || m.Name == "first_response_p50_ms" {
			t.Fatalf("%s had no samples and must not get an MDE", m.Name)
		}
	}
	if strings.Join(n.Packs, ",") != "restaurant" || n.NetworkProfile != "us-east" || len(n.Runs) != 3 {
		t.Fatalf("noise floor lost its series: %+v", n)
	}
}

func TestMeasureNoiseRefusesRunsFromDifferentSeries(t *testing.T) {
	if _, err := MeasureNoise([]LabeledRun{noiseRun("r1", 500, true), noiseRun("r2", 500, true)}); err == nil {
		t.Fatal("two runs are too few to read a spread from")
	}
	_, err := MeasureNoise([]LabeledRun{noiseRun("r1", 500, true), noiseRun("r2", 500, true), noiseRun("r1", 500, true)})
	if err == nil || !strings.Contains(err.Error(), "given twice") {
		t.Fatalf("the same run twice must be refused, got %v", err)
	}
	moved := noiseRun("r3", 500, true)
	moved.Summary.Manifest.NetworkProfile = "laptop"
	_, err = MeasureNoise([]LabeledRun{noiseRun("r1", 500, true), noiseRun("r2", 500, true), moved})
	if err == nil || !strings.Contains(err.Error(), "network_profile") {
		t.Fatalf("a run on another network profile must be refused, got %v", err)
	}
}

func TestCompareFlagsOnlyChangesBiggerThanTheNoiseFloor(t *testing.T) {
	n, err := MeasureNoise([]LabeledRun{noiseRun("r1", 500, true), noiseRun("r2", 540, true), noiseRun("r3", 470, true)})
	if err != nil {
		t.Fatal(err)
	}
	md := CompareMarkdown(CompareConfig{Baseline: 0, MDE: &n, Runs: []LabeledRun{
		noiseRun("baseline", 500, true),
		noiseRun("within", 560, true),
		noiseRun("slower", 600, true),
	}})
	if !strings.Contains(md, "non_tool_p50_ms regression (+100 ms, MDE 70)") {
		t.Fatalf("a change past the MDE must be flagged:\n%s", md)
	}
	for _, line := range strings.Split(md, "\n") {
		if strings.HasPrefix(line, "| within |") && strings.Contains(line, "regression") {
			t.Fatalf("a change within the MDE must not be flagged: %s", line)
		}
	}
	if strings.Contains(md, "different series") {
		t.Fatalf("a baseline from the measured series must be gated:\n%s", md)
	}

	other := noiseRun("baseline", 500, true)
	other.Summary.Manifest.ScenarioHash = "new"
	md = CompareMarkdown(CompareConfig{Baseline: 0, MDE: &n, Runs: []LabeledRun{other, noiseRun("slower", 600, true)}})
	if !strings.Contains(md, "different series than the noise floor (scenario_hash") || strings.Contains(md, "regression") {
		t.Fatalf("a baseline from another series must flag nothing:\n%s", md)
	}

	livekit := noiseRun("livekit", 900, true)
	livekit.Summary.Manifest.Target = "livekit"
	md = CompareMarkdown(CompareConfig{Baseline: 0, MDE: &n, Runs: []LabeledRun{noiseRun("baseline", 500, true), livekit}})
	if !strings.Contains(md, "not gated: target is \"livekit\"") || strings.Contains(md, "regression") {
		t.Fatalf("a run of another target must not be judged on this target's noise floor:\n%s", md)
	}
}
