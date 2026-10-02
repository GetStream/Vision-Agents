package report

import (
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

func TestCompareMarkdownMarksTheBetterPassRate(t *testing.T) {
	ours := BuildSummary("accelerated", "a", 2, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 400}}, CallerTurns: 4, AgentTurns: 4}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 420}}, CallerTurns: 4, AgentTurns: 5}},
	})
	theirs := BuildSummary("livekit", "b", 2, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 600}}, CallerTurns: 4, AgentTurns: 4}},
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 2, Passed: false, Outcome: OutcomeFail, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 800}}, CallerTurns: 4, AgentTurns: 6}},
	})
	md := CompareMarkdown(CompareConfig{
		Runs: []LabeledRun{
			{Label: "accelerated", Summary: ours},
			{Label: "livekit", Summary: theirs},
		},
		Baseline: -1,
	})
	if !strings.Contains(md, "accelerated") || !strings.Contains(md, "Pass rate") {
		t.Fatalf("markdown:\n%s", md)
	}
	if !strings.Contains(md, "V2V P50") {
		t.Fatalf("missing latency:\n%s", md)
	}
}

func TestCompareDisclosureReadsThePipelineTriples(t *testing.T) {
	run := func(label, stt, llm, tts string) LabeledRun {
		sum := BuildSummary(label, label, 1, []CallResult{
			{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 400}}}},
		})
		sum.Manifest = RunManifest{TargetSTT: stt, TargetLLM: llm, TargetTTS: tts}
		return LabeledRun{Label: label, Summary: sum}
	}

	differing := CompareMarkdown(CompareConfig{Baseline: -1, Runs: []LabeledRun{
		run("accelerated", "gemini/transcribe", "gemini/flash-lite", "inworld/flash"),
		run("livekit", "", "gpt-realtime-2", ""),
	}})
	if !strings.Contains(differing, "not matched models") {
		t.Fatalf("differing pipelines must be disclosed as a product comparison:\n%s", differing)
	}
	if !strings.Contains(differing, "gpt-realtime-2") || !strings.Contains(differing, "inworld/flash") {
		t.Fatalf("both pipelines must appear:\n%s", differing)
	}

	matched := CompareMarkdown(CompareConfig{Baseline: -1, Runs: []LabeledRun{
		run("accelerated", "gemini/transcribe", "gemini/flash-lite", "inworld/flash"),
		run("livekit", "gemini/transcribe", "gemini/flash-lite", "inworld/flash"),
	}})
	if !strings.Contains(matched, "framework overhead") {
		t.Fatalf("a matched triple must be disclosed as framework overhead:\n%s", matched)
	}
}

func TestCompareBaselineFlagsMDE(t *testing.T) {
	old := BuildSummary("accelerated", "old", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 400}}}},
	})
	newer := BuildSummary("accelerated", "new", 1, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{V2V: []score.Timing{{V2VMS: 520}}}},
	})
	md := CompareMarkdown(CompareConfig{
		Runs: []LabeledRun{
			{Label: "baseline", Summary: old},
			{Label: "new", Summary: newer},
		},
		Baseline: 0,
		MDEV2VMS: 50,
	})
	if !strings.Contains(md, "regression") {
		t.Fatalf("expected regression flag:\n%s", md)
	}
}

func TestCompareShowsFirstResponseWithSampleCount(t *testing.T) {
	run := func(label string, first *score.Timing) LabeledRun {
		return LabeledRun{Label: label, Summary: BuildSummary(label, label, 1, []CallResult{
			{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass, Metrics: score.Metrics{FirstResponse: first}},
		})}
	}
	md := CompareMarkdown(CompareConfig{Baseline: 0, Runs: []LabeledRun{
		run("accelerated", &score.Timing{TurnID: "t1", V2VMS: 700}),
		run("livekit", &score.Timing{TurnID: "t1", V2VMS: 900}),
		run("livekit-inference", nil),
	}})
	if !strings.Contains(md, "| First response P50 (ms) | 700 (n=1) ** | 900 (n=1) | — |") {
		t.Fatalf("first response row missing or a run without samples was marked best:\n%s", md)
	}
	if !strings.Contains(md, "| livekit | +0.0 pp | +0 ms | +200 ms |") {
		t.Fatalf("first response baseline delta missing:\n%s", md)
	}
}
