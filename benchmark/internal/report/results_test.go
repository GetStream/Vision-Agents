package report

import (
	"strings"
	"testing"
)

func outcomeCall(id, outcome string) CallResult {
	pack, _, _ := strings.Cut(id, ".")
	return CallResult{ScenarioID: id, Pack: pack, Outcome: outcome, Passed: outcome == OutcomePass}
}

func TestSummarizeResultsCountsByPackAndScenario(t *testing.T) {
	results := SummarizeResults([]CallResult{
		outcomeCall("healthcare.coherence", OutcomePass),
		outcomeCall("healthcare.interrupt", OutcomePass),
		outcomeCall("healthcare.selectivity", OutcomeFail),
		outcomeCall("healthcare.tool_filler", OutcomeFail),
		outcomeCall("restaurant.coherence", OutcomeFail),
		outcomeCall("restaurant.interrupt", OutcomeInvalid),
		outcomeCall("telecom.coherence", OutcomeFail),
	})
	if got := results.Overall.Text(); got != "2/6 passed, 1 invalid" {
		t.Fatalf("overall = %q: an invalid trial has no verdict, so it is counted beside the total", got)
	}
	// healthcare 50, restaurant 0, telecom 0: the mean of the packs, not 2 of 6.
	if results.Score != 17 {
		t.Fatalf("score = %d, want 17", results.Score)
	}
	var coherence GroupResult
	for _, kind := range results.ByKind {
		if kind.Name == "coherence" {
			coherence = kind
		}
	}
	if coherence.Passed != 1 || coherence.Valid != 3 || coherence.Score() != 33 {
		t.Fatalf("coherence across packs = %+v", coherence)
	}
	if len(results.ByPack) != 3 || results.ByPack[1].Name != "restaurant" || results.ByPack[1].Invalid != 1 {
		t.Fatalf("by pack = %+v", results.ByPack)
	}
}

func TestSummarizeResultsHasNoScoreWithoutAVerdict(t *testing.T) {
	results := SummarizeResults([]CallResult{outcomeCall("telecom.golden", OutcomeInvalid)})
	if results.Score != -1 || results.ScoreText() != "—" || results.Overall.Text() != "0/0 passed, 1 invalid" {
		t.Fatalf("a run with nothing scored must not read as 0: %+v", results)
	}
}

func TestDigestSaysHowManyPassed(t *testing.T) {
	runs := []LabeledRun{steadyRun("accelerated", "restaurant", 1800, 1, 2)}
	d := BuildDigest(runs)
	if text := d.SlackText("Voicebench"); !strings.Contains(text, "• Passed: accelerated 1/2 passed · score 50 (restaurant 50)") {
		t.Fatalf("slack text:\n%s", text)
	}
	page, err := d.HTML("Voicebench", []byte("png"), runs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(page, "<strong>1/2</strong>") || !strings.Contains(page, "By scenario") {
		t.Fatalf("the report should carry the results table")
	}
}
