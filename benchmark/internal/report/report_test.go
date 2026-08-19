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
	dir := t.TempDir()
	if err := Write(dir, sum); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "summary.json")); err != nil {
		t.Fatal(err)
	}
}
