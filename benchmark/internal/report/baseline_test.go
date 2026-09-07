package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResolveBaselinePrefersARunDirectory(t *testing.T) {
	root := t.TempDir()
	run := filepath.Join(root, "out", "run-1")
	if err := os.MkdirAll(run, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(run, "summary.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ResolveBaseline(root, run)
	if err != nil {
		t.Fatal(err)
	}
	if got != run {
		t.Fatalf("got %s", got)
	}
}

func TestResolveBaselinePicksTheNewestStoredCommit(t *testing.T) {
	root := t.TempDir()
	older := filepath.Join(root, "baselines", "accelerated", "aaa1111")
	newer := filepath.Join(root, "baselines", "accelerated", "000ffff")
	for _, dir := range []string{older, newer} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(older, "summary.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(filepath.Join(newer, "summary.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ResolveBaseline(root, "accelerated")
	if err != nil {
		t.Fatal(err)
	}
	if got != newer {
		t.Fatalf("got %s want %s", got, newer)
	}
}

func TestStoreBaselineMergesASecondPack(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "out", "restaurant")
	second := filepath.Join(root, "out", "healthcare")
	for _, dir := range []string{first, second} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	writeSummary(t, first, BuildSummary("accelerated", "r1", 3, []CallResult{
		{ScenarioID: "restaurant.golden", Pack: "restaurant", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass},
	}))
	writeSummary(t, second, BuildSummary("accelerated", "r2", 3, []CallResult{
		{ScenarioID: "healthcare.golden", Pack: "healthcare", Category: "golden", Trial: 1, Passed: true, Outcome: OutcomePass},
	}))
	if err := os.WriteFile(filepath.Join(first, "manifest.json"), []byte(`{"git_commit":"abc"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(second, "manifest.json"), []byte(`{"git_commit":"abc"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := StoreBaseline(root, "accelerated", "abc1234", first); err != nil {
		t.Fatal(err)
	}
	if err := StoreBaseline(root, "accelerated", "abc1234", second); err != nil {
		t.Fatal(err)
	}
	sum, err := LoadSummary(filepath.Join(root, "baselines", "accelerated", "abc1234"))
	if err != nil {
		t.Fatal(err)
	}
	if len(sum.Calls) != 2 {
		t.Fatalf("%d calls", len(sum.Calls))
	}
	if len(sum.Packs) != 2 {
		t.Fatalf("%d packs", len(sum.Packs))
	}
}

func writeSummary(t *testing.T, dir string, sum Summary) {
	t.Helper()
	raw, err := json.Marshal(sum)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "summary.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestStoreBaselineCopiesTheRun(t *testing.T) {
	root := t.TempDir()
	run := filepath.Join(root, "out")
	if err := os.MkdirAll(run, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(run, "summary.json"), []byte(`{"k":3}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(run, "manifest.json"), []byte(`{"git_commit":"abc"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := StoreBaseline(root, "accelerated", "abc1234", run); err != nil {
		t.Fatal(err)
	}
	got, err := ResolveBaseline(root, "accelerated")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(got, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"k":3}` {
		t.Fatalf("%s", raw)
	}
}
