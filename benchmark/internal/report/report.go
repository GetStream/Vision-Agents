package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

const SchemaVersion = 1

// CallResult is one trial of one scenario.
type CallResult struct {
	ScenarioID string        `json:"scenario_id"`
	Pack       string        `json:"pack"`
	Category   string        `json:"category"`
	Trial      int           `json:"trial"`
	Passed     bool          `json:"passed"`
	Metrics    score.Metrics `json:"metrics"`
	Dir        string        `json:"dir"`
	Error      string        `json:"error,omitempty"`
}

// Summary is the leaderboard-ready document.
type Summary struct {
	SchemaVersion int           `json:"schema_version"`
	System        string        `json:"system"`
	RunID         string        `json:"run_id"`
	Started       time.Time     `json:"started"`
	K             int           `json:"k"`
	Packs         []PackSummary `json:"packs"`
	Calls         []CallResult  `json:"calls"`
}

// PackSummary is one vertical column.
type PackSummary struct {
	Pack       string         `json:"pack"`
	Cells      []CategoryCell `json:"cells"`
	V2VP50     int            `json:"v2v_p50_ms"`
	NonToolP50 int            `json:"non_tool_p50_ms"`
	Spikes     int            `json:"spike_count"`
	Cutoff     float64        `json:"false_cutoff_rate"`
}

// CategoryCell is pass@k / pass^k for one call type.
type CategoryCell struct {
	Category string `json:"category"`
	PassAtK  bool   `json:"pass_at_k"`
	PassHatK bool   `json:"pass_hat_k"`
	Trials   int    `json:"trials"`
	Passed   int    `json:"passed"`
}

// BuildSummary aggregates call results.
func BuildSummary(system, runID string, k int, calls []CallResult) Summary {
	s := Summary{SchemaVersion: SchemaVersion, System: system, RunID: runID, Started: time.Now().UTC(), K: k, Calls: calls}
	packs := map[string][]CallResult{}
	for _, c := range calls {
		packs[c.Pack] = append(packs[c.Pack], c)
	}
	names := make([]string, 0, len(packs))
	for n := range packs {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, pack := range names {
		s.Packs = append(s.Packs, summarizePack(pack, packs[pack], k))
	}
	return s
}

func summarizePack(pack string, calls []CallResult, k int) PackSummary {
	byCat := map[string][]bool{}
	var v2v []int
	var nonTool []int
	spikes := 0
	cutoffs := 0
	for _, c := range calls {
		byCat[c.Category] = append(byCat[c.Category], c.Passed)
		if c.Metrics.V2VP50 > 0 {
			v2v = append(v2v, c.Metrics.V2VP50)
		}
		if c.Metrics.NonToolP50 > 0 {
			nonTool = append(nonTool, c.Metrics.NonToolP50)
		}
		spikes += c.Metrics.SpikeCount
		cutoffs += c.Metrics.FalseCutoff
	}
	cats := make([]string, 0, len(byCat))
	for c := range byCat {
		cats = append(cats, c)
	}
	sort.Strings(cats)
	out := PackSummary{Pack: pack}
	for _, cat := range cats {
		passed := 0
		for _, p := range byCat[cat] {
			if p {
				passed++
			}
		}
		out.Cells = append(out.Cells, CategoryCell{
			Category: cat,
			PassAtK:  score.PassAtK(byCat[cat]),
			PassHatK: score.PassHatK(byCat[cat]),
			Trials:   len(byCat[cat]),
			Passed:   passed,
		})
	}
	if len(v2v) > 0 {
		sort.Ints(v2v)
		out.V2VP50 = v2v[len(v2v)/2]
	}
	if len(nonTool) > 0 {
		sort.Ints(nonTool)
		out.NonToolP50 = nonTool[len(nonTool)/2]
	}
	out.Spikes = spikes
	if len(calls) > 0 {
		out.Cutoff = float64(cutoffs) / float64(len(calls))
	}
	return out
}

// Write dumps summary.json, report.md, and copies per-call already on disk.
func Write(dir string, summary Summary) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "summary.json"), raw, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "report.md"), []byte(Markdown(summary)), 0o644)
}

// Markdown renders the three-column leaderboard view.
func Markdown(s Summary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Voicebench %s\n\n", s.RunID)
	fmt.Fprintf(&b, "System: `%s`  \nK: %d  \nSchema: %d\n\n", s.System, s.K, s.SchemaVersion)
	b.WriteString("| Pack | V2V P50 | Non-tool P50 | Spikes | False cutoff / call |\n| --- | ---: | ---: | ---: | ---: |\n")
	for _, p := range s.Packs {
		fmt.Fprintf(&b, "| %s | %d ms | %d ms | %d | %.2f |\n", p.Pack, p.V2VP50, p.NonToolP50, p.Spikes, p.Cutoff)
	}
	b.WriteString("\n## Pass@k / pass^k\n\n")
	b.WriteString("| Pack | Category | pass@k | pass^k | passed/trials |\n| --- | --- | --- | --- | ---: |\n")
	for _, p := range s.Packs {
		for _, c := range p.Cells {
			fmt.Fprintf(&b, "| %s | %s | %t | %t | %d/%d |\n", p.Pack, c.Category, c.PassAtK, c.PassHatK, c.Passed, c.Trials)
		}
	}
	b.WriteString("\n## Calls\n\n")
	b.WriteString("| Scenario | Trial | Passed | V2V P50 | Non-tool P50 | Spikes | Gates |\n| --- | ---: | --- | ---: | ---: | ---: | --- |\n")
	for _, c := range s.Calls {
		fmt.Fprintf(&b, "| %s | %d | %t | %d | %d | %d | %s |\n", c.ScenarioID, c.Trial, c.Passed, c.Metrics.V2VP50, c.Metrics.NonToolP50, c.Metrics.SpikeCount, strings.Join(c.Metrics.GateNotes, ","))
	}
	b.WriteString("\nHard gates are end-state AND policy AND entity fidelity AND tool order AND say-do AND filler AND barge-in stop AND hold/selectivity. STT, judge, and TTS must succeed. V2V latency and spikes are reported, not gated. Human-band % uses non-tool turns only.\n")
	return b.String()
}
