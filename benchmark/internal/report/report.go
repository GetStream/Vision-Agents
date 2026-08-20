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
const BenchmarkVersion = "0.1.0"
const MethodologyVersion = "voicebench-live-v1"

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
	SchemaVersion      int               `json:"schema_version"`
	BenchmarkVersion   string            `json:"benchmark_version"`
	MethodologyVersion string            `json:"methodology_version"`
	Providers          map[string]string `json:"providers"`
	System             string            `json:"system"`
	RunID              string            `json:"run_id"`
	Started            time.Time         `json:"started"`
	K                  int               `json:"k"`
	Packs              []PackSummary     `json:"packs"`
	Calls              []CallResult      `json:"calls"`
}

// PackSummary is one vertical column.
type PackSummary struct {
	Pack             string         `json:"pack"`
	Cells            []CategoryCell `json:"cells"`
	V2VP50           int            `json:"v2v_p50_ms"`
	NonToolP50       int            `json:"non_tool_p50_ms"`
	Spikes           int            `json:"spike_count"`
	Cutoff           float64        `json:"false_cutoff_rate"`
	CallDurationP50  int            `json:"call_duration_p50_ms"`
	ToolCountPerCall float64        `json:"tool_count_per_call"`
	ToolErrors       int            `json:"tool_errors"`
	ToolWaitP50      int            `json:"tool_wait_p50_ms"`
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
	s := Summary{
		SchemaVersion:      SchemaVersion,
		BenchmarkVersion:   BenchmarkVersion,
		MethodologyVersion: MethodologyVersion,
		Providers:          defaultProviders(),
		System:             system,
		RunID:              runID,
		Started:            time.Now().UTC(),
		K:                  k,
		Calls:              calls,
	}
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

func defaultProviders() map[string]string {
	return map[string]string{
		"caller_tts": "elevenlabs",
		"stt":        "deepgram:nova-3",
		"judge":      "openai:gpt-4.1-mini",
	}
}

func summarizePack(pack string, calls []CallResult, k int) PackSummary {
	byCat := map[string][]bool{}
	var v2v []int
	var nonTool []int
	var durations []int
	var toolWait []int
	spikes := 0
	cutoffs := 0
	toolCount := 0
	toolErrors := 0
	for _, c := range calls {
		byCat[c.Category] = append(byCat[c.Category], c.Passed)
		if c.Metrics.V2VP50 > 0 {
			v2v = append(v2v, c.Metrics.V2VP50)
		}
		if c.Metrics.NonToolP50 > 0 {
			nonTool = append(nonTool, c.Metrics.NonToolP50)
		}
		if c.Metrics.CallDurationMS > 0 {
			durations = append(durations, c.Metrics.CallDurationMS)
		}
		if c.Metrics.ToolWaitMS > 0 {
			toolWait = append(toolWait, c.Metrics.ToolWaitMS)
		}
		spikes += c.Metrics.SpikeCount
		cutoffs += c.Metrics.FalseCutoff
		toolCount += c.Metrics.ToolCount
		toolErrors += c.Metrics.ToolErrorCount
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
	if len(durations) > 0 {
		sort.Ints(durations)
		out.CallDurationP50 = durations[len(durations)/2]
	}
	if len(toolWait) > 0 {
		sort.Ints(toolWait)
		out.ToolWaitP50 = toolWait[len(toolWait)/2]
	}
	out.Spikes = spikes
	out.ToolErrors = toolErrors
	if len(calls) > 0 {
		out.Cutoff = float64(cutoffs) / float64(len(calls))
		out.ToolCountPerCall = float64(toolCount) / float64(len(calls))
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
	fmt.Fprintf(&b, "System: `%s`  \nK: %d  \nSchema: %d  \nBenchmark: `%s`  \nMethodology: `%s`\n\n", s.System, s.K, s.SchemaVersion, s.BenchmarkVersion, s.MethodologyVersion)
	b.WriteString("## Methodology\n\n")
	b.WriteString("Voicebench evaluates live voice agents through scripted calls against a seeded scenario backend. A trial passes only if all hard gates pass: final state, expected tools and arguments, tool order, entity fidelity, policy, say-do consistency, filler behavior, barge-in, and selectivity. Latency is reported separately unless it affects interruption/selectivity gates. Each scenario is repeated k times; pass@k means any trial passed, pass^k means every trial passed. Targets are Voicebench acceptance thresholds, not universal industry standards or state-of-the-art claims.\n\n")
	b.WriteString("## Scorecard\n\n")
	b.WriteString("| Benchmark | Target | Ours | Gap |\n| --- | --- | --- | --- |\n")
	for _, r := range Scorecard(s) {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", r.Name, r.Target, r.Ours, r.Gap)
	}
	b.WriteString("\n## Failure Summary\n\n")
	failures := failureSummary(s.Calls)
	if len(failures) == 0 {
		b.WriteString("No hard-gate failures.\n")
	} else {
		b.WriteString("| Gate | Count |\n| --- | ---: |\n")
		for _, f := range failures {
			fmt.Fprintf(&b, "| %s | %d |\n", f.Name, f.Count)
		}
	}
	b.WriteString("\n## Latency\n\n")
	b.WriteString("| Pack | V2V P50 | Non-tool P50 | Spikes | False cutoff / call |\n| --- | ---: | ---: | ---: | ---: |\n")
	for _, p := range s.Packs {
		fmt.Fprintf(&b, "| %s | %d ms | %d ms | %d | %.2f |\n", p.Pack, p.V2VP50, p.NonToolP50, p.Spikes, p.Cutoff)
	}
	b.WriteString("\n## Operations\n\n")
	b.WriteString("| Pack | Call duration P50 | Tool count / call | Tool errors | Tool wait P50 |\n| --- | ---: | ---: | ---: | ---: |\n")
	for _, p := range s.Packs {
		fmt.Fprintf(&b, "| %s | %d ms | %.2f | %d | %d ms |\n", p.Pack, p.CallDurationP50, p.ToolCountPerCall, p.ToolErrors, p.ToolWaitP50)
	}
	b.WriteString("\n## Pass@k / pass^k\n\n")
	b.WriteString("| Pack | Category | pass@k | pass^k | passed/trials |\n| --- | --- | --- | --- | ---: |\n")
	for _, p := range s.Packs {
		for _, c := range p.Cells {
			fmt.Fprintf(&b, "| %s | %s | %t | %t | %d/%d |\n", p.Pack, c.Category, c.PassAtK, c.PassHatK, c.Passed, c.Trials)
		}
	}
	b.WriteString("\n## Calls\n\n")
	b.WriteString("| Scenario | Trial | Passed | Duration | V2V P50 | Non-tool P50 | Tools | Tool wait | Spikes | Gates | Artifacts |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
	for _, c := range s.Calls {
		fmt.Fprintf(&b, "| %s | %d | %t | %d ms | %d ms | %d ms | %d | %d ms | %d | %s | %s |\n", c.ScenarioID, c.Trial, c.Passed, c.Metrics.CallDurationMS, c.Metrics.V2VP50, c.Metrics.NonToolP50, c.Metrics.ToolCount, c.Metrics.ToolWaitMS, c.Metrics.SpikeCount, strings.Join(c.Metrics.GateNotes, ","), artifactLinks(c))
	}
	b.WriteString("\nHard gates are end-state AND expected tools/arguments AND policy AND entity fidelity AND tool order AND say-do AND filler AND barge-in stop AND hold/selectivity. STT, judge, and TTS must succeed. V2V latency and spikes are reported, not gated. Human-band % uses non-tool turns only.\n")
	return b.String()
}

type failureCount struct {
	Name  string
	Count int
}

func failureSummary(calls []CallResult) []failureCount {
	counts := map[string]int{}
	for _, c := range calls {
		for _, note := range c.Metrics.GateNotes {
			if note != "" {
				counts[note]++
			}
		}
		if c.Error != "" {
			counts["trial_error"]++
		}
	}
	names := make([]string, 0, len(counts))
	for name := range counts {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]failureCount, 0, len(names))
	for _, name := range names {
		out = append(out, failureCount{Name: name, Count: counts[name]})
	}
	return out
}

func artifactLinks(c CallResult) string {
	if c.Dir == "" {
		return ""
	}
	dir := filepath.Base(c.Dir)
	return fmt.Sprintf("[audio](%s/mixed.wav) [transcript](%s/transcript.json) [tools](%s/tools.json) [state](%s/state.json) [metrics](%s/metrics.json)", dir, dir, dir, dir, dir)
}
