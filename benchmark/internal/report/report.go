package report

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

const SchemaVersion = 3
const BenchmarkVersion = "0.4.0"
const MethodologyVersion = "voicebench-live-v3"

const KindAgent = "agent"
const KindSTT = "stt"
const KindTTS = "tts"

const (
	OutcomePass    = "pass"
	OutcomeFail    = "fail"
	OutcomeInvalid = "invalid"
)

// CallResult is one trial of one scenario.
type CallResult struct {
	ScenarioID    string        `json:"scenario_id"`
	Pack          string        `json:"pack"`
	Category      string        `json:"category"`
	Trial         int           `json:"trial"`
	Outcome       string        `json:"outcome"`
	Passed        bool          `json:"passed"`
	InvalidReason []string      `json:"invalid_reasons,omitempty"`
	Metrics       score.Metrics `json:"metrics"`
	Dir           string        `json:"dir"`
	Error         string        `json:"error,omitempty"`
	Warnings      []string      `json:"warnings,omitempty"`
}

// RunManifest identifies the code, inputs, and runtime configuration behind a result.
type RunManifest struct {
	GitCommit                string            `json:"git_commit"`
	GitDirty                 bool              `json:"git_dirty"`
	ScenarioHash             string            `json:"scenario_hash"`
	ContractHash             string            `json:"contract_hash"`
	Transport                string            `json:"transport"`
	Target                   string            `json:"target"`
	TargetModel              string            `json:"target_model,omitempty"`
	TargetVoice              string            `json:"target_voice,omitempty"`
	TargetSTT                string            `json:"target_stt,omitempty"`
	TargetLLM                string            `json:"target_llm,omitempty"`
	TargetTTS                string            `json:"target_tts,omitempty"`
	TargetSubagent           string            `json:"target_subagent,omitempty"`
	CallerModel              string            `json:"caller_model"`
	CallerVoice              string            `json:"caller_voice"`
	GoVersion                string            `json:"go_version"`
	NetworkProfile           string            `json:"network_profile,omitempty"`
	JudgeCalibrationHash     string            `json:"judge_calibration_hash,omitempty"`
	JudgeCalibrationReviewer string            `json:"judge_calibration_reviewer,omitempty"`
	Command                  []string          `json:"command"`
	Labels                   map[string]string `json:"labels,omitempty"`
}

// Summary is the leaderboard-ready document.
type Summary struct {
	SchemaVersion      int               `json:"schema_version"`
	BenchmarkVersion   string            `json:"benchmark_version"`
	MethodologyVersion string            `json:"methodology_version"`
	Kind               string            `json:"kind"`
	Providers          map[string]string `json:"providers"`
	Manifest           RunManifest       `json:"manifest"`
	System             string            `json:"system"`
	RunID              string            `json:"run_id"`
	Started            time.Time         `json:"started"`
	K                  int               `json:"k"`
	Packs              []PackSummary     `json:"packs"`
	Calls              []CallResult      `json:"calls"`
}

// PackSummary is one vertical column.
type PackSummary struct {
	Pack             string            `json:"pack"`
	Scenarios        []ScenarioSummary `json:"scenarios"`
	Cells            []CategoryCell    `json:"cells"`
	V2VP50           int               `json:"v2v_p50_ms"`
	V2VP95           int               `json:"v2v_p95_ms"`
	NonToolP50       int               `json:"non_tool_p50_ms"`
	V2VSamples       int               `json:"v2v_samples"`
	NonToolSamples   int               `json:"non_tool_samples"`
	DroppedTurns     int               `json:"dropped_turns"`
	Spikes           int               `json:"spike_count"`
	Cutoff           float64           `json:"false_cutoff_rate"`
	CallDurationP50  int               `json:"call_duration_p50_ms"`
	ToolCountPerCall float64           `json:"tool_count_per_call"`
	ToolErrors       int               `json:"tool_errors"`
	ToolWaitP50      int               `json:"tool_wait_p50_ms"`
	CallerTurnsP50   int               `json:"caller_turns_p50"`
	AgentTurnsP50    int               `json:"agent_turns_p50"`
}

// CategoryCell is pass@k / pass^k for one call type.
type CategoryCell struct {
	Category string `json:"category"`
	Complete bool   `json:"complete"`
	PassAtK  bool   `json:"pass_at_k"`
	PassHatK bool   `json:"pass_hat_k"`
	Trials   int    `json:"trials"`
	Passed   int    `json:"passed"`
	Invalid  int    `json:"invalid"`
}

// ScenarioSummary is the reliability result for exactly one scenario.
type ScenarioSummary struct {
	ScenarioID string  `json:"scenario_id"`
	Category   string  `json:"category"`
	Complete   bool    `json:"complete"`
	PassAtK    bool    `json:"pass_at_k"`
	PassHatK   bool    `json:"pass_hat_k"`
	Requested  int     `json:"requested"`
	Valid      int     `json:"valid"`
	Passed     int     `json:"passed"`
	Invalid    int     `json:"invalid"`
	PassRate   float64 `json:"pass_rate"`
	CI95Low    float64 `json:"ci95_low"`
	CI95High   float64 `json:"ci95_high"`
}

// BuildSummary aggregates call results.
func BuildSummary(system, runID string, k int, calls []CallResult) Summary {
	s := Summary{
		SchemaVersion:      SchemaVersion,
		BenchmarkVersion:   BenchmarkVersion,
		MethodologyVersion: MethodologyVersion,
		Kind:               KindAgent,
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
		"judge":      "openai:" + score.JudgeModel,
	}
}

func summarizePack(pack string, calls []CallResult, k int) PackSummary {
	byScenario := map[string][]CallResult{}
	var v2v []int
	var nonTool []int
	var durations []int
	var toolWait []int
	var callerTurns []int
	var agentTurns []int
	spikes := 0
	cutoffs := 0
	toolCount := 0
	toolErrors := 0
	dropped := 0
	validCalls := 0
	for _, call := range calls {
		byScenario[call.ScenarioID] = append(byScenario[call.ScenarioID], call)
		if callOutcome(call) == OutcomeInvalid {
			continue
		}
		validCalls++
		// Pool raw samples rather than per-call percentiles.
		for _, timing := range call.Metrics.V2V {
			if timing.V2VMS < 0 {
				continue
			}
			v2v = append(v2v, timing.V2VMS)
			if !timing.Tool {
				nonTool = append(nonTool, timing.V2VMS)
			}
		}
		dropped += len(call.Metrics.Dropped)
		if call.Metrics.CallDurationMS > 0 {
			durations = append(durations, call.Metrics.CallDurationMS)
		}
		if call.Metrics.ToolWaitMS > 0 {
			toolWait = append(toolWait, call.Metrics.ToolWaitMS)
		}
		if call.Metrics.CallerTurns > 0 {
			callerTurns = append(callerTurns, call.Metrics.CallerTurns)
		}
		if call.Metrics.AgentTurns > 0 {
			agentTurns = append(agentTurns, call.Metrics.AgentTurns)
		}
		spikes += call.Metrics.SpikeCount
		cutoffs += call.Metrics.FalseCutoff
		toolCount += call.Metrics.ToolCount
		toolErrors += call.Metrics.ToolErrorCount
	}

	scenarioIDs := make([]string, 0, len(byScenario))
	for id := range byScenario {
		scenarioIDs = append(scenarioIDs, id)
	}
	sort.Strings(scenarioIDs)
	out := PackSummary{Pack: pack}
	byCategory := map[string][]ScenarioSummary{}
	for _, id := range scenarioIDs {
		scenarioCalls := byScenario[id]
		summary := ScenarioSummary{
			ScenarioID: id,
			Category:   scenarioCalls[0].Category,
			Requested:  len(scenarioCalls),
		}
		var passed []bool
		for _, call := range scenarioCalls {
			switch callOutcome(call) {
			case OutcomeInvalid:
				summary.Invalid++
			case OutcomePass:
				summary.Valid++
				summary.Passed++
				passed = append(passed, true)
			default:
				summary.Valid++
				passed = append(passed, false)
			}
		}
		summary.Complete = summary.Requested == k && summary.Invalid == 0
		summary.PassAtK = summary.Complete && score.PassAtK(passed)
		summary.PassHatK = summary.Complete && score.PassHatK(passed)
		summary.PassRate, summary.CI95Low, summary.CI95High = wilson95(summary.Passed, summary.Valid)
		out.Scenarios = append(out.Scenarios, summary)
		byCategory[summary.Category] = append(byCategory[summary.Category], summary)
	}

	categories := make([]string, 0, len(byCategory))
	for category := range byCategory {
		categories = append(categories, category)
	}
	sort.Strings(categories)
	for _, category := range categories {
		cell := CategoryCell{Category: category, Complete: true, PassAtK: true, PassHatK: true}
		for _, summary := range byCategory[category] {
			cell.Trials += summary.Requested
			cell.Passed += summary.Passed
			cell.Invalid += summary.Invalid
			cell.Complete = cell.Complete && summary.Complete
			cell.PassAtK = cell.PassAtK && summary.PassAtK
			cell.PassHatK = cell.PassHatK && summary.PassHatK
		}
		out.Cells = append(out.Cells, cell)
	}
	if len(v2v) > 0 {
		sort.Ints(v2v)
		out.V2VP50 = score.Percentile(v2v, 50)
		out.V2VP95 = score.Percentile(v2v, 95)
	}
	if len(nonTool) > 0 {
		sort.Ints(nonTool)
		out.NonToolP50 = score.Percentile(nonTool, 50)
	}
	if len(durations) > 0 {
		sort.Ints(durations)
		out.CallDurationP50 = score.Percentile(durations, 50)
	}
	if len(toolWait) > 0 {
		sort.Ints(toolWait)
		out.ToolWaitP50 = score.Percentile(toolWait, 50)
	}
	if len(callerTurns) > 0 {
		sort.Ints(callerTurns)
		out.CallerTurnsP50 = score.Percentile(callerTurns, 50)
	}
	if len(agentTurns) > 0 {
		sort.Ints(agentTurns)
		out.AgentTurnsP50 = score.Percentile(agentTurns, 50)
	}
	out.V2VSamples = len(v2v)
	out.NonToolSamples = len(nonTool)
	out.DroppedTurns = dropped
	out.Spikes = spikes
	out.ToolErrors = toolErrors
	if validCalls > 0 {
		out.Cutoff = float64(cutoffs) / float64(validCalls)
		out.ToolCountPerCall = float64(toolCount) / float64(validCalls)
	}
	return out
}

func wilson95(successes, trials int) (float64, float64, float64) {
	if trials == 0 {
		return 0, 0, 0
	}
	p := float64(successes) / float64(trials)
	const z = 1.959963984540054
	n := float64(trials)
	denominator := 1 + z*z/n
	center := (p + z*z/(2*n)) / denominator
	margin := z * math.Sqrt(p*(1-p)/n+z*z/(4*n*n)) / denominator
	return p, max(0, center-margin), min(1, center+margin)
}

func callOutcome(call CallResult) string {
	if call.Outcome != "" {
		return call.Outcome
	}
	if call.Error != "" || len(call.InvalidReason) > 0 {
		return OutcomeInvalid
	}
	if call.Passed {
		return OutcomePass
	}
	return OutcomeFail
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
	manifest, err := json.MarshalIndent(summary.Manifest, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), manifest, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "report.md"), []byte(Markdown(summary)), 0o644)
}

// Markdown renders the three-column leaderboard view.
func Markdown(s Summary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Voicebench %s\n\n", s.RunID)
	fmt.Fprintf(&b, "System: `%s`  \nKind: `%s`  \nK: %d  \nSchema: %d  \nBenchmark: `%s`  \nMethodology: `%s`\n\n", s.System, s.kind(), s.K, s.SchemaVersion, s.BenchmarkVersion, s.MethodologyVersion)
	b.WriteString("## Methodology\n\n")
	b.WriteString("Voicebench evaluates live voice agents through scripted calls against a seeded scenario backend. Trials are pass, fail, or invalid; evaluator failures are invalid and never count as agent failures. Reliability is computed per scenario. pass@k means at least one of the requested valid trials passed, pass^k means every requested trial passed, and neither is awarded when the requested trial set is incomplete. Latency is reported separately unless it affects interruption/selectivity gates. Targets are Voicebench acceptance thresholds, not universal industry standards or state-of-the-art claims.\n\n")
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
	warnings := runWarnings(s)
	warnings = append(warnings, callWarnings(s.Calls)...)
	if len(warnings) > 0 {
		b.WriteString("\n## Warnings\n\n")
		for _, w := range warnings {
			fmt.Fprintf(&b, "- %s\n", w)
		}
	}
	b.WriteString("\n## Latency\n\n")
	b.WriteString("| Pack | V2V P50 | V2V P95 | Non-tool P50 | Spikes | Turns dropped | False cutoff / call |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, p := range s.Packs {
		fmt.Fprintf(&b, "| %s | %d ms (n=%d) | %d ms | %d ms (n=%d) | %d | %d | %.2f |\n", p.Pack, p.V2VP50, p.V2VSamples, p.V2VP95, p.NonToolP50, p.NonToolSamples, p.Spikes, p.DroppedTurns, p.Cutoff)
	}
	b.WriteString("\n## Operations\n\n")
	b.WriteString("| Pack | Call duration P50 | Tool count / call | Tool errors | Tool wait P50 | Caller turns P50 | Agent turns P50 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, p := range s.Packs {
		fmt.Fprintf(&b, "| %s | %d ms | %.2f | %d | %d ms | %d | %d |\n", p.Pack, p.CallDurationP50, p.ToolCountPerCall, p.ToolErrors, p.ToolWaitP50, p.CallerTurnsP50, p.AgentTurnsP50)
	}
	b.WriteString("\n## Pass@k / pass^k\n\n")
	b.WriteString("| Pack | Scenario | Category | complete | pass@k | pass^k | passed/valid | pass rate (95% CI) | invalid |\n| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |\n")
	for _, pack := range s.Packs {
		for _, summary := range pack.Scenarios {
			fmt.Fprintf(&b, "| %s | %s | %s | %t | %t | %t | %d/%d | %.1f%% (%.1f–%.1f%%) | %d |\n", pack.Pack, summary.ScenarioID, summary.Category, summary.Complete, summary.PassAtK, summary.PassHatK, summary.Passed, summary.Valid, 100*summary.PassRate, 100*summary.CI95Low, 100*summary.CI95High, summary.Invalid)
		}
	}
	b.WriteString("\n## Calls\n\n")
	b.WriteString("| Scenario | Trial | Outcome | Duration | V2V P50 | Non-tool P50 | Tools | Tool wait | Spikes | Gates | Artifacts |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
	for _, call := range s.Calls {
		fmt.Fprintf(&b, "| %s | %d | %s | %d ms | %d ms | %d ms | %d | %d ms | %d | %s | %s |\n", call.ScenarioID, call.Trial, callOutcome(call), call.Metrics.CallDurationMS, call.Metrics.V2VP50, call.Metrics.NonToolP50, call.Metrics.ToolCount, call.Metrics.ToolWaitMS, call.Metrics.SpikeCount, strings.Join(call.Metrics.GateNotes, ","), artifactLinks(call))
	}
	b.WriteString("\nP50s are pooled over every measured turn in the pack, not a median of per-call medians; n is that sample count. Turns dropped are scripted turns with no usable reply gap, listed per call in `metrics.json` under `dropped_turns`.\n")
	b.WriteString("\nHard gates are end-state AND successful expected tools/arguments AND policy AND entity fidelity AND tool order AND say-do AND filler AND barge-in stop AND hold/selectivity. Required evaluator failures make a trial invalid rather than failed. V2V latency and spikes are reported, not gated. Human-band % uses non-tool turns only.\n")
	return b.String()
}

func (s Summary) kind() string {
	if s.Kind != "" {
		return s.Kind
	}
	return KindAgent
}

// InvalidTrials counts trials that produced no verdict.
func (s Summary) InvalidTrials() int {
	invalid := 0
	for _, call := range s.Calls {
		if callOutcome(call) == OutcomeInvalid {
			invalid++
		}
	}
	return invalid
}

func runWarnings(summary Summary) []string {
	var warnings []string
	if summary.Manifest.GitDirty {
		warnings = append(warnings, "source tree is dirty; this run is not an immutable baseline")
	}
	if summary.Manifest.NetworkProfile == "" {
		warnings = append(warnings, "network profile is missing; network comparability is unknown")
	}
	if summary.Manifest.JudgeCalibrationHash == "" {
		warnings = append(warnings, "judge calibration fixture is missing")
	} else if summary.Manifest.JudgeCalibrationReviewer == "" {
		warnings = append(warnings, "judge calibration labels have not been human-reviewed")
	}
	return warnings
}

func callWarnings(calls []CallResult) []string {
	var out []string
	for _, c := range calls {
		for _, w := range c.Warnings {
			out = append(out, fmt.Sprintf("%s trial %d: %s", c.ScenarioID, c.Trial, w))
		}
		if callOutcome(c) == OutcomeInvalid {
			reason := c.Error
			if reason == "" {
				reason = strings.Join(c.InvalidReason, ", ")
			}
			out = append(out, fmt.Sprintf("%s trial %d: invalid, reliability incomplete (%s)", c.ScenarioID, c.Trial, reason))
		} else if callOutcome(c) == OutcomeFail && c.Error != "" {
			out = append(out, fmt.Sprintf("%s trial %d: target failure (%s)", c.ScenarioID, c.Trial, c.Error))
		}
	}
	return out
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
		if callOutcome(c) == OutcomeInvalid {
			counts["evaluator_invalid"]++
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
	var links []string
	for _, artifact := range []struct {
		name string
		file string
	}{
		{name: "audio", file: "mixed.wav"},
		{name: "transcript", file: "transcript.json"},
		{name: "judge", file: "judge.json"},
		{name: "tools", file: "tools.json"},
		{name: "state", file: "state.json"},
		{name: "events", file: "events.json"},
		{name: "metrics", file: "metrics.json"},
	} {
		if _, err := os.Stat(filepath.Join(c.Dir, artifact.file)); err == nil {
			links = append(links, fmt.Sprintf("[%s](%s/%s)", artifact.name, dir, artifact.file))
		}
	}
	return strings.Join(links, " ")
}
