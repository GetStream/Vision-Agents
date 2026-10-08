package report

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
)

// NoiseFloor is the run-to-run spread of one unchanged target, measured from repeated runs
// of the same commit, scenarios, contracts and network profile. Each metric's MDE is the
// largest difference between any two of those runs: a change no bigger than that cannot be
// told from noise.
type NoiseFloor struct {
	MethodologyVersion string         `json:"methodology_version"`
	Target             string         `json:"target"`
	GitCommit          string         `json:"git_commit"`
	NetworkProfile     string         `json:"network_profile"`
	ScenarioHash       string         `json:"scenario_hash"`
	ContractHash       string         `json:"contract_hash"`
	Packs              []string       `json:"packs"`
	K                  int            `json:"k"`
	Runs               []string       `json:"runs"`
	Metrics            []MetricSpread `json:"metrics"`
}

// MetricSpread is one metric's value in each run and the spread between them.
type MetricSpread struct {
	Name   string    `json:"name"`
	Unit   string    `json:"unit"`
	Values []float64 `json:"values"`
	Mean   float64   `json:"mean"`
	StdDev float64   `json:"stddev"`
	MDE    float64   `json:"mde"`
}

// minNoiseRuns is the fewest repeated runs a noise floor is drawn from. Five is recommended.
const minNoiseRuns = 3

type noiseMetric struct {
	name         string
	unit         string
	higherBetter bool
	value        func(runStats) (float64, bool)
}

// noiseMetrics are the metrics a noise floor measures and compare gates on.
var noiseMetrics = []noiseMetric{
	{"pass_rate", "pp", true, func(s runStats) (float64, bool) {
		return 100 * float64(s.Passed) / float64(max(s.Valid, 1)), s.Valid > 0
	}},
	{"v2v_p50_ms", "ms", false, func(s runStats) (float64, bool) {
		return float64(s.V2VP50), len(s.nonTool)+len(s.tool) > 0
	}},
	{"non_tool_p50_ms", "ms", false, func(s runStats) (float64, bool) {
		return float64(s.NonToolP50), len(s.nonTool) > 0
	}},
	{"non_tool_p95_ms", "ms", false, func(s runStats) (float64, bool) {
		return float64(p95(s.nonTool)), len(s.nonTool) > 0
	}},
	{"tool_p50_ms", "ms", false, func(s runStats) (float64, bool) {
		return float64(p50(s.tool)), len(s.tool) > 0
	}},
	{"first_response_p50_ms", "ms", false, func(s runStats) (float64, bool) {
		return float64(s.FirstResponseP50), s.FirstResponseSamples > 0
	}},
}

// MeasureNoise reads repeated runs of one unchanged target and records each metric's spread.
// It refuses runs that differ in anything that would make them different series.
func MeasureNoise(runs []LabeledRun) (NoiseFloor, error) {
	if len(runs) < minNoiseRuns {
		return NoiseFloor{}, fmt.Errorf("noise: need at least %d runs of the same target, got %d", minNoiseRuns, len(runs))
	}
	first := runs[0].Summary
	out := NoiseFloor{
		MethodologyVersion: first.MethodologyVersion,
		Target:             first.Manifest.Target,
		GitCommit:          first.Manifest.GitCommit,
		NetworkProfile:     first.Manifest.NetworkProfile,
		ScenarioHash:       first.Manifest.ScenarioHash,
		ContractHash:       first.Manifest.ContractHash,
		Packs:              packsOf(first),
		K:                  first.K,
	}
	for _, run := range runs {
		if err := out.matches(run.Summary); err != nil {
			return NoiseFloor{}, fmt.Errorf("noise: %s: %w", run.Label, err)
		}
		// The same run twice would shrink the spread it is meant to measure.
		if slices.Contains(out.Runs, run.Summary.RunID) {
			return NoiseFloor{}, fmt.Errorf("noise: %s: run %s is given twice", run.Label, run.Summary.RunID)
		}
		out.Runs = append(out.Runs, run.Summary.RunID)
	}
	stats := make([]runStats, len(runs))
	for i, run := range runs {
		stats[i] = summarizeRun(run.Summary)
	}
	for _, metric := range noiseMetrics {
		spread := MetricSpread{Name: metric.name, Unit: metric.unit}
		for _, st := range stats {
			if v, ok := metric.value(st); ok {
				spread.Values = append(spread.Values, v)
			}
		}
		// A metric some runs could not measure has no spread to read.
		if len(spread.Values) < len(stats) {
			continue
		}
		spread.Mean, spread.StdDev = meanStdDev(spread.Values)
		spread.MDE = slices.Max(spread.Values) - slices.Min(spread.Values)
		out.Metrics = append(out.Metrics, spread)
	}
	return out, nil
}

// matches reports the first field on which sum is not a repeat of the runs behind the noise
// floor: a different series, commit or k.
func (n NoiseFloor) matches(sum Summary) error {
	if err := n.matchesSeries(sum); err != nil {
		return err
	}
	return firstMismatch([]fieldPair{
		{"git_commit", n.GitCommit, sum.Manifest.GitCommit},
		{"k", fmt.Sprint(n.K), fmt.Sprint(sum.K)},
	})
}

// matchesSeries reports the first field on which sum belongs to a different series from the
// noise floor, which then says nothing about its spread. Commit and k may differ.
func (n NoiseFloor) matchesSeries(sum Summary) error {
	m := sum.Manifest
	return firstMismatch([]fieldPair{
		{"methodology_version", n.MethodologyVersion, sum.MethodologyVersion},
		{"target", n.Target, m.Target},
		{"network_profile", n.NetworkProfile, m.NetworkProfile},
		{"scenario_hash", n.ScenarioHash, m.ScenarioHash},
		{"contract_hash", n.ContractHash, m.ContractHash},
		{"packs", strings.Join(n.Packs, ","), strings.Join(packsOf(sum), ",")},
	})
}

type fieldPair struct{ name, want, got string }

func firstMismatch(fields []fieldPair) error {
	for _, field := range fields {
		if field.want != field.got {
			return fmt.Errorf("%s is %q, want %q", field.name, field.got, field.want)
		}
	}
	return nil
}

// LoadNoiseFloor reads a noise floor written by voicebench noise.
func LoadNoiseFloor(path string) (NoiseFloor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return NoiseFloor{}, err
	}
	var out NoiseFloor
	if err := json.Unmarshal(raw, &out); err != nil {
		return NoiseFloor{}, fmt.Errorf("%s: %w", path, err)
	}
	return out, nil
}

// NoiseMarkdown renders each metric's value per run and its MDE.
func NoiseMarkdown(n NoiseFloor) string {
	var b strings.Builder
	b.WriteString("# Voicebench noise floor\n\n")
	fmt.Fprintf(&b, "Target `%s` at `%s`, packs %s, k=%d, network profile `%s`, %d runs.\n\n",
		n.Target, n.GitCommit, strings.Join(n.Packs, ", "), n.K, orDash(n.NetworkProfile), len(n.Runs))
	b.WriteString("The MDE is the largest difference between any two runs. A change no bigger than it cannot be told from noise.\n\n")
	b.WriteString("| Metric | Runs | Mean | Std dev | MDE |\n| --- | --- | ---: | ---: | ---: |\n")
	for _, m := range n.Metrics {
		values := make([]string, len(m.Values))
		for i, v := range m.Values {
			values[i] = fmt.Sprintf("%.0f", v)
		}
		fmt.Fprintf(&b, "| %s | %s | %.0f | %.1f | %.0f %s |\n", m.Name, strings.Join(values, ", "), m.Mean, m.StdDev, m.MDE, m.Unit)
	}
	return b.String()
}

// mdeFlags names every metric whose change from base to run is bigger than its MDE.
func mdeFlags(n NoiseFloor, base, run runStats) []string {
	var flags []string
	for _, spread := range n.Metrics {
		for _, metric := range noiseMetrics {
			if metric.name != spread.Name {
				continue
			}
			was, okBase := metric.value(base)
			now, okRun := metric.value(run)
			delta := now - was
			if !okBase || !okRun || math.Abs(delta) <= spread.MDE {
				continue
			}
			verdict := "regression"
			if (delta > 0) == metric.higherBetter {
				verdict = "improvement"
			}
			flags = append(flags, fmt.Sprintf("%s %s (%+.0f %s, MDE %.0f)", spread.Name, verdict, delta, spread.Unit, spread.MDE))
		}
	}
	return flags
}

func packsOf(sum Summary) []string {
	packs := make([]string, 0, len(sum.Packs))
	for _, pack := range sum.Packs {
		packs = append(packs, pack.Pack)
	}
	sort.Strings(packs)
	return packs
}

func meanStdDev(values []float64) (float64, float64) {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	return mean, math.Sqrt(variance / float64(len(values)-1))
}
