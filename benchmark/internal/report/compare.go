package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// LabeledRun is one summary.json with a display name.
type LabeledRun struct {
	Label   string
	Dir     string
	Summary Summary
}

// CompareConfig is a head-to-head or baseline diff.
type CompareConfig struct {
	Runs     []LabeledRun
	Baseline int
	MDEV2VMS int
}

// LoadSummary reads summary.json from a run directory.
func LoadSummary(dir string) (Summary, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "summary.json"))
	if err != nil {
		return Summary{}, err
	}
	var sum Summary
	if err := json.Unmarshal(raw, &sum); err != nil {
		return Summary{}, err
	}
	return sum, nil
}

type compareCell struct {
	Text          string
	Value         float64
	Lo, Hi        float64
	HasCI         bool
	LowerIsBetter bool
}

type compareRow struct {
	Name  string
	Cells []compareCell
	Best  int
	Star  []bool
}

// CompareMarkdown renders a per-metric table across labeled runs.
func CompareMarkdown(cfg CompareConfig) string {
	if len(cfg.Runs) == 0 {
		return ""
	}
	rows := compareRows(cfg.Runs)
	var b strings.Builder
	b.WriteString("# Voicebench compare\n\n")
	b.WriteString("Headline rows compare products as configured, not matched models. Intervals that do not overlap the best run are marked *.\n\n")
	b.WriteString("| Metric")
	for _, run := range cfg.Runs {
		fmt.Fprintf(&b, " | %s", run.Label)
	}
	b.WriteString(" |\n| ---")
	for range cfg.Runs {
		b.WriteString(" | ---")
	}
	b.WriteString(" |\n")
	for _, row := range rows {
		fmt.Fprintf(&b, "| %s", row.Name)
		for i, cell := range row.Cells {
			mark := ""
			if i == row.Best && len(cfg.Runs) > 1 {
				mark = " **"
			}
			if i < len(row.Star) && row.Star[i] {
				mark += "*"
			}
			fmt.Fprintf(&b, " | %s%s", cell.Text, mark)
		}
		b.WriteString(" |\n")
	}
	if cfg.Baseline >= 0 && cfg.Baseline < len(cfg.Runs) && len(cfg.Runs) > 1 {
		b.WriteString("\n## Versus baseline\n\n")
		b.WriteString(baselineSection(cfg))
	}
	b.WriteString("\nEach run's manifest target, model, and network profile should match before a gap is treated as a product result.\n")
	return b.String()
}

func compareRows(runs []LabeledRun) []compareRow {
	stats := make([]runStats, len(runs))
	for i, run := range runs {
		stats[i] = summarizeRun(run.Summary)
	}
	return []compareRow{
		compareRateRow("Pass rate", stats, func(s runStats) (int, int) { return s.Passed, s.Valid }),
		comparePointRow("V2V P50 (ms)", stats, func(s runStats) float64 { return float64(s.V2VP50) }, true),
		comparePointRow("V2V P95 (ms)", stats, func(s runStats) float64 { return float64(s.V2VP95) }, true),
		comparePointRow("Non-tool P50 (ms)", stats, func(s runStats) float64 { return float64(s.NonToolP50) }, true),
		comparePointRow("Caller turns P50", stats, func(s runStats) float64 { return float64(s.CallerTurnsP50) }, true),
		comparePointRow("Agent turns P50", stats, func(s runStats) float64 { return float64(s.AgentTurnsP50) }, true),
	}
}

type runStats struct {
	Passed, Valid, Invalid int
	V2VP50, V2VP95         int
	NonToolP50             int
	CallerTurnsP50         int
	AgentTurnsP50          int
}

func summarizeRun(sum Summary) runStats {
	var out runStats
	var v2v []int
	var nonTool []int
	var callerTurns []int
	var agentTurns []int
	for _, call := range sum.Calls {
		switch callOutcome(call) {
		case OutcomeInvalid:
			out.Invalid++
			continue
		case OutcomePass:
			out.Passed++
			out.Valid++
		default:
			out.Valid++
		}
		for _, timing := range call.Metrics.V2V {
			if timing.V2VMS < 0 {
				continue
			}
			v2v = append(v2v, timing.V2VMS)
			if !timing.Tool {
				nonTool = append(nonTool, timing.V2VMS)
			}
		}
		if call.Metrics.CallerTurns > 0 {
			callerTurns = append(callerTurns, call.Metrics.CallerTurns)
		}
		if call.Metrics.AgentTurns > 0 {
			agentTurns = append(agentTurns, call.Metrics.AgentTurns)
		}
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
	if len(callerTurns) > 0 {
		sort.Ints(callerTurns)
		out.CallerTurnsP50 = score.Percentile(callerTurns, 50)
	}
	if len(agentTurns) > 0 {
		sort.Ints(agentTurns)
		out.AgentTurnsP50 = score.Percentile(agentTurns, 50)
	}
	if out.Valid == 0 && len(sum.Packs) > 0 {
		for _, pack := range sum.Packs {
			out.V2VP50 = pack.V2VP50
			out.V2VP95 = pack.V2VP95
			out.NonToolP50 = pack.NonToolP50
			out.CallerTurnsP50 = pack.CallerTurnsP50
			out.AgentTurnsP50 = pack.AgentTurnsP50
			break
		}
	}
	return out
}

func compareRateRow(name string, stats []runStats, pick func(runStats) (int, int)) compareRow {
	row := compareRow{Name: name, Best: 0, Star: make([]bool, len(stats))}
	best := -1.0
	for _, st := range stats {
		passed, valid := pick(st)
		p, lo, hi := wilson95(passed, valid)
		row.Cells = append(row.Cells, compareCell{
			Text:  fmt.Sprintf("%.1f%% (%.1f–%.1f, n=%d)", 100*p, 100*lo, 100*hi, valid),
			Value: p, Lo: lo, Hi: hi, HasCI: true,
		})
		if p > best {
			best = p
			row.Best = len(row.Cells) - 1
		}
	}
	for i, cell := range row.Cells {
		if i != row.Best && cell.HasCI && cell.Hi < row.Cells[row.Best].Lo {
			row.Star[i] = true
		}
	}
	return row
}

func comparePointRow(name string, stats []runStats, pick func(runStats) float64, lowerBetter bool) compareRow {
	row := compareRow{Name: name, Best: 0, Star: make([]bool, len(stats))}
	for i, st := range stats {
		v := pick(st)
		row.Cells = append(row.Cells, compareCell{
			Text:          fmt.Sprintf("%.0f", v),
			Value:         v,
			LowerIsBetter: lowerBetter,
		})
		if i == 0 {
			continue
		}
		if lowerBetter && v < row.Cells[row.Best].Value {
			row.Best = i
		}
		if !lowerBetter && v > row.Cells[row.Best].Value {
			row.Best = i
		}
	}
	return row
}

func baselineSection(cfg CompareConfig) string {
	base := summarizeRun(cfg.Runs[cfg.Baseline].Summary)
	var b strings.Builder
	fmt.Fprintf(&b, "Baseline: `%s`.\n\n", cfg.Runs[cfg.Baseline].Label)
	if cfg.MDEV2VMS <= 0 {
		b.WriteString("No V2V MDE is configured, so latency deltas are reported without a regression gate.\n\n")
	} else {
		fmt.Fprintf(&b, "V2V P50 changes larger than %d ms are flagged.\n\n", cfg.MDEV2VMS)
	}
	b.WriteString("| Run | Pass rate delta | V2V P50 delta | Flag |\n| --- | ---: | ---: | --- |\n")
	for i, run := range cfg.Runs {
		if i == cfg.Baseline {
			continue
		}
		st := summarizeRun(run.Summary)
		baseRate, _, _ := wilson95(base.Passed, base.Valid)
		rate, _, _ := wilson95(st.Passed, st.Valid)
		v2v := st.V2VP50 - base.V2VP50
		flag := ""
		if cfg.MDEV2VMS > 0 && absInt(v2v) >= cfg.MDEV2VMS {
			if v2v > 0 {
				flag = "regression"
			} else {
				flag = "improvement"
			}
		}
		fmt.Fprintf(&b, "| %s | %+.1f pp | %+d ms | %s |\n", run.Label, 100*(rate-baseRate), v2v, flag)
	}
	return b.String()
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
