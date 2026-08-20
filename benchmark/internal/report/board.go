package report

import (
	"fmt"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// Verdict is how a scorecard row compares to the Voicebench target.
type Verdict int

const (
	VerdictOK Verdict = iota
	VerdictMiss
	VerdictWarn
	VerdictSkip
)

// Row is one target-vs-ours line on the scorecard.
type Row struct {
	Name    string
	Target  string
	Ours    string
	Gap     string
	Verdict Verdict
}

type scenarioBench struct {
	Name   string
	Needle string
	Target string
}

var scenarioBenches = []scenarioBench{
	{Name: "Task completion", Needle: ".golden", Target: "pass"},
	{Name: "Coherence (2 min)", Needle: "coherence", Target: "pass"},
	{Name: "Noise", Needle: "noise", Target: "pass"},
	{Name: "Ignore other talkers", Needle: "selectivity", Target: "hold"},
	{Name: "Tool filler", Needle: "tool_filler", Target: "pass"},
	{Name: "Barge-in / interrupt", Needle: "interrupt", Target: "pass"},
	{Name: "Entity fidelity", Needle: "entity_dense", Target: "pass"},
	{Name: "Policy", Needle: "adversarial", Target: "pass"},
}

// Scorecard compares each checklist benchmark and latency metric to the Voicebench target.
func Scorecard(s Summary) []Row {
	var rows []Row
	for _, b := range scenarioBenches {
		rows = append(rows, scenarioRow(s, b))
	}
	rows = append(rows, latencyRows(s)...)
	return rows
}

func scenarioRow(s Summary, b scenarioBench) Row {
	calls := matchCalls(s.Calls, b.Needle)
	if len(calls) == 0 {
		return Row{Name: b.Name, Target: b.Target, Ours: "—", Gap: "not run", Verdict: VerdictSkip}
	}
	passed := 0
	for _, c := range calls {
		if c.Passed {
			passed++
		}
	}
	ours := fmt.Sprintf("%d/%d pass", passed, len(calls))
	if score.PassAtK(passedFlags(calls)) {
		gap := "ok"
		if !score.PassHatK(passedFlags(calls)) {
			gap = "flaky"
			return Row{Name: b.Name, Target: b.Target, Ours: ours, Gap: gap, Verdict: VerdictWarn}
		}
		return Row{Name: b.Name, Target: b.Target, Ours: ours, Gap: gap, Verdict: VerdictOK}
	}
	return Row{Name: b.Name, Target: b.Target, Ours: ours, Gap: "miss", Verdict: VerdictMiss}
}

func latencyRows(s Summary) []Row {
	var nonTool, v2v, barge []int
	spikes := 0
	cutoffs := 0
	n := 0
	for _, c := range s.Calls {
		if c.Error != "" && c.Metrics.V2VP50 == 0 && c.Metrics.NonToolP50 == 0 {
			continue
		}
		n++
		if c.Metrics.NonToolP50 > 0 {
			nonTool = append(nonTool, c.Metrics.NonToolP50)
		}
		if c.Metrics.V2VP50 > 0 {
			v2v = append(v2v, c.Metrics.V2VP50)
		}
		if strings.Contains(c.ScenarioID, "interrupt") && c.Metrics.BargeInStopMS >= 0 {
			barge = append(barge, c.Metrics.BargeInStopMS)
		}
		spikes += c.Metrics.SpikeCount
		cutoffs += c.Metrics.FalseCutoff
	}
	rows := []Row{
		msBandRow("Reply gap (non-tool P50)", nonTool, score.HumanBandMinMS, score.HumanBandMaxMS),
		msBandRow("Voice-to-voice P50", v2v, score.HumanBandMinMS, score.HumanBandMaxMS),
		msCapRow("Barge-in stop", barge, score.MaxBargeInStopMS),
	}
	if n == 0 {
		return rows
	}
	rows = append(rows,
		countRow("Latency spikes", spikes, 0),
		rateRow("False cutoffs / call", float64(cutoffs)/float64(n), 0),
	)
	return rows
}

func msBandRow(name string, samples []int, lo, hi int) Row {
	target := fmt.Sprintf("%d–%d ms", lo, hi)
	if len(samples) == 0 {
		return Row{Name: name, Target: target, Ours: "—", Gap: "not run", Verdict: VerdictSkip}
	}
	ours := median(samples)
	row := Row{Name: name, Target: target, Ours: fmt.Sprintf("%d ms", ours)}
	if ours >= lo && ours <= hi {
		row.Gap = "ok"
		row.Verdict = VerdictOK
		return row
	}
	if ours > hi {
		row.Gap = fmt.Sprintf("+%d ms", ours-hi)
		row.Verdict = VerdictMiss
		return row
	}
	row.Gap = fmt.Sprintf("%d ms under", lo-ours)
	row.Verdict = VerdictWarn
	return row
}

func msCapRow(name string, samples []int, capMS int) Row {
	target := fmt.Sprintf("≤ %d ms", capMS)
	if len(samples) == 0 {
		return Row{Name: name, Target: target, Ours: "—", Gap: "not run", Verdict: VerdictSkip}
	}
	ours := maxInt(samples)
	row := Row{Name: name, Target: target, Ours: fmt.Sprintf("%d ms", ours)}
	if ours <= capMS {
		row.Gap = "ok"
		row.Verdict = VerdictOK
		return row
	}
	row.Gap = fmt.Sprintf("+%d ms", ours-capMS)
	row.Verdict = VerdictMiss
	return row
}

func countRow(name string, ours, target int) Row {
	row := Row{Name: name, Target: fmt.Sprintf("%d", target), Ours: fmt.Sprintf("%d", ours)}
	if ours <= target {
		row.Gap = "ok"
		row.Verdict = VerdictOK
		return row
	}
	row.Gap = fmt.Sprintf("+%d", ours-target)
	row.Verdict = VerdictMiss
	return row
}

func rateRow(name string, ours, target float64) Row {
	row := Row{Name: name, Target: fmt.Sprintf("%.2f", target), Ours: fmt.Sprintf("%.2f", ours)}
	if ours <= target {
		row.Gap = "ok"
		row.Verdict = VerdictOK
		return row
	}
	row.Gap = fmt.Sprintf("+%.2f", ours-target)
	row.Verdict = VerdictMiss
	return row
}

func matchCalls(calls []CallResult, needle string) []CallResult {
	var out []CallResult
	for _, c := range calls {
		if needle[0] == '.' {
			if strings.HasSuffix(c.ScenarioID, needle) {
				out = append(out, c)
			}
			continue
		}
		if strings.Contains(c.ScenarioID, needle) {
			out = append(out, c)
		}
	}
	return out
}

func passedFlags(calls []CallResult) []bool {
	out := make([]bool, len(calls))
	for i, c := range calls {
		out[i] = c.Passed
	}
	return out
}

func median(vals []int) int {
	sort.Ints(vals)
	return vals[len(vals)/2]
}

func maxInt(vals []int) int {
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}
