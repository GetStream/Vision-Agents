package report

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

// Digest is a comparison cut down to what a chat message carries: the headline rows of
// CompareMarkdown, one cell per run, and what the runs were.
type Digest struct {
	Runs    []string
	Packs   []string
	K       int
	Commit  string
	Network string
	Started time.Time
	Rows    []DigestRow
	// Results is each run's trials counted by pack and scenario type, in the order of Runs.
	Results []Results
	// Scenarios are the scripts the calls followed, by id, so the report can set what the
	// caller said beside what the agent heard. Without them it shows what was heard alone.
	Scenarios map[string]scenario.Scenario
}

// DigestRow is one headline metric across the runs.
type DigestRow struct {
	Name string
	// Unit is "%" for a rate and "ms" for a time.
	Unit           string
	HigherIsBetter bool
	Cells          []DigestCell
	// Best is the index of the best cell, or -1 when no run has a value.
	Best int
}

// DigestCell is one run's value. Lo and Hi are its 95% interval when HasCI is set.
type DigestCell struct {
	Value, Lo, Hi float64
	HasCI         bool
	Samples       int
	Missing       bool
	// Behind marks a cell whose interval lies wholly on the wrong side of the best one,
	// so the gap is not noise.
	Behind bool
}

// digestRows are the CompareMarkdown rows a digest keeps, and what each is called there.
var digestRows = []struct {
	compare string
	name    string
	unit    string
	higher  bool
}{
	{"Pass rate", "Pass rate", "%", true},
	{"Reply time, non-tool P50 (ms)", "Reply time P50", "ms", false},
	{"Reply time, non-tool P95 (ms)", "Reply time P95", "ms", false},
	{"Reply time, tool turns P50 (ms)", "Tool turn P50", "ms", false},
	{"First response P50 (ms)", "First response P50", "ms", false},
}

// MergeRuns folds runs with the same label into one, so a target run once per pack reads
// as one system across every pack.
func MergeRuns(runs []LabeledRun) []LabeledRun {
	var merged []LabeledRun
	index := map[string]int{}
	for _, run := range runs {
		i, seen := index[run.Label]
		if !seen {
			index[run.Label] = len(merged)
			merged = append(merged, run)
			continue
		}
		sum := &merged[i].Summary
		sum.Packs = append(sum.Packs, run.Summary.Packs...)
		sum.Calls = append(sum.Calls, run.Summary.Calls...)
		if run.Summary.Started.Before(sum.Started) {
			sum.Started = run.Summary.Started
		}
	}
	return merged
}

// BuildDigest reduces the runs to their headline rows.
func BuildDigest(runs []LabeledRun) Digest {
	d := Digest{}
	packs := map[string]bool{}
	for i, run := range runs {
		d.Runs = append(d.Runs, run.Label)
		d.Results = append(d.Results, SummarizeResults(run.Summary.Calls))
		for _, pack := range run.Summary.Packs {
			packs[pack.Pack] = true
		}
		if i == 0 || run.Summary.Started.Before(d.Started) {
			d.Started = run.Summary.Started
		}
		d.K = max(d.K, run.Summary.K)
		if d.Commit == "" {
			d.Commit = run.Summary.Manifest.GitCommit
		}
		if d.Network == "" {
			d.Network = run.Summary.Manifest.NetworkProfile
		}
	}
	for pack := range packs {
		d.Packs = append(d.Packs, pack)
	}
	sort.Strings(d.Packs)

	stats := make([]runStats, len(runs))
	for i, run := range runs {
		stats[i] = summarizeRun(run.Summary)
	}
	rows := map[string]compareRow{}
	for _, row := range compareRows(runs) {
		rows[row.Name] = row
	}
	for _, want := range digestRows {
		row := rows[want.compare]
		out := DigestRow{Name: want.name, Unit: want.unit, HigherIsBetter: want.higher, Best: row.Best}
		for i, cell := range row.Cells {
			c := DigestCell{
				Value: cell.Value, Lo: cell.Lo, Hi: cell.Hi, HasCI: cell.HasCI,
				Behind:  row.Star[i],
				Missing: cell.Text == "—",
				Samples: digestSamples(want.compare, stats[i]),
			}
			if want.unit == "%" {
				c.Value, c.Lo, c.Hi = 100*c.Value, 100*c.Lo, 100*c.Hi
			}
			out.Cells = append(out.Cells, c)
		}
		d.Rows = append(d.Rows, out)
	}
	return d
}

func digestSamples(row string, st runStats) int {
	switch row {
	case "Pass rate":
		return st.Valid
	case "Reply time, tool turns P50 (ms)":
		return len(st.tool)
	case "First response P50 (ms)":
		return st.FirstResponseSamples
	}
	return len(st.nonTool)
}

// ReadCauses reads each call's artifacts and counts the failed calls of every run by why
// they failed, against the scripts in d.Scenarios.
func (d *Digest) ReadCauses(runs []LabeledRun) {
	for i, run := range runs {
		if i >= len(d.Results) {
			return
		}
		var details []CallDetail
		for _, call := range run.Summary.Calls {
			details = append(details, LoadCallDetail(call, d.scenario(call.ScenarioID)))
		}
		d.Results[i].Causes = CauseCounts(details)
	}
}

// scenario is the script a call followed, or nil when it is not known.
func (d Digest) scenario(id string) *scenario.Scenario {
	if sc, ok := d.Scenarios[id]; ok {
		return &sc
	}
	return nil
}

// SlackText is the message that goes with the digest image, in Slack's mrkdwn.
func (d Digest) SlackText(title string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "*%s*\n", title)
	b.WriteString(d.Meta() + "\n")
	for _, row := range d.Rows {
		cells := make([]string, 0, len(row.Cells))
		for i, cell := range row.Cells {
			text := d.Runs[i] + " " + cell.Text(row.Unit)
			if i == row.Best && row.Decided() {
				text = "*" + text + "*"
			}
			cells = append(cells, text)
		}
		fmt.Fprintf(&b, "• %s: %s%s\n", row.Name, strings.Join(cells, " · "), row.verdict(d.Runs))
	}
	runs := make([]string, 0, len(d.Results))
	for i, results := range d.Results {
		packs := make([]string, 0, len(results.ByPack))
		for _, pack := range results.ByPack {
			packs = append(packs, pack.Name+" "+pack.ScoreText())
		}
		runs = append(runs, fmt.Sprintf("%s %s · score %s (%s)",
			d.Runs[i], results.Overall.Text(), results.ScoreText(), strings.Join(packs, ", ")))
	}
	if len(runs) > 0 {
		fmt.Fprintf(&b, "• Passed: %s\n", strings.Join(runs, " · "))
	}
	var why []string
	for i, results := range d.Results {
		if len(results.Causes) > 0 {
			why = append(why, d.Runs[i]+" "+CausesText(results.Causes))
		}
	}
	if len(why) > 0 {
		fmt.Fprintf(&b, "• Why it failed: %s\n", strings.Join(why, " · "))
	}
	return b.String()
}

// Meta is one line saying what was run, where, and from which commit.
func (d Digest) Meta() string {
	meta := []string{strings.Join(d.Packs, ", "), fmt.Sprintf("k=%d", d.K)}
	if d.Network != "" {
		meta = append(meta, d.Network)
	}
	if d.Commit != "" {
		meta = append(meta, shortCommit(d.Commit))
	}
	if !d.Started.IsZero() {
		meta = append(meta, d.Started.UTC().Format("Jan 2 15:04 UTC"))
	}
	return strings.Join(meta, " · ")
}

// Text is the cell as it reads in a message.
func (c DigestCell) Text(unit string) string {
	if c.Missing {
		return "—"
	}
	if unit == "%" {
		return fmt.Sprintf("%.0f%%", c.Value)
	}
	return seconds(c.Value)
}

// Decided says whether the best cell clears every other one, so the row has a winner.
func (r DigestRow) Decided() bool {
	if r.Best < 0 || len(r.Cells) < 2 {
		return false
	}
	for i, cell := range r.Cells {
		if i != r.Best && !cell.Missing && !cell.Behind {
			return false
		}
	}
	return true
}

func (r DigestRow) verdict(runs []string) string {
	if r.Best < 0 || len(r.Cells) < 2 {
		return ""
	}
	if r.Decided() {
		return " → " + runs[r.Best]
	}
	for _, cell := range r.Cells {
		if cell.HasCI {
			return " → within noise"
		}
	}
	return ""
}

func seconds(ms float64) string {
	if ms < 1000 {
		return fmt.Sprintf("%.0f ms", ms)
	}
	return fmt.Sprintf("%.2f s", ms/1000)
}

func shortCommit(commit string) string {
	if len(commit) > 7 {
		return commit[:7]
	}
	return commit
}
