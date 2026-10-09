package report

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"html/template"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// HTML is the full report that goes with a digest: the card, every metric with its
// interval, and every call with what failed it. It is one file with nothing to fetch, so
// it opens from a Slack attachment as it is.
func (d Digest) HTML(title string, card []byte, runs []LabeledRun) (string, error) {
	page := digestPage{
		Title: title,
		Meta:  d.Meta(),
		Card:  template.URL("data:image/png;base64," + base64.StdEncoding.EncodeToString(card)),
		Runs:  d.Runs,
	}
	for _, row := range d.Rows {
		out := digestPageRow{Name: row.Name, Verdict: strings.TrimPrefix(row.verdict(d.Runs), " → ")}
		for i, cell := range row.Cells {
			text := cell.Text(row.Unit)
			if cell.HasCI && !cell.Missing {
				text += " (" + interval(cell, row.Unit) + ")"
			}
			if cell.Samples > 0 {
				text += fmt.Sprintf(", n=%d", cell.Samples)
			}
			out.Cells = append(out.Cells, digestPageCell{Text: text, Best: i == row.Best && row.Decided()})
		}
		page.Rows = append(page.Rows, out)
	}
	for i, run := range runs {
		calls := append([]CallResult(nil), run.Summary.Calls...)
		sort.SliceStable(calls, func(i, j int) bool {
			if calls[i].Pack != calls[j].Pack {
				return calls[i].Pack < calls[j].Pack
			}
			return calls[i].ScenarioID < calls[j].ScenarioID
		})
		section := digestPageRun{Label: run.Label, Pipeline: pipelineOf(run.Summary.Manifest)}
		if i < len(d.Results) {
			section.Results = d.Results[i]
		}
		for _, call := range calls {
			section.Calls = append(section.Calls, digestPageCall{
				Scenario: call.ScenarioID,
				Trial:    call.Trial,
				Outcome:  callOutcome(call),
				Failed:   strings.Join(CallFailures(call), "; "),
				Reply:    callReplyP50(call),
				First:    callFirstResponse(call),
				Tools:    call.Metrics.ToolCount,
			})
		}
		page.Sections = append(page.Sections, section)
	}
	var out bytes.Buffer
	if err := digestTemplate.Execute(&out, page); err != nil {
		return "", err
	}
	return out.String(), nil
}

func interval(cell DigestCell, unit string) string {
	if unit == "%" {
		return fmt.Sprintf("%.0f–%.0f%%", cell.Lo, cell.Hi)
	}
	return seconds(cell.Lo) + "–" + seconds(cell.Hi)
}

// CallFailures is what failed a call, most telling first: the deterministic gates, then
// the judge's policy calls, then why a trial was not counted at all.
func CallFailures(call CallResult) []string {
	m := call.Metrics
	var out []string
	out = append(out, m.EndStateFail...)
	out = append(out, m.ExpectedToolFail...)
	out = append(out, m.PolicyFail...)
	out = append(out, m.SayDoFail...)
	out = append(out, call.InvalidReason...)
	if call.Error != "" {
		out = append(out, call.Error)
	}
	if len(out) > 4 {
		out = append(out[:4], fmt.Sprintf("and %d more", len(out)-4))
	}
	return out
}

func callReplyP50(call CallResult) string {
	var gaps []int
	for _, timing := range call.Metrics.V2V {
		if timing.V2VMS >= 0 && !timing.Tool {
			gaps = append(gaps, timing.V2VMS)
		}
	}
	if len(gaps) == 0 {
		return "—"
	}
	sort.Ints(gaps)
	return seconds(float64(score.Percentile(gaps, 50)))
}

func callFirstResponse(call CallResult) string {
	first := call.Metrics.FirstResponse
	if first == nil || first.V2VMS < 0 {
		return "—"
	}
	return seconds(float64(first.V2VMS))
}

type digestPage struct {
	Title    string
	Meta     string
	Card     template.URL
	Runs     []string
	Rows     []digestPageRow
	Sections []digestPageRun
}

type digestPageRow struct {
	Name    string
	Verdict string
	Cells   []digestPageCell
}

type digestPageCell struct {
	Text string
	Best bool
}

type digestPageRun struct {
	Label    string
	Pipeline string
	Results  Results
	Calls    []digestPageCall
}

type digestPageCall struct {
	Scenario string
	Trial    int
	Outcome  string
	Failed   string
	Reply    string
	First    string
	Tools    int
}

var digestTemplate = template.Must(template.New("digest").Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}}</title>
<style>
  :root { --bg: #f4f5f7; --surface: #fdfdfc; --ink: #121419; --muted: #5b616c; --rule: #dde0e5;
          --pass: #1f8a5b; --pass-bg: #ddf3e8; --fail: #c23b3b; --fail-bg: #fbe5e5; --invalid-bg: #eceef1; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #111316; --surface: #191b1f; --ink: #f1f2f4; --muted: #a3a9b3; --rule: #2b2f35;
            --pass: #4cc48d; --pass-bg: #163024; --fail: #ec6f6f; --fail-bg: #3b1d1d; --invalid-bg: #25282d; }
  }
  body { background: var(--bg); color: var(--ink); margin: 0;
         font: 15px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; display: grid; gap: 28px; }
  h1 { font-size: 28px; margin: 0; } h2 { font-size: 20px; margin: 0 0 10px; }
  .meta { color: var(--muted); }
  img { max-width: 100%; border-radius: 10px; border: 1px solid var(--rule); }
  .panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 4px 16px; overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 10px; border-top: 1px solid var(--rule); vertical-align: top; }
  th { border-top: 0; color: var(--muted); font-size: 12px; letter-spacing: .05em; text-transform: uppercase; }
  .best { font-weight: 700; }
  .verdict { color: var(--muted); font-size: 13px; }
  .pill { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 6px; }
  .pass { background: var(--pass-bg); color: var(--pass); }
  .fail { background: var(--fail-bg); color: var(--fail); }
  .invalid { background: var(--invalid-bg); color: var(--muted); }
  .why { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<main>
  <header><h1>{{.Title}}</h1><div class="meta">{{.Meta}}</div></header>
  <img src="{{.Card}}" alt="Scorecard">
  <section>
    <h2>Headline metrics</h2>
    <div class="panel"><table>
      <tr><th>Metric</th>{{range .Runs}}<th>{{.}}</th>{{end}}</tr>
      {{range .Rows}}<tr><td>{{.Name}}{{if .Verdict}}<div class="verdict">{{.Verdict}}</div>{{end}}</td>
        {{range .Cells}}<td{{if .Best}} class="best"{{end}}>{{.Text}}</td>{{end}}</tr>
      {{end}}
    </table></div>
  </section>
  {{range .Sections}}
  <section>
    <h2>{{.Label}}</h2>
    <div class="meta">{{.Pipeline}}</div>
    {{with .Results}}<div class="panel"><table>
      <tr><th>Group</th><th>Passed</th><th>Invalid</th><th>Score</th></tr>
      <tr><td><strong>All</strong></td><td><strong>{{.Overall.Passed}}/{{.Overall.Valid}}</strong></td><td>{{.Overall.Invalid}}</td><td><strong>{{.ScoreText}}</strong></td></tr>
      <tr><th colspan="4">By pack</th></tr>
      {{range .ByPack}}<tr><td>{{.Name}}</td><td>{{.Passed}}/{{.Valid}}</td><td>{{.Invalid}}</td><td>{{.ScoreText}}</td></tr>{{end}}
      <tr><th colspan="4">By scenario</th></tr>
      {{range .ByKind}}<tr><td>{{.Name}}</td><td>{{.Passed}}/{{.Valid}}</td><td>{{.Invalid}}</td><td>{{.ScoreText}}</td></tr>{{end}}
    </table><div class="why">Score is the pass rate on 0-100 over trials that produced a verdict. The overall score is the mean of the packs, so each pack counts the same.</div></div>{{end}}
    <div class="panel"><table>
      <tr><th>Scenario</th><th>Outcome</th><th>Reply P50</th><th>First response</th><th>Tools</th></tr>
      {{range .Calls}}<tr>
        <td>{{.Scenario}}{{if gt .Trial 1}} #{{.Trial}}{{end}}</td>
        <td><span class="pill {{.Outcome}}">{{.Outcome}}</span>{{if .Failed}}<div class="why">{{.Failed}}</div>{{end}}</td>
        <td>{{.Reply}}</td><td>{{.First}}</td><td>{{.Tools}}</td>
      </tr>{{end}}
    </table></div>
  </section>
  {{end}}
</main>
</body>
</html>
`))
