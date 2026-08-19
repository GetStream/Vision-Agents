package report

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
)

var (
	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15"))
	metaStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	okStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	missStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	warnStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	skipStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	headStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15"))
)

// Table renders a gold-vs-ours scorecard for the terminal.
func Table(s Summary) string {
	rows := Scorecard(s)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		StyleFunc(func(row, col int) lipgloss.Style {
			if row == table.HeaderRow {
				return headStyle
			}
			if row < 0 || row >= len(rows) {
				return lipgloss.NewStyle()
			}
			return verdictStyle(rows[row].Verdict)
		}).
		Headers("BENCHMARK", "GOLD", "OURS", "GAP")
	for _, r := range rows {
		t.Row(r.Name, r.Gold, r.Ours, r.Gap)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s  %s\n",
		titleStyle.Render("Voicebench"),
		metaStyle.Render(fmt.Sprintf("%s  k=%d  %s", s.System, s.K, s.RunID)))
	b.WriteString(t.Render())
	b.WriteByte('\n')
	return b.String()
}

func verdictStyle(v Verdict) lipgloss.Style {
	switch v {
	case VerdictOK:
		return okStyle
	case VerdictMiss:
		return missStyle
	case VerdictWarn:
		return warnStyle
	default:
		return skipStyle
	}
}

// FprintTable writes the scorecard to w (stdout when w is nil).
func FprintTable(w io.Writer, s Summary) {
	if w == nil {
		w = os.Stdout
	}
	fmt.Fprint(w, Table(s))
}
