package tui

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/charmbracelet/glamour"
	glamourstyles "github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"golang.org/x/term"
)

// spinnerFrames is what work still going looks like.
var spinnerFrames = [...]string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// markdownCacheSize is how many rendered answers are kept. An answer is re-rendered on
// every frame otherwise, and re-rendered for real whenever the terminal is resized.
const markdownCacheSize = 200

// durationColumn is wide enough for a tool that ran for hours.
const durationColumn = 7

// bodyIndent is how far what was said sits from the bar marking whose turn it is. It is
// the margin a rendered answer already gives itself, matched by everything beside it.
const bodyIndent = 2

func (m *Model) spinner() string { return spinnerFrames[m.frame%len(spinnerFrames)] }

// transcript is the conversation as it stands, newest last.
func (m *Model) transcript() string {
	width := max(minContent, m.view.Width)
	blocks := make([]string, 0, len(m.messages)+1)
	for _, message := range m.messages {
		blocks = append(blocks, m.turn(message, width))
	}
	if m.notice != "" {
		blocks = append(blocks, m.notice)
	}
	return strings.Join(blocks, "\n\n")
}

// turn draws one side of the conversation behind a coloured bar, so a long answer with
// tool activity in it still reads as one thing somebody said. What was said is set in
// from the bar by as much as a rendered answer sets itself in, so that prose, tools and
// the state under them all share one left edge.
func (m *Model) turn(message stream.ConversationMessage, width int) string {
	body := max(minContent, width-2-bodyIndent)
	if message.Role == "user" {
		// A question is wrapped on its words as plain text, and the escape sequences are
		// taken out of it: what somebody typed, or what a transcript recorded, is not
		// styling. A word longer than the terminal is still broken, so that a pasted URL
		// cannot push a question off the side.
		question := ansi.Wrap(ansi.Strip(message.Text), body, "")
		return gutter(m.styles.you, m.branding.You, indent(question, bodyIndent), width-2)
	}
	var lines []string
	if message.Text != "" {
		lines = append(lines, m.markdown(message.Text, body))
	}
	below := m.toolRows(message.Tools, body)
	if message.State != "" || message.FinishedAt != nil {
		below = append(below, m.stateLine(message))
	}
	if len(below) > 0 {
		lines = append(lines, indent(strings.Join(below, "\n"), bodyIndent))
	}
	return gutter(m.styles.accent, m.branding.Agent, strings.Join(lines, "\n"), width-2)
}

// gutter marks a turn with a bar down its left edge, labelled with whose turn it is.
func gutter(style lipgloss.Style, name, body string, width int) string {
	bar := style.Render("▎") + " "
	lines := []string{bar + style.Render(name)}
	if body != "" {
		for _, line := range strings.Split(body, "\n") {
			lines = append(lines, bar+truncate(line, width))
		}
	}
	return strings.Join(lines, "\n")
}

// toolRows lists what the agent did to answer: a mark for how it went, what it was, how
// long it took, and whatever it has to say for itself. The titles share a column so the
// durations line up under each other.
func (m *Model) toolRows(tools []stream.ToolActivity, width int) []string {
	if len(tools) == 0 {
		return nil
	}
	titles := 0
	for _, tool := range tools {
		titles = max(titles, ansi.StringWidth(tool.Title))
	}
	titles = min(titles, max(8, width/3))
	rows := make([]string, 0, len(tools))
	for _, tool := range tools {
		mark, style := m.toolMark(tool.Status)
		title := ansi.Truncate(tool.Title, titles, "…")
		title += strings.Repeat(" ", max(0, titles-ansi.StringWidth(title)))
		duration := seconds(elapsed(tool.StartedAt, tool.FinishedAt))
		duration = strings.Repeat(" ", max(0, durationColumn-len(duration))) + duration
		note := tool.Summary
		if note == "" {
			note = tool.Phase
		}
		row := style.Render(mark) + " " + m.styles.text.Render(title) + "  " + m.styles.muted.Render(duration)
		if note != "" {
			row += "  " + m.styles.muted.Render(note)
		}
		rows = append(rows, truncate(row, width))
	}
	return rows
}

// toolMark is how a tool's outcome is drawn: still going is the spinner, so a row that
// is doing something looks like it.
func (m *Model) toolMark(status string) (string, lipgloss.Style) {
	switch status {
	case "completed":
		return "✓", m.styles.good
	case "failed":
		return "✗", m.styles.bad
	case "cancelled":
		return "–", m.styles.muted
	}
	return m.spinner(), m.styles.accent
}

// stateLine closes a turn with where the answer got to, how long that has taken, and
// whether it survived being written down.
func (m *Model) stateLine(message stream.ConversationMessage) string {
	// An unfinished answer times the state it is in rather than the whole turn, which is
	// what the header's total is for.
	start := message.StartedAt
	if message.FinishedAt == nil && !message.StateStartedAt.IsZero() {
		start = message.StateStartedAt
	}
	line := label(message.State) + " · " + seconds(elapsed(start, message.FinishedAt))
	if message.FinishedAt != nil {
		if message.Saved {
			line += " · saved"
		} else {
			line += " · saving…"
		}
	}
	if message.Error != "" {
		line += " · " + message.Error
	}
	if message.State == "failed" {
		return m.styles.bad.Render(line)
	}
	return m.styles.muted.Render(line)
}

// label says a state the way a person would. An unrecognised state is shown as it came,
// because a backend that learns a new one should not go silent here.
func label(state string) string {
	switch state {
	case "thinking":
		return "Thinking…"
	case "queued":
		return "Waiting for a worker"
	case "tools":
		return "Running tools"
	case "writing":
		return "Writing the answer"
	case "completed":
		return "Completed"
	case "failed":
		return "Failed"
	case "cancelled":
		return "Cancelled"
	case "interrupted":
		return "Interrupted"
	}
	return state
}

// settled says whether a state is one an answer stays in. A turn that has ended without
// saying when is still a turn that has ended, so it stops being waited on.
func settled(state string) bool {
	switch state {
	case "completed", "failed", "cancelled", "interrupted":
		return true
	}
	return false
}

// elapsed is how long something has been going for, or how long it went for.
func elapsed(start time.Time, end *time.Time) time.Duration {
	if start.IsZero() {
		return 0
	}
	if end != nil {
		return end.Sub(start)
	}
	return time.Since(start)
}

// markdownStyle is the style answers are rendered in, decided once for the process.
//
// It follows the terminal's background, but it must not be glamour's own "auto" style to
// do it. Auto asks the terminal directly, writing an OSC 11 query to the tty and reading
// the reply back off it, and every renderer built with it asks again. By the time an
// answer is being rendered, Bubble Tea holds that tty in raw mode and is reading it: the
// reply reaches Bubble Tea first and is parsed as though it had been typed, so the
// escape sequence lands in the composer, while the query waits for a reply that has
// already been taken and stalls the frame until it times out.
//
// Lipgloss answers the same question without asking again, because Bubble Tea's own
// package init asks it once before any program starts and caches it. That is early
// enough to be safe, since nothing owns the terminal yet.
var markdownStyle = sync.OnceValue(func() string {
	// Not a terminal, so there is no background to match and nothing to ask anyway.
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		return glamourstyles.NoTTYStyle
	}
	if lipgloss.HasDarkBackground() {
		return glamourstyles.DarkStyle
	}
	return glamourstyles.LightStyle
})

// renderer returns the renderer for one wrap width, building each at most once.
//
// Glamour compiles a style into a chain of markdown extensions, which is not work worth
// repeating for every token of an answer still being written. Rendering does not mutate
// the renderer, so one serves every answer at that width.
func (m *Model) renderer(width int) *glamour.TermRenderer {
	if existing, ok := m.renderers[width]; ok {
		return existing
	}
	built, err := glamour.NewTermRenderer(glamour.WithStandardStyle(markdownStyle()),
		glamour.WithWordWrap(max(minContent, width-2)))
	if err != nil {
		return nil
	}
	if m.renderers == nil {
		m.renderers = map[int]*glamour.TermRenderer{}
	}
	m.renderers[width] = built
	return built
}

// markdown renders an answer, falling back to what was written if it cannot be rendered:
// an answer nobody can read is worse than an unstyled one.
func (m *Model) markdown(text string, width int) string {
	key := strconv.Itoa(width) + "\x00" + text
	if rendered, ok := m.cache[key]; ok {
		return rendered
	}
	rendered := text
	// The wrap leaves room for the margin the style puts around a document.
	if renderer := m.renderer(width); renderer != nil {
		if out, err := renderer.Render(text); err == nil {
			rendered = strings.Trim(out, "\n")
		}
	}
	if len(m.cache) >= markdownCacheSize {
		m.cache = make(map[string]string, markdownCacheSize)
	}
	m.cache[key] = rendered
	return rendered
}
