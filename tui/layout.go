package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

const (
	// minTranscript is how many lines of conversation are worth keeping. Chrome is given
	// up rather than letting the transcript fall below it.
	minTranscript = 4
	// minContent is the narrowest the conversation is wrapped to, however narrow the
	// terminal gets.
	minContent = 20
	// minFramed is the width below which the rounded frames cost more than they say.
	minFramed = 28
	// plainest is the last thing tried: no banner, no frames, no help.
	plainest = 2
)

// relayout sizes the transcript to whatever the chrome around it leaves over. The chrome
// grows and shrinks with what there is to say — a banner, a scope, a caller's own header
// lines, a truncation notice — so it is measured rather than assumed, and on a terminal
// too short for all of it the banner and then the frames are given up first.
func (m *Model) relayout() {
	if m.width <= 0 || m.height <= 0 {
		return
	}
	start := 0
	if m.width < minFramed {
		start = plainest
	}
	for m.compact = start; ; m.compact++ {
		m.view.Width = max(minContent, m.width-2)
		m.input.SetWidth(m.contentWidth())
		height := m.height - lipgloss.Height(m.header()) - 1 - lipgloss.Height(m.footer())
		if height >= minTranscript || m.compact >= plainest {
			m.view.Height = max(1, height)
			return
		}
	}
}

// framed says whether the header and the composer are drawn in a rounded frame.
func (m *Model) framed() bool { return m.compact < plainest }

// contentWidth is where text is wrapped: inside the frame when there is one.
func (m *Model) contentWidth() int {
	if m.framed() {
		return max(minContent, m.width-4)
	}
	return max(minContent, m.width-2)
}

// frameWidth is what a framed block's style is given, so that the frame itself ends up
// exactly as wide as the terminal.
func (m *Model) frameWidth() int { return max(minContent+2, m.width-2) }

// header is the card above the conversation: who is being talked to, which conversation
// it is, and whatever else the application wants said.
func (m *Model) header() string {
	width := m.contentWidth()
	lines := []string{spread(m.styles.title.Render(m.branding.Title), m.styles.muted.Render(m.conversationLabel()), width)}
	if !m.framed() {
		return truncate(lines[0], m.width)
	}
	if m.branding.Subtitle != "" {
		lines = append(lines, m.styles.muted.Render(m.branding.Subtitle))
	}
	if m.scope != "" {
		lines = append(lines, m.styles.accent.Render(m.scope))
	}
	if m.options.Header != nil {
		for _, line := range m.options.Header(m.State()) {
			if line != "" {
				lines = append(lines, m.styles.muted.Render(line))
			}
		}
	}
	if m.truncated {
		lines = append(lines, m.styles.muted.Render("Earlier history is outside this session; /older reads it"))
	}
	for i, line := range lines {
		lines[i] = truncate(line, width)
	}
	card := m.styles.frame.Width(m.frameWidth()).Render(strings.Join(lines, "\n"))
	if m.compact == 0 && m.branding.Banner != "" {
		return m.styles.accent.Render(truncateBlock(m.branding.Banner, m.width)) + "\n" + card
	}
	return card
}

// conversationLabel names the conversation in the header, or says why it has no name.
func (m *Model) conversationLabel() string {
	switch {
	case m.conversationID != "":
		return m.conversationID
	case m.connecting:
		return "opening…"
	default:
		return "no conversation"
	}
}

// footer is the composer and the line under it: what is happening on the left, the keys
// that matter on the right.
func (m *Model) footer() string {
	composer := m.input.View()
	if m.framed() {
		composer = m.styles.frame.Width(m.frameWidth()).Render(composer)
	}
	lines := []string{composer}
	status := m.statusLine()
	width := m.contentWidth()
	help := m.styles.muted.Render(m.branding.Help)
	switch row, fits := spreadIfFits(status, help, width); {
	case m.compact >= plainest:
		lines = append(lines, truncate(status, width))
	case fits:
		lines = append(lines, row)
	default:
		lines = append(lines, truncate(status, width), truncate(help, width))
	}
	// Line the status up with the framed text above it rather than with the frame.
	for i := 1; i < len(lines); i++ {
		lines[i] = indent(lines[i], m.width-width-2)
	}
	return strings.Join(lines, "\n")
}

// statusLine says what the conversation is doing, and for how long it has been doing it.
func (m *Model) statusLine() string {
	status := m.status
	style := m.styles.muted
	switch {
	case m.statusFailed:
		style = m.styles.bad
	case m.busy || m.connecting:
		style = m.styles.accent
	}
	if m.busy && !m.submittedAt.IsZero() {
		status += " · " + seconds(time.Since(m.submittedAt)) + " total"
	}
	if m.busy || m.connecting {
		return m.styles.accent.Render(m.spinner()) + " " + style.Render(status)
	}
	return style.Render(status)
}

// seconds is how every duration in the interface is written.
func seconds(d time.Duration) string { return fmt.Sprintf("%.1fs", d.Seconds()) }

// spread puts two pieces on one line with the second against the right edge. The left is
// shortened first, because the right is the shorter and the more particular of the two.
func spread(left, right string, width int) string {
	if width <= 0 {
		return ""
	}
	rw := ansi.StringWidth(right)
	if rw >= width {
		return ansi.Truncate(right, width, "…")
	}
	if ansi.StringWidth(left)+1+rw > width {
		left = ansi.Truncate(left, width-rw-1, "…")
	}
	return left + strings.Repeat(" ", max(1, width-ansi.StringWidth(left)-rw)) + right
}

// spreadIfFits spreads two pieces only when both fit whole, with a gap between them.
func spreadIfFits(left, right string, width int) (string, bool) {
	if ansi.StringWidth(left)+2+ansi.StringWidth(right) > width {
		return "", false
	}
	return spread(left, right, width), true
}

// truncate keeps one line inside a width without breaking the escape sequences in it.
func truncate(line string, width int) string {
	if width <= 0 {
		return ""
	}
	return ansi.Truncate(line, width, "…")
}

// truncateBlock truncates every line of a block.
func truncateBlock(block string, width int) string {
	lines := strings.Split(block, "\n")
	for i, line := range lines {
		lines[i] = truncate(line, width)
	}
	return strings.Join(lines, "\n")
}

// indent shifts a block right, to line up with framed content beside it.
func indent(block string, by int) string {
	if by <= 0 {
		return block
	}
	pad := strings.Repeat(" ", by)
	lines := strings.Split(block, "\n")
	for i, line := range lines {
		lines[i] = pad + line
	}
	return strings.Join(lines, "\n")
}

// fit makes a block exactly height lines, so the interface occupies the terminal and
// nothing past the bottom of it.
func fit(block string, height int) string {
	if height <= 0 {
		return ""
	}
	lines := strings.Split(block, "\n")
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines[:height], "\n")
}
