package tui

import (
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/lipgloss"
)

// Theme is the palette the terminal UI draws with. Every colour is adaptive, so one
// theme reads on a light terminal and a dark one. A zero colour takes the default,
// which means a theme can change one thing and leave the rest alone.
type Theme struct {
	// Accent is the agent's colour: the frames, the title and whatever is happening now.
	Accent lipgloss.TerminalColor
	// You is the colour of the person's side of the conversation.
	You lipgloss.TerminalColor
	// Text is ordinary copy and Muted is everything said quietly: timings, phases,
	// summaries and the help line.
	Text, Muted lipgloss.TerminalColor
	// Border draws the header and composer frames.
	Border lipgloss.TerminalColor
	// Good and Bad mark work that finished and work that did not.
	Good, Bad lipgloss.TerminalColor
}

// DefaultTheme is the palette of the Vision-Agents dashboard, so a conversation looks
// related whether it is read in a terminal or in a browser.
func DefaultTheme() Theme {
	return Theme{
		Accent: lipgloss.AdaptiveColor{Light: "#0284c7", Dark: "#38bdf8"},
		You:    lipgloss.AdaptiveColor{Light: "#7c3aed", Dark: "#a78bfa"},
		Text:   lipgloss.AdaptiveColor{Light: "#16161a", Dark: "#ededf0"},
		Muted:  lipgloss.AdaptiveColor{Light: "#6b6b76", Dark: "#9494a0"},
		Border: lipgloss.AdaptiveColor{Light: "#d4d4d9", Dark: "#3f3f4a"},
		Good:   lipgloss.AdaptiveColor{Light: "#15803d", Dark: "#4ade80"},
		Bad:    lipgloss.AdaptiveColor{Light: "#b91c1c", Dark: "#f87171"},
	}
}

// withDefaults fills in whatever the caller left unset.
func (t Theme) withDefaults() Theme {
	d := DefaultTheme()
	for _, pair := range []struct {
		set  *lipgloss.TerminalColor
		zero lipgloss.TerminalColor
	}{
		{&t.Accent, d.Accent},
		{&t.You, d.You},
		{&t.Text, d.Text},
		{&t.Muted, d.Muted},
		{&t.Border, d.Border},
		{&t.Good, d.Good},
		{&t.Bad, d.Bad},
	} {
		if *pair.set == nil {
			*pair.set = pair.zero
		}
	}
	return t
}

// Branding is what makes the terminal UI one application's rather than another's. Every
// field has a plain default, so an application that says nothing still gets a usable
// conversation.
type Branding struct {
	// Banner is drawn above the header, usually a few lines of block characters. Empty
	// leaves it out, and a terminal too short to spare the lines leaves it out anyway.
	Banner string
	// Title names the agent and Subtitle says in a few words what it is for.
	Title, Subtitle string
	// You and Agent label the two sides of the conversation.
	You, Agent string
	// Placeholder is what an empty composer invites.
	Placeholder string
	// Help is the key reminder under the composer.
	Help string
}

func (b Branding) withDefaults() Branding {
	if b.Title == "" {
		b.Title = "Agent"
	}
	if b.You == "" {
		b.You = "YOU"
	}
	if b.Agent == "" {
		b.Agent = "AGENT"
	}
	if b.Placeholder == "" {
		b.Placeholder = "Ask a question…"
	}
	if b.Help == "" {
		b.Help = "enter send · alt-enter newline · esc cancel · pgup/pgdn scroll · /help"
	}
	return b
}

// styles are the theme resolved into what the renderers actually use.
type styles struct {
	title, accent, text, muted, good, bad, you lipgloss.Style
	frame                                      lipgloss.Style
}

// composer is where a question is typed. Everything a text area draws around itself — a
// prompt column, a highlighted cursor line, an end-of-buffer marker — is taken off,
// because the frame the composer sits in is what says where it is.
func composer(branding Branding, s styles) textarea.Model {
	input := textarea.New()
	input.Placeholder = branding.Placeholder
	input.Prompt = ""
	input.ShowLineNumbers = false
	input.CharLimit = maxQuestion
	input.SetHeight(composerHeight)
	bare := lipgloss.NewStyle()
	for _, style := range []*textarea.Style{&input.FocusedStyle, &input.BlurredStyle} {
		style.Base, style.CursorLine, style.EndOfBuffer, style.Prompt = bare, bare, bare, bare
		style.Text, style.Placeholder = s.text, s.muted
	}
	input.Focus()
	return input
}

func newStyles(t Theme) styles {
	t = t.withDefaults()
	return styles{
		title:  lipgloss.NewStyle().Foreground(t.Accent).Bold(true),
		accent: lipgloss.NewStyle().Foreground(t.Accent),
		text:   lipgloss.NewStyle().Foreground(t.Text),
		muted:  lipgloss.NewStyle().Foreground(t.Muted),
		good:   lipgloss.NewStyle().Foreground(t.Good),
		bad:    lipgloss.NewStyle().Foreground(t.Bad),
		you:    lipgloss.NewStyle().Foreground(t.You).Bold(true),
		frame:  lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(t.Border).Padding(0, 1),
	}
}
