package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// busyConversation is as much as the interface ever has to show at once: a banner, a
// subtitle, a scope, the application's own header lines, a truncation notice, a long
// markdown answer, several tools and the command list.
func busyConversation(t *testing.T) *Model {
	t.Helper()
	m := newModel(t, Options{
		Branding: Branding{
			Banner:      "   ▄▄▄  STREAM\n  ▀▄▄   SUPPORT\n  ▄▄▀   source-backed answers",
			Title:       "Stream Support",
			Subtitle:    "source-backed answers",
			Placeholder: "Ask about Chat, Video, Moderation or Feeds…",
		},
		Header: func(State) []string {
			return []string{"Organization 1234 · memory enabled", "Documentation and SDK source available"}
		},
		Commands: []Command{{Name: "ticket", Args: "<id>", Help: "open a support ticket", Run: nothing}},
	})
	started := time.Now().Add(-30 * time.Second)
	finished := started.Add(24 * time.Second)
	m.scope = "chat / react"
	m.truncated = true
	m.notice = commandHelp(m.styles, m.options.Commands, 70)
	m.busy, m.submittedAt = true, started
	m.conversationID = "agent:support-27392eaf-1f43-4c1a-87e4-4dd113482218"
	m.messages = []stream.ConversationMessage{
		{ID: "q", Role: "user", Text: strings.Repeat("a long question that has to be wrapped somewhere ", 4)},
		{ID: "a", Role: "assistant", State: "tools", StartedAt: started, Text: "## Heading\n\n" + strings.Repeat("An answer with `code` in it and a [link](https://example.com/very/long/path). ", 6),
			Tools: []stream.ToolActivity{
				{ID: "t1", Title: "Search the compiled documentation", Status: "completed", Summary: "Verified 2 source citations against the pinned revision", StartedAt: started, FinishedAt: &finished},
				{ID: "t2", Title: "Read source", Status: "running", Phase: "executing", StartedAt: started},
			}},
	}
	m.refresh()
	return m
}

func TestTheInterfaceFillsTheTerminalAndNothingBeyondIt(t *testing.T) {
	m := busyConversation(t)
	for _, size := range []struct{ width, height int }{
		{80, 24}, {80, 30}, {120, 40}, {200, 60}, {100, 20},
		{60, 16}, {40, 14}, {30, 12}, {24, 10}, {20, 8}, {12, 6}, {10, 4},
	} {
		m.Update(tea.WindowSizeMsg{Width: size.width, Height: size.height})
		view := m.View()
		if got := lipgloss.Height(view); got != size.height {
			t.Errorf("at %dx%d the interface is %d lines tall", size.width, size.height, got)
		}
		for i, line := range strings.Split(view, "\n") {
			if got := ansi.StringWidth(line); got > size.width {
				t.Errorf("at %dx%d line %d is %d columns wide: %q", size.width, size.height, i, got, plain(line))
			}
		}
		if m.view.Height < 1 {
			t.Errorf("at %dx%d the conversation was given %d lines", size.width, size.height, m.view.Height)
		}
	}
}

func TestTheFramesReachBothEdgesOfTheTerminal(t *testing.T) {
	m := busyConversation(t)
	for _, width := range []int{32, 60, 88, 120, 200} {
		m.Update(tea.WindowSizeMsg{Width: width, Height: 40})
		framed := 0
		for _, line := range screen(m) {
			if !strings.ContainsAny(line, "╭│╰") {
				continue
			}
			framed++
			if got := ansi.StringWidth(line); got != width {
				t.Errorf("at %d columns a frame line is %d wide: %q", width, got, line)
			}
		}
		if framed == 0 {
			t.Errorf("at %d columns nothing was framed", width)
		}
	}
}

func TestChromeIsGivenUpBeforeTheConversationIs(t *testing.T) {
	m := busyConversation(t)

	m.Update(tea.WindowSizeMsg{Width: 80, Height: 40})
	if m.compact != 0 {
		t.Fatalf("a tall terminal is drawn at compactness %d", m.compact)
	}
	if !strings.Contains(plain(m.View()), "STREAM") {
		t.Error("a tall terminal does not show the banner")
	}
	if m.view.Height < minTranscript {
		t.Errorf("a tall terminal left %d lines of conversation", m.view.Height)
	}

	// The banner is the first thing to go, and the frames the next.
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 16})
	if strings.Contains(plain(m.View()), "▄▄▄") {
		t.Errorf("a short terminal still shows the banner:\n%s", plain(m.View()))
	}
	if !strings.Contains(plain(m.View()), "Stream Support") {
		t.Error("a short terminal forgot who is being talked to")
	}

	m.Update(tea.WindowSizeMsg{Width: 80, Height: 9})
	if m.framed() {
		t.Errorf("a very short terminal still spends lines on frames:\n%s", plain(m.View()))
	}
	if m.view.Height < 1 {
		t.Error("the conversation was squeezed out entirely")
	}

	// A narrow terminal has no room for a frame either, whatever its height.
	m.Update(tea.WindowSizeMsg{Width: 24, Height: 40})
	if m.framed() {
		t.Error("a narrow terminal spends columns on frames")
	}
}

func TestTheHeaderSaysWhichConversationThisIs(t *testing.T) {
	m := newModel(t, Options{Branding: Branding{Title: "Jean"}})

	if header := plain(m.header()); !strings.Contains(header, "no conversation") {
		t.Errorf("a conversation that has not opened says:\n%s", header)
	}
	m.connecting = true
	if header := plain(m.header()); !strings.Contains(header, "opening…") {
		t.Errorf("a conversation being opened says:\n%s", header)
	}
	m.connecting, m.conversationID = false, "agent:support-9"
	header := plain(m.header())
	if !strings.Contains(header, "agent:support-9") || !strings.Contains(header, "Jean") {
		t.Errorf("an open conversation says:\n%s", header)
	}
}

func TestTheHeaderCarriesWhatTheApplicationAsksItTo(t *testing.T) {
	var asked State
	m := newModel(t, Options{
		Branding: Branding{Title: "Jean", Subtitle: "answers questions"},
		Header: func(state State) []string {
			asked = state
			return []string{"Organization 1234", ""}
		},
	})
	m.conversationID, m.scope, m.busy = "agent:support-9", "chat / react", true

	header := plain(m.header())
	for _, want := range []string{"answers questions", "chat / react", "Organization 1234"} {
		if !strings.Contains(header, want) {
			t.Errorf("the header does not say %q:\n%s", want, header)
		}
	}
	// An empty line is left out rather than drawn as a blank row inside the frame.
	if lipgloss.Height(m.header()) != 6 {
		t.Errorf("the header is %d lines tall:\n%s", lipgloss.Height(m.header()), header)
	}
	if asked.ConversationID != "agent:support-9" || asked.Scope != "chat / react" || !asked.Busy {
		t.Errorf("the application was asked about %+v", asked)
	}

	m.truncated = true
	if !strings.Contains(plain(m.header()), "/older") {
		t.Errorf("a truncated session does not say how to read further back:\n%s", plain(m.header()))
	}
}

func TestTheStatusLineTimesTheQuestionAndMarksFailure(t *testing.T) {
	m := newModel(t, Options{})

	m.say("Ready")
	if status := plain(m.statusLine()); status != "Ready" {
		t.Errorf("an idle conversation says %q", status)
	}

	m.busy, m.submittedAt = true, time.Now().Add(-12300*time.Millisecond)
	status := plain(m.statusLine())
	if !strings.Contains(status, "12.3s total") {
		t.Errorf("a question being answered says %q", status)
	}
	if !strings.HasPrefix(status, m.spinner()) {
		t.Errorf("a question being answered does not look like it: %q", status)
	}

	m.busy = false
	m.fail(context.DeadlineExceeded)
	if !m.statusFailed || !strings.Contains(plain(m.statusLine()), "deadline") {
		t.Errorf("a failure says %q", plain(m.statusLine()))
	}
	m.say("Ready")
	if m.statusFailed {
		t.Error("the failure outlived what went wrong")
	}
}

func TestTheHelpMovesUnderTheStatusWhenTheyCannotShare(t *testing.T) {
	m := busyConversation(t)

	m.Update(tea.WindowSizeMsg{Width: 200, Height: 40})
	footer := strings.Split(plain(m.footer()), "\n")
	last := footer[len(footer)-1]
	if !strings.Contains(last, "total") || !strings.Contains(last, "enter send") {
		t.Errorf("a wide terminal splits the footer:\n%s", plain(m.footer()))
	}

	m.Update(tea.WindowSizeMsg{Width: 60, Height: 40})
	footer = strings.Split(plain(m.footer()), "\n")
	if len(footer) < 2 {
		t.Fatalf("the footer is one line:\n%s", plain(m.footer()))
	}
	status, help := footer[len(footer)-2], footer[len(footer)-1]
	if !strings.Contains(status, "total") || !strings.Contains(help, "enter send") {
		t.Errorf("a narrow terminal does not stack the footer:\n%s", plain(m.footer()))
	}
}

func TestSpreadPutsTheRightAgainstTheRightEdge(t *testing.T) {
	if got := spread("left", "right", 20); got != "left           right" {
		t.Errorf("spread gave %q", got)
	}
	// The left is what gives way, because the right is the more particular of the two.
	if got := spread("a very long title indeed", "id-9", 12); got != "a very l… id-9" && ansi.StringWidth(got) > 12 {
		t.Errorf("spread gave %q at %d columns", got, ansi.StringWidth(got))
	}
	if got := spread("left", "a right too long for this", 8); ansi.StringWidth(got) != 8 {
		t.Errorf("spread gave %q at %d columns", got, ansi.StringWidth(got))
	}
	if got := spread("left", "right", 0); got != "" {
		t.Errorf("spread into nothing gave %q", got)
	}
	if _, fits := spreadIfFits("left", "right", 10); fits {
		t.Error("two pieces with no gap between them were said to fit")
	}
	if _, fits := spreadIfFits("left", "right", 11); !fits {
		t.Error("two pieces with a gap between them were said not to fit")
	}
}

func TestFitMakesABlockExactlyAsTallAsAsked(t *testing.T) {
	if got := lipgloss.Height(fit("one\ntwo", 5)); got != 5 {
		t.Errorf("a short block came out %d lines tall", got)
	}
	if got := fit("one\ntwo\nthree", 2); got != "one\ntwo" {
		t.Errorf("a tall block came out as %q", got)
	}
	if got := fit("one", 0); got != "" {
		t.Errorf("fitting into nothing gave %q", got)
	}
}

func TestIndentShiftsEveryLine(t *testing.T) {
	if got := indent("one\ntwo", 2); got != "  one\n  two" {
		t.Errorf("indent gave %q", got)
	}
	if got := indent("one", 0); got != "one" {
		t.Errorf("indenting by nothing gave %q", got)
	}
}

func TestNothingIsDrawnBeforeTheTerminalSaysHowBigItIs(t *testing.T) {
	m, err := New(t.Context(), Options{Open: func(context.Context, string) (Session, error) { return nil, nil }})
	if err != nil {
		t.Fatal(err)
	}
	// A size is assumed until the terminal says otherwise, so an embedded conversation
	// still draws something.
	if m.View() == "" {
		t.Error("the conversation drew nothing at its assumed size")
	}
	m.width, m.height = 0, 0
	if m.View() != "" {
		t.Error("the conversation drew something into a terminal of no size")
	}
}

func nothing(context.Context, []string) (string, error) { return "", nil }
