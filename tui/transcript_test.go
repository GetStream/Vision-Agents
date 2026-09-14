package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// conversationAt is a question and the answer being written to it, as the backend reports
// it while it works.
func conversationAt(state string, started time.Time) []stream.ConversationMessage {
	return []stream.ConversationMessage{
		{ID: "q", Role: "user", Text: "What does useChatContext return?"},
		{ID: "a", Role: "assistant", State: state, StartedAt: started},
	}
}

func TestBothSidesOfTheConversationAreLabelled(t *testing.T) {
	m := newModel(t, Options{Branding: Branding{You: "ME", Agent: "JEAN"}})
	m.messages = conversationAt("writing", time.Now())
	m.messages[1].Text = "It returns ChatContextValue."

	transcript := plain(m.transcript())
	if !strings.Contains(transcript, "ME") || !strings.Contains(transcript, "JEAN") {
		t.Errorf("neither side is named:\n%s", transcript)
	}
	if !shown(m, "What does useChatContext return?") {
		t.Errorf("the question is missing:\n%s", transcript)
	}
	if !shown(m, "It returns ChatContextValue.") {
		t.Errorf("the answer is missing:\n%s", transcript)
	}
	// Every line of a turn carries the bar, so a turn reads as one thing somebody said.
	for _, line := range strings.Split(transcript, "\n") {
		if line != "" && !strings.HasPrefix(line, "▎") {
			t.Errorf("line %q is outside both turns", line)
		}
	}
}

func TestEveryStateIsSaidInWords(t *testing.T) {
	m := newModel(t, Options{})
	for _, state := range []string{"thinking", "queued", "tools", "writing", "completed", "failed", "cancelled", "interrupted"} {
		m.messages = conversationAt(state, time.Now())
		if !shown(m, label(state)) {
			t.Errorf("state %q is not said as %q:\n%s", state, label(state), plain(m.transcript()))
		}
	}
	// A state this version has never heard of is shown rather than swallowed.
	m.messages = conversationAt("reticulating", time.Now())
	if !shown(m, "reticulating") {
		t.Errorf("an unknown state went missing:\n%s", plain(m.transcript()))
	}
}

func TestAnUnfinishedAnswerTimesTheStateItIsIn(t *testing.T) {
	m := newModel(t, Options{})
	// The turn began ten seconds ago, but it has only been writing for three.
	m.messages = conversationAt("writing", time.Now().Add(-10*time.Second))
	m.messages[1].StateStartedAt = time.Now().Add(-3 * time.Second)
	if !shown(m, "Writing the answer · 3.0s") {
		t.Errorf("the state is timed from the wrong moment:\n%s", plain(m.transcript()))
	}

	// Once it is finished, the whole turn is what was timed.
	finished := m.messages[1].StartedAt.Add(12 * time.Second)
	m.messages[1].State, m.messages[1].FinishedAt = "completed", &finished
	if !shown(m, "Completed · 12.0s") {
		t.Errorf("the finished turn is timed from the wrong moment:\n%s", plain(m.transcript()))
	}
}

func TestAFinishedAnswerSaysWhetherItSurvivedBeingSaved(t *testing.T) {
	m := newModel(t, Options{})
	started := time.Now().Add(-4 * time.Second)
	finished := started.Add(4 * time.Second)
	m.messages = conversationAt("completed", started)
	m.messages[1].FinishedAt = &finished

	if !shown(m, "saving…") {
		t.Errorf("an unsaved answer does not say so:\n%s", plain(m.transcript()))
	}
	m.messages[1].Saved = true
	if !shown(m, "· saved") || shown(m, "saving…") {
		t.Errorf("a saved answer does not say so:\n%s", plain(m.transcript()))
	}
	m.messages[1].Saved, m.messages[1].Error = false, "stream unavailable"
	if !shown(m, "stream unavailable") {
		t.Errorf("a failed save is not reported:\n%s", plain(m.transcript()))
	}
}

func TestAToolSaysHowItWentHowLongItTookAndWhatItFound(t *testing.T) {
	m := newModel(t, Options{})
	started := time.Now().Add(-25 * time.Second)
	finished := started.Add(24900 * time.Millisecond)
	m.messages = conversationAt("tools", started)
	m.messages[1].Tools = []stream.ToolActivity{
		{ID: "t1", Title: "Search source", Status: "completed", Phase: "done", Summary: "Verified 2 citations", StartedAt: started, FinishedAt: &finished},
		{ID: "t2", Title: "Read source", Status: "running", Phase: "executing", StartedAt: started},
	}

	if !shown(m, "✓ Search source 24.9s Verified 2 citations") {
		t.Errorf("a finished tool does not read right:\n%s", plain(m.transcript()))
	}
	// With nothing to say for itself, a tool reports the phase it is in instead.
	if !shown(m, "Read source") || !shown(m, "executing") {
		t.Errorf("a running tool does not read right:\n%s", plain(m.transcript()))
	}

	for status, mark := range map[string]string{"completed": "✓", "failed": "✗", "cancelled": "–"} {
		m.messages[1].Tools[0].Status = status
		if !shown(m, mark+" Search source") {
			t.Errorf("status %q is not marked %q:\n%s", status, mark, plain(m.transcript()))
		}
	}
	// Work still going is marked with whichever frame the spinner is on.
	m.messages[1].Tools[0].Status = "running"
	if !shown(m, m.spinner()+" Search source") {
		t.Errorf("running work is not marked as running:\n%s", plain(m.transcript()))
	}
}

func TestToolDurationsLineUpUnderEachOther(t *testing.T) {
	m := newModel(t, Options{})
	started := time.Now()
	finished := started.Add(time.Second)
	m.messages = conversationAt("tools", started)
	m.messages[1].Tools = []stream.ToolActivity{
		{ID: "t1", Title: "Read", Status: "completed", StartedAt: started, FinishedAt: &finished},
		{ID: "t2", Title: "Search the source", Status: "completed", StartedAt: started, FinishedAt: &finished},
	}

	var columns []int
	for _, row := range m.toolRows(m.messages[1].Tools, 76) {
		columns = append(columns, strings.Index(plain(row), "1.0s"))
	}
	if len(columns) != 2 || columns[0] != columns[1] || columns[0] < 0 {
		t.Errorf("the durations are at columns %v", columns)
	}
}

func TestAQuestionCannotStyleTheConversation(t *testing.T) {
	m := newModel(t, Options{})
	m.messages = []stream.ConversationMessage{{ID: "q", Role: "user", Text: "look \x1b[31mat\x1b[0m this"}}

	if !shown(m, "look at this") {
		t.Errorf("the question did not survive:\n%s", plain(m.transcript()))
	}
	// The bar is styled; nothing inside the question is.
	question := strings.SplitN(m.transcript(), "\n", 2)[1]
	if strings.Contains(strings.TrimPrefix(question, m.styles.you.Render("▎")), "\x1b[31m") {
		t.Errorf("an escape sequence in a question reached the screen: %q", question)
	}
}

func TestALongQuestionIsWrappedRatherThanCut(t *testing.T) {
	m := newModel(t, Options{})
	m.messages = []stream.ConversationMessage{{ID: "q", Role: "user", Text: strings.Repeat("word ", 60)}}

	transcript := plain(m.transcript())
	if lines := strings.Count(transcript, "\n"); lines < 4 {
		t.Errorf("a 300-character question came out on %d lines:\n%s", lines+1, transcript)
	}
	if !strings.Contains(transcript, "word word") {
		t.Errorf("the question was lost:\n%s", transcript)
	}
	if strings.Contains(transcript, "…") {
		t.Errorf("the question was truncated rather than wrapped:\n%s", transcript)
	}
}

func TestAnAnswerIsRenderedAsMarkdownAndKeptForTheNextFrame(t *testing.T) {
	m := newModel(t, Options{})
	answer := "# Heading\n\nSome `code` and a [link](https://example.com)."

	rendered := m.markdown(answer, 70)
	if plain(rendered) == answer {
		t.Errorf("the answer was not rendered at all: %q", rendered)
	}
	if !strings.Contains(plain(rendered), "Heading") || !strings.Contains(plain(rendered), "code") {
		t.Errorf("the answer lost its words: %q", plain(rendered))
	}
	if again := m.markdown(answer, 70); again != rendered {
		t.Error("the same answer at the same width was rendered twice")
	}
	// A resize is a different render, not a cache hit.
	if narrow := m.markdown(answer, 40); narrow == rendered {
		t.Error("the answer was not re-rendered for a different width")
	}
	if len(m.cache) != 2 {
		t.Errorf("the cache holds %d renders", len(m.cache))
	}
}

func TestTheRenderCacheDoesNotGrowForever(t *testing.T) {
	m := newModel(t, Options{})
	for i := 0; i < markdownCacheSize+10; i++ {
		m.markdown(strings.Repeat("a", i+1), 70)
	}
	if len(m.cache) > markdownCacheSize {
		t.Errorf("the cache holds %d renders, over the %d it is allowed", len(m.cache), markdownCacheSize)
	}
}

func TestElapsedIsWhatHasPassedOrWhatDid(t *testing.T) {
	start := time.Now().Add(-3 * time.Second)
	finished := start.Add(10 * time.Second)

	if got := elapsed(start, &finished); got != 10*time.Second {
		t.Errorf("finished work took %s", got)
	}
	if got := elapsed(start, nil); got < 3*time.Second || got > 4*time.Second {
		t.Errorf("work still going has taken %s", got)
	}
	// Nothing has started, so nothing has taken any time.
	if got := elapsed(time.Time{}, nil); got != 0 {
		t.Errorf("unstarted work has taken %s", got)
	}
	if got := seconds(1234500 * time.Millisecond); got != "1234.5s" {
		t.Errorf("a long duration is written %q", got)
	}
}

func TestTheSpinnerGoesRoundAndComesBack(t *testing.T) {
	m := newModel(t, Options{})
	first := m.spinner()
	for i := 0; i < len(spinnerFrames); i++ {
		m.Update(tick(time.Now()))
	}
	if m.spinner() != first {
		t.Errorf("the spinner landed on %q rather than back on %q", m.spinner(), first)
	}
}
