package tui

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	tea "github.com/charmbracelet/bubbletea"
)

func TestAskingSomethingShowsItBeforeTheBackendHasSavedIt(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})

	settle(m, ask(m, "what does useChatContext return?"))

	if !m.busy || m.status != "Thinking…" {
		t.Errorf("the conversation is busy %v saying %q", m.busy, m.status)
	}
	if m.submittedAt.IsZero() {
		t.Error("the question is not being timed")
	}
	if !shown(m, "what does useChatContext return?") {
		t.Errorf("the question is not on screen:\n%s", plain(m.transcript()))
	}
	if m.input.Value() != "" {
		t.Errorf("the composer still holds %q", m.input.Value())
	}
	backend.commands(t, "respond")
}

func TestAQuestionIsNotAskedTwiceWhileOneIsBeingAnswered(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})

	settle(m, ask(m, "the first question"))
	settle(m, ask(m, "the second question"))

	if shown(m, "the second question") {
		t.Errorf("a second question went out mid-answer:\n%s", plain(m.transcript()))
	}
	// It is still in the composer, so nothing anybody typed was thrown away.
	if m.input.Value() != "the second question" {
		t.Errorf("the composer holds %q", m.input.Value())
	}
}

func TestNothingIsAskedBeforeThereIsASessionToAskIt(t *testing.T) {
	m := newModel(t, Options{})
	settle(m, ask(m, "what is the weather"))
	if m.busy || shown(m, "what is the weather") {
		t.Errorf("a question went out with no session:\n%s", plain(m.transcript()))
	}
}

func TestAnEmptyComposerSubmitsNothing(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})

	settle(m, ask(m, "   \n  "))
	if m.busy || len(m.messages) != 0 {
		t.Errorf("whitespace was asked as a question: %+v", m.messages)
	}
}

func TestAltEnterWritesAnotherLineRatherThanSending(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})

	m.input.SetValue("the first line")
	send(m, "alt+enter")
	if !strings.Contains(m.input.Value(), "\n") {
		t.Errorf("the composer holds %q", m.input.Value())
	}
	if m.busy {
		t.Error("a newline sent the question")
	}
}

func TestEscapeCancelsOnlyWhatIsBeingAnswered(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})

	// Nothing is being answered, so there is nothing to cancel.
	settle(m, send(m, "esc"))
	if m.status != "Ready" {
		t.Errorf("escape on an idle conversation says %q", m.status)
	}

	settle(m, ask(m, "a question worth cancelling"))
	settle(m, send(m, "esc"))
	if m.status != "Cancelling…" {
		t.Errorf("escape mid-answer says %q", m.status)
	}
	backend.commands(t, "respond", "interrupt")
}

func TestCtrlCLeaves(t *testing.T) {
	m := newModel(t, Options{})
	if _, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC}); cmd == nil {
		t.Fatal("ctrl-c did not leave")
	} else if _, quit := cmd().(tea.QuitMsg); !quit {
		t.Error("ctrl-c did something other than leave")
	}
}

func TestQuitLeaves(t *testing.T) {
	m := newModel(t, Options{})
	cmd := ask(m, "/quit")
	if cmd == nil {
		t.Fatal("/quit did not leave")
	}
	if messages := step(cmd); len(messages) == 0 {
		t.Fatal("/quit produced nothing")
	} else if _, quit := messages[0].(tea.QuitMsg); !quit {
		t.Errorf("/quit produced %T", messages[0])
	}
}

func TestReadingFurtherUpIsNotInterruptedByTheAnswerBelow(t *testing.T) {
	m := newModel(t, Options{})
	m.messages = []stream.ConversationMessage{{ID: "q", Role: "user", Text: strings.Repeat("a line of the question\n", 60)}}
	m.refresh()
	m.view.GotoBottom()

	send(m, "pgup")
	if m.view.AtBottom() {
		t.Fatal("paging up did not move")
	}
	offset := m.view.YOffset

	m.busy = true
	m.Update(tick(time.Now()))
	m.Update(received{generation: m.generation, ok: true})
	if m.view.YOffset != offset {
		t.Errorf("the view moved from %d to %d while somebody was reading", offset, m.view.YOffset)
	}

	// Asking something new is asking to see it.
	backend := newRouter(t)
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})
	m.busy = false
	settle(m, ask(m, "another question"))
	if !m.view.AtBottom() {
		t.Error("a new question did not bring the view back down")
	}
}

func TestAResizeRewrapsEveryAnswer(t *testing.T) {
	m := newModel(t, Options{})
	m.markdown("an answer", 70)
	if len(m.cache) == 0 {
		t.Fatal("nothing was rendered")
	}
	m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	if len(m.cache) != 0 {
		t.Errorf("%d renders wrapped to the old width survived", len(m.cache))
	}
	if m.view.Width <= 80 || m.input.Width() <= 80 {
		t.Errorf("the conversation is %d and the composer %d columns wide", m.view.Width, m.input.Width())
	}
}

func TestTheQuestionShownIsReplacedByTheOneThatWasSaved(t *testing.T) {
	m := newModel(t, Options{})
	m.upsert(stream.ConversationMessage{ID: pendingID, Role: "user", Text: "the question"})
	m.upsert(stream.ConversationMessage{ID: "q1", Role: "user", Text: "the question"})
	m.upsert(stream.ConversationMessage{ID: "a1", Role: "assistant", Text: "part of the answer"})
	m.upsert(stream.ConversationMessage{ID: "a1", Role: "assistant", Text: "the whole answer"})

	if len(m.messages) != 2 {
		t.Fatalf("the conversation holds %d messages: %+v", len(m.messages), m.messages)
	}
	if m.messages[0].ID != "q1" || m.messages[1].Text != "the whole answer" {
		t.Errorf("the conversation holds %+v", m.messages)
	}
	// The answer now has a question to belong to.
	if m.activeQuestionID != "q1" {
		t.Errorf("the question being answered is %q", m.activeQuestionID)
	}
}

func TestAnAnswerToAnEarlierQuestionDoesNotTakeOverTheStatusLine(t *testing.T) {
	m := newModel(t, Options{})
	m.activeQuestionID = "q2"
	m.busy = true

	m.Update(received{generation: m.generation, ok: true, event: stream.Event{
		Kind:  "conversation_updated",
		Frame: stream.Frame{"message": stream.ConversationMessage{ID: "a1", QuestionID: "q1", Role: "assistant", State: "completed"}},
	}})
	if !m.busy || m.status == label("completed") {
		t.Errorf("an earlier answer left the conversation busy %v saying %q", m.busy, m.status)
	}

	m.Update(received{generation: m.generation, ok: true, event: stream.Event{
		Kind:  "conversation_updated",
		Frame: stream.Frame{"message": stream.ConversationMessage{ID: "a2", QuestionID: "q2", Role: "assistant", State: "completed"}},
	}})
	if m.busy || m.status != label("completed") {
		t.Errorf("the answer being waited on left the conversation busy %v saying %q", m.busy, m.status)
	}
}

func TestWhatASupersededSessionSaysIsDropped(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})
	stale := m.generation

	// Opening another conversation is the end of the last one's say in things.
	settle(m, ask(m, "/new"))
	m.Update(received{generation: stale, ok: true, event: stream.Event{
		Kind:  "conversation_updated",
		Frame: stream.Frame{"message": stream.ConversationMessage{ID: "a", Role: "assistant", Text: "an answer to the last conversation"}},
	}})
	if shown(m, "an answer to the last conversation") {
		t.Errorf("the replaced session still writes to the screen:\n%s", plain(m.transcript()))
	}

	m.Update(opened{generation: stale, session: backend.session(t, ""), page: stream.ConversationPage{
		Messages: []stream.ConversationMessage{{ID: "old", Role: "user", Text: "history from the last conversation"}},
	}})
	if shown(m, "history from the last conversation") {
		t.Errorf("the replaced session's history landed:\n%s", plain(m.transcript()))
	}
}

func TestASessionThatEndsSaysHowToGetBackToIt(t *testing.T) {
	m := newModel(t, Options{})
	m.conversationID, m.busy = "agent:support-9", true

	m.Update(received{generation: m.generation, ok: false})
	if m.busy {
		t.Error("the conversation is still waiting on a session that ended")
	}
	if !strings.Contains(m.status, "/resume agent:support-9") {
		t.Errorf("a session that ended says %q", m.status)
	}
}

func TestAnErrorFromTheBackendReachesTheStatusLine(t *testing.T) {
	m := newModel(t, Options{})
	m.busy = true

	m.Update(received{generation: m.generation, ok: true, event: stream.Event{Kind: "error", Error: "the worker is unavailable"}})
	if m.busy || !m.statusFailed || m.status != "the worker is unavailable" {
		t.Errorf("the conversation is busy %v saying %q", m.busy, m.status)
	}
}

func TestAConversationThatWillNotOpenSaysWhy(t *testing.T) {
	m := newModel(t, Options{Open: func(context.Context, string) (Session, error) {
		return nil, errors.New("the router is not running")
	}})

	settle(m, m.open(""))
	if m.connecting || m.session != nil {
		t.Error("the conversation thinks it opened")
	}
	if !m.statusFailed || m.status != "the router is not running" {
		t.Errorf("the status line says %q", m.status)
	}
}

func TestOpeningAConversationForgetsTheOneBeforeIt(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{
		Messages: []stream.ConversationMessage{{ID: "q", Role: "user", Text: "an earlier question"}},
	})
	m.scope, m.before, m.truncated = "chat / react", "cursor-2", true

	m.open("")
	if len(m.messages) != 0 || m.scope != "" || m.before != "" || m.truncated {
		t.Errorf("the new conversation starts with %d messages, scope %q, cursor %q, truncated %v",
			len(m.messages), m.scope, m.before, m.truncated)
	}
	if !m.connecting || m.session != nil {
		t.Errorf("the new conversation is connecting %v with session %v", m.connecting, m.session != nil)
	}
}

func TestTheScopeFollowsWhatTheToolsAreWorkingIn(t *testing.T) {
	m := newModel(t, Options{})
	m.rescope(stream.ConversationMessage{Tools: []stream.ToolActivity{
		{ID: "t1", Product: "chat"},
		{ID: "t2", Product: "video", SDK: "ios-swiftui"},
	}})
	if m.scope != "video / ios-swiftui" {
		t.Errorf("the scope is %q", m.scope)
	}

	// A tool that works in no particular scope leaves the last one standing.
	m.rescope(stream.ConversationMessage{Tools: []stream.ToolActivity{{ID: "t3"}}})
	if m.scope != "video / ios-swiftui" {
		t.Errorf("the scope is %q", m.scope)
	}
	if m.State().Scope != "video / ios-swiftui" {
		t.Errorf("the state reports scope %q", m.State().Scope)
	}
}

func TestTheStateIsWhatTheApplicationNeedsToKnow(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	if state := m.State(); state.SessionID != "" {
		t.Errorf("a conversation with no session reports session %q", state.SessionID)
	}

	attach(t, m, backend.session(t, "agent:support-9"), stream.ConversationPage{})
	state := m.State()
	if state.ConversationID != "agent:support-9" || state.SessionID != "session-1" {
		t.Errorf("the state is %+v", state)
	}
}
