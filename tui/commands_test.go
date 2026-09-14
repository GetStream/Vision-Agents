package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func TestParseCommandSeparatesTheNameFromWhatFollowsIt(t *testing.T) {
	for _, c := range []struct {
		line string
		name string
		args []string
		ok   bool
	}{
		{line: "/new", name: "new", ok: true},
		{line: "/resume agent:support-9", name: "resume", args: []string{"agent:support-9"}, ok: true},
		{line: "/ticket  4821  urgent", name: "ticket", args: []string{"4821", "urgent"}, ok: true},
		{line: "what does /new do"},
		{line: "/"},
		{line: "/   "},
	} {
		name, args, ok := parseCommand(c.line)
		if ok != c.ok || name != c.name || strings.Join(args, ",") != strings.Join(c.args, ",") {
			t.Errorf("%q parsed as %q %v %v", c.line, name, args, ok)
		}
	}
}

func TestACommandThatCouldNotBeReachedIsRefused(t *testing.T) {
	for _, c := range []struct {
		why     string
		command Command
	}{
		{"nameless", Command{Run: nothing}},
		{"a name with a space in it", Command{Name: "open ticket", Run: nothing}},
		{"a name with a slash in it", Command{Name: "/ticket", Run: nothing}},
		{"nothing to run", Command{Name: "ticket"}},
		{"shadowing a built-in", Command{Name: "resume", Run: nothing}},
	} {
		if err := validateCommands([]Command{c.command}); err == nil {
			t.Errorf("a command with %s was accepted", c.why)
		}
	}
	twice := []Command{{Name: "ticket", Run: nothing}, {Name: "ticket", Run: nothing}}
	if err := validateCommands(twice); err == nil {
		t.Error("the same command was accepted twice")
	}
	if err := validateCommands([]Command{{Name: "ticket", Run: nothing}}); err != nil {
		t.Errorf("a usable command was refused: %v", err)
	}

	// A conversation is not built at all around a command nobody could type.
	if _, err := New(t.Context(), Options{
		Open:     func(context.Context, string) (Session, error) { return nil, nil },
		Commands: []Command{{Name: "new", Run: nothing}},
	}); err == nil {
		t.Error("a conversation was built around a command shadowing /new")
	}
}

func TestAConversationNeedsSomewhereToOpen(t *testing.T) {
	if _, err := New(t.Context(), Options{}); err == nil {
		t.Error("a conversation with nothing to open was built")
	}
}

func TestHelpListsEveryCommandIncludingTheApplicationsOwn(t *testing.T) {
	m := newModel(t, Options{Commands: []Command{{Name: "ticket", Args: "<id>", Help: "open a support ticket", Run: nothing}}})

	settle(m, ask(m, "/help"))
	transcript := plain(m.transcript())
	for _, want := range []string{"/new", "/resume <id>", "/older", "/help", "/quit", "/ticket <id>", "open a support ticket"} {
		if !strings.Contains(transcript, want) {
			t.Errorf("the help does not list %q:\n%s", want, transcript)
		}
	}
	if m.input.Value() != "" {
		t.Errorf("the composer still holds %q", m.input.Value())
	}
	// The help goes away once there is a question to read instead.
	backend := newRouter(t)
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})
	settle(m, ask(m, "what is the weather"))
	if strings.Contains(plain(m.transcript()), "open a support ticket") {
		t.Error("the help outlived the question that replaced it")
	}
}

func TestAnUnknownCommandSaysWhereToLook(t *testing.T) {
	m := newModel(t, Options{})
	settle(m, ask(m, "/wibble"))
	if !strings.Contains(m.status, "/wibble") || !strings.Contains(m.status, "/help") {
		t.Errorf("an unknown command says %q", m.status)
	}
}

func TestResumeNeedsToBeToldWhatToResume(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})

	settle(m, ask(m, "/resume"))
	if !strings.Contains(m.status, "/resume <id>") {
		t.Errorf("a bare /resume says %q", m.status)
	}
	if m.session != nil {
		t.Error("a bare /resume opened a conversation anyway")
	}

	settle(m, ask(m, "/resume agent:support-42"))
	if m.conversationID != "agent:support-42" {
		t.Errorf("the conversation opened is %q", m.conversationID)
	}
}

func TestOpeningAnotherConversationWaitsForThisAnswer(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})
	attach(t, m, backend.session(t, ""), stream.ConversationPage{})
	first := m.session

	m.busy = true
	for _, command := range []string{"/new", "/resume agent:support-42"} {
		settle(m, ask(m, command))
		if !strings.Contains(m.status, "cancel") {
			t.Errorf("%s while busy says %q", command, m.status)
		}
		if m.session != first {
			t.Errorf("%s replaced the session mid-answer", command)
		}
	}
}

func TestOlderIsHonestAboutWhatItCanRead(t *testing.T) {
	backend := newRouter(t)
	m := newModel(t, Options{Open: backend.opener(t)})

	// A conversation that is not persisted has no saved history to page through.
	settle(m, ask(m, "/older"))
	if !strings.Contains(m.status, "not available") {
		t.Errorf("/older without history says %q", m.status)
	}

	m.options.History = BackendHistory(stream.Backend{URL: backend.URL, CustomerID: "acme"}, "jean")
	settle(m, ask(m, "/older"))
	if !strings.Contains(m.status, "nothing before this") {
		t.Errorf("/older at the start of a conversation says %q", m.status)
	}

	m.conversationID, m.before = "agent:support-1", "cursor-2"
	backend.page("cursor-2", stream.ConversationPage{
		Messages: []stream.ConversationMessage{{ID: "old", Role: "user", Text: "an earlier question"}},
		Before:   "cursor-3",
	})
	m.messages = []stream.ConversationMessage{{ID: "q", Role: "user", Text: "the newest question"}}
	settle(m, ask(m, "/older"))

	if !strings.Contains(m.status, "Loaded 1 earlier") {
		t.Errorf("/older says %q", m.status)
	}
	if len(m.messages) != 2 || m.messages[0].ID != "old" {
		t.Errorf("the earlier page landed as %+v", m.messages)
	}
	if m.before != "cursor-3" {
		t.Errorf("the next cursor is %q", m.before)
	}
	if cursors := backend.cursors; len(cursors) == 0 || cursors[len(cursors)-1] != "cursor-2" {
		t.Errorf("the backend was asked for %v", cursors)
	}

	// The end of the history is the end of the paging.
	backend.page("cursor-3", stream.ConversationPage{})
	settle(m, ask(m, "/older"))
	if m.before != "" {
		t.Errorf("there is nothing left to read, but the cursor is %q", m.before)
	}
	if !strings.Contains(m.status, "nothing before this") {
		t.Errorf("the end of the history says %q", m.status)
	}
}

func TestAnApplicationsOwnCommandRunsOffTheEventLoop(t *testing.T) {
	asked := make(chan []string, 1)
	m := newModel(t, Options{Commands: []Command{{
		Name: "ticket",
		Run: func(_ context.Context, args []string) (string, error) {
			asked <- args
			return "Opened ticket " + args[0], nil
		},
	}}})

	settle(m, ask(m, "/ticket 4821"))
	select {
	case args := <-asked:
		if len(args) != 1 || args[0] != "4821" {
			t.Errorf("the command was given %v", args)
		}
	default:
		t.Fatal("the command never ran")
	}
	if m.status != "Opened ticket 4821" {
		t.Errorf("the status line says %q", m.status)
	}
}

func TestAFailingCommandSaysWhatWentWrong(t *testing.T) {
	m := newModel(t, Options{Commands: []Command{{
		Name: "ticket",
		Run:  func(context.Context, []string) (string, error) { return "", errors.New("the tracker is down") },
	}}})

	settle(m, ask(m, "/ticket 4821"))
	if !m.statusFailed || m.status != "the tracker is down" {
		t.Errorf("the status line says %q, failed %v", m.status, m.statusFailed)
	}
}
