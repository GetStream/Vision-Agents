package tui

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/gorilla/websocket"
)

// TestAWholeConversationAgainstARealSession drives the conversation end to end: it opens
// a real session on a real socket, reads the history the conversation starts from, asks a
// question, watches the answer being written with the tools that found it, and leaves.
func TestAWholeConversationAgainstARealSession(t *testing.T) {
	backend := newRouter(t)
	started := time.Now().Add(-24 * time.Second)
	finished := started.Add(23400 * time.Millisecond)
	backend.announce = func(connection *websocket.Conn) {
		_ = connection.WriteJSON(update(stream.ConversationMessage{
			ID: "q1", Role: "user", Text: "what does useChatContext return?",
		}))
		_ = connection.WriteJSON(update(stream.ConversationMessage{
			ID: "a1", QuestionID: "q1", Role: "assistant", State: "tools", StartedAt: started,
			Tools: []stream.ToolActivity{{
				ID: "t1", Name: "search_docs", Title: "Search the documentation",
				Product: "chat", SDK: "react", Status: "running", Phase: "executing",
				StartedAt: started,
			}},
		}))
		_ = connection.WriteJSON(update(stream.ConversationMessage{
			ID: "a1", QuestionID: "q1", Role: "assistant", State: "completed",
			StartedAt: started, FinishedAt: &finished, Saved: true,
			Text: "`useChatContext` returns a **ChatContextValue**.",
			Tools: []stream.ToolActivity{{
				ID: "t1", Name: "search_docs", Title: "Search the documentation",
				Product: "chat", SDK: "react", Status: "completed", Summary: "2 pages",
				StartedAt: started, FinishedAt: &finished,
			}},
		}))
	}
	backend.page("", stream.ConversationPage{
		Messages: []stream.ConversationMessage{{ID: "q0", Role: "user", Text: "a question from last time"}},
		Before:   "cursor-2",
	})

	opened := make(chan string, 1)
	m := newModel(t, Options{
		Open:           backend.opener(t),
		History:        BackendHistory(stream.Backend{URL: backend.URL, CustomerID: "acme"}, "jean"),
		ConversationID: "agent:support-1",
		Branding:       Branding{Title: "Jean", Subtitle: "answers questions about Stream"},
		OnOpen: func(_ context.Context, session Session) error {
			opened <- session.ID()
			return nil
		},
	})

	settle(m, m.Init())

	// The conversation opened on what it was asked to, with the history behind it.
	if m.session == nil || m.conversationID != "agent:support-1" {
		t.Fatalf("the conversation opened on %q with session %v", m.conversationID, m.session != nil)
	}
	if !shown(m, "a question from last time") {
		t.Errorf("the saved history is not on screen:\n%s", plain(m.transcript()))
	}
	if m.before != "cursor-2" {
		t.Errorf("there is more saved history, but the cursor is %q", m.before)
	}
	select {
	case id := <-opened:
		if id != "session-1" {
			t.Errorf("the application was told about session %q", id)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the application was never told the conversation opened")
	}

	// Everything the session had to say arrived and was drawn.
	if !shown(m, "what does useChatContext return?") {
		t.Errorf("the question did not arrive:\n%s", plain(m.transcript()))
	}
	if !shown(m, "✓ Search the documentation 23.4s 2 pages") {
		t.Errorf("the tool did not arrive:\n%s", plain(m.transcript()))
	}
	// The words are asserted rather than the styling, which is whatever the terminal
	// running the test can express.
	if !shown(m, "useChatContext") || !strings.Contains(plain(m.transcript()), "ChatContextValue") {
		t.Errorf("the answer did not arrive:\n%s", plain(m.transcript()))
	}
	if !shown(m, "Completed · 23.4s · saved") {
		t.Errorf("how the answer ended is not on screen:\n%s", plain(m.transcript()))
	}
	// The answer replaced the half-written one rather than being added beneath it.
	if len(m.messages) != 3 {
		t.Errorf("the conversation holds %d messages: %+v", len(m.messages), m.messages)
	}
	if m.busy {
		t.Error("the conversation is still waiting on a finished answer")
	}
	// What the tools worked in reached the header.
	if m.scope != "chat / react" {
		t.Errorf("the scope is %q", m.scope)
	}
	if header := plain(m.header()); !strings.Contains(header, "chat / react") || !strings.Contains(header, "Jean") {
		t.Errorf("the header says:\n%s", header)
	}

	// And a question of our own goes down the same socket.
	settle(m, ask(m, "and what about useChannelStateContext?"))
	sent := backend.commands(t, "respond")
	last := sent[len(sent)-1]
	if last["text"] != "and what about useChannelStateContext?" {
		t.Errorf("the backend was asked %v", last)
	}
}

func TestLeavingClosesTheSession(t *testing.T) {
	backend := newRouter(t)
	closed := make(chan struct{})
	m := newModel(t, Options{Open: func(context.Context, string) (Session, error) {
		return leaving{Session: backend.session(t, ""), closed: closed}, nil
	}})

	settle(m, m.Init())
	m.close(m.session)
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was left open")
	}
}

func TestASessionThatWillNotCloseIsNotWaitedOnForever(t *testing.T) {
	// The conversation's own context is already over, which is what leaving on Ctrl-C
	// looks like, and the session still has to be given a chance to end.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m, err := New(ctx, Options{
		Open:   func(context.Context, string) (Session, error) { return nil, nil },
		Logger: slog.New(slog.DiscardHandler),
	})
	if err != nil {
		t.Fatal(err)
	}

	deadlines := make(chan bool, 1)
	m.close(stubborn{deadlines: deadlines})
	if _, ok := <-deadlines; !ok {
		t.Fatal("the session was never asked to close")
	} else if !ok {
		t.Error("the session was given a context with no deadline on it")
	}
}

// leaving reports being closed.
type leaving struct {
	Session
	closed chan struct{}
}

func (l leaving) Close(ctx context.Context) error {
	defer close(l.closed)
	return l.Session.Close(ctx)
}

// stubborn refuses to close, and says whether it was given a deadline to do it by.
type stubborn struct {
	Session
	deadlines chan bool
}

func (s stubborn) Close(ctx context.Context) error {
	_, hasDeadline := ctx.Deadline()
	s.deadlines <- hasDeadline
	close(s.deadlines)
	return errors.New("the router is gone")
}
