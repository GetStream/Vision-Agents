package tui

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
	"github.com/gorilla/websocket"
)

// router is a stand-in for the acceleration backend: the endpoint that opens a session,
// the socket the session is then watched on, and the saved history a conversation is read
// back from. It is a real HTTP server with a real WebSocket upgrader, so what the
// conversation is tested against is the exchange rather than a description of it.
type router struct {
	*httptest.Server

	mu sync.Mutex
	// sent is every command the conversation put down the socket.
	sent []map[string]any
	// cursors is every history request, as the cursor it was made with.
	cursors []string

	// conversationID is what a new session is told it is saved under.
	conversationID string
	// truncated says the session was given less than the whole history.
	truncated bool
	// pages answers history requests, keyed by the cursor asked for.
	pages map[string]stream.ConversationPage
	// announce writes whatever the conversation should see as soon as it is watching.
	announce func(*websocket.Conn)
}

func newRouter(t *testing.T) *router {
	t.Helper()
	backend := &router{conversationID: "agent:support-1", pages: map[string]stream.ConversationPage{}}
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/agents/sessions", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.CreateSessionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		backend.mu.Lock()
		id, truncated := backend.conversationID, backend.truncated
		backend.mu.Unlock()
		// A resumed conversation keeps the id it was resumed under.
		if request.ConversationId != nil && *request.ConversationId != "" {
			id = *request.ConversationId
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(acceleration.Session{
			Id: "session-1", AgentId: "agent-1", UserId: "jean", State: "running",
			ConversationId: &id, ContextTruncated: &truncated, CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("GET /v1/agents/configs", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]acceleration.AgentConfig{})
	})

	mux.HandleFunc("GET /v1/agents/conversations/{cid}/messages", func(w http.ResponseWriter, r *http.Request) {
		before := r.URL.Query().Get("before")
		backend.mu.Lock()
		backend.cursors = append(backend.cursors, before)
		page := backend.pages[before]
		backend.mu.Unlock()
		if page.Messages == nil {
			page.Messages = []stream.ConversationMessage{}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(page)
	})

	mux.HandleFunc("DELETE /v1/agents/sessions/{id}", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		backend.mu.Lock()
		announce := backend.announce
		backend.mu.Unlock()
		if announce != nil {
			announce(connection)
		}
		for {
			var frame map[string]any
			if err := connection.ReadJSON(&frame); err != nil {
				return
			}
			backend.mu.Lock()
			backend.sent = append(backend.sent, frame)
			backend.mu.Unlock()
		}
	})

	backend.Server = httptest.NewServer(mux)
	t.Cleanup(backend.Close)
	return backend
}

// page is the history answered for a cursor. An empty cursor is the newest page.
func (r *router) page(before string, page stream.ConversationPage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pages[before] = page
}

// commands is what the conversation has told the backend to do so far.
func (r *router) commands(t *testing.T, want ...string) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		r.mu.Lock()
		sent := append([]map[string]any{}, r.sent...)
		r.mu.Unlock()
		kinds := make([]string, len(sent))
		for i, frame := range sent {
			kinds[i], _ = frame["type"].(string)
		}
		if len(want) == 0 || containsInOrder(kinds, want) {
			return sent
		}
		if time.Now().After(deadline) {
			t.Fatalf("the backend was sent %v, waiting for %v", kinds, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func containsInOrder(kinds, want []string) bool {
	for _, kind := range kinds {
		if len(want) > 0 && kind == want[0] {
			want = want[1:]
		}
	}
	return len(want) == 0
}

// session opens a real session on the stand-in router.
func (r *router) session(t *testing.T, conversationID string) Session {
	t.Helper()
	quiet := slog.New(slog.DiscardHandler)
	agent, err := agents.New(agents.Options{
		Name:   "jean",
		Logger: quiet,
		LLM: stream.Accelerated(stream.Config{
			Backend: stream.Backend{URL: r.URL, CustomerID: "acme"},
			Logger:  quiet,
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := agent.Chat(t.Context(), agents.ChatOptions{Persist: true, ConversationID: conversationID})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	return session
}

// opener opens a session on the stand-in router, for a conversation to be built around.
func (r *router) opener(t *testing.T) Opener {
	return func(_ context.Context, conversationID string) (Session, error) {
		return r.session(t, conversationID), nil
	}
}

// update is the message a conversation_updated frame carries.
func update(message stream.ConversationMessage) map[string]any {
	return map[string]any{"type": "conversation_updated", "message": message}
}

// newModel builds a conversation at a usable size, with an opener that refuses unless the
// test gave one, since most tests are about what is already on screen.
func newModel(t *testing.T, options Options) *Model {
	t.Helper()
	if options.Open == nil {
		options.Open = func(context.Context, string) (Session, error) {
			t.Error("the conversation opened a session it was not given")
			return nil, context.Canceled
		}
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.DiscardHandler)
	}
	m, err := New(t.Context(), options)
	if err != nil {
		t.Fatal(err)
	}
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	return m
}

// attach puts a live session behind the conversation, as opening one would.
func attach(t *testing.T, m *Model, session Session, page stream.ConversationPage) {
	t.Helper()
	m.Update(opened{generation: m.generation, session: session, page: page})
	if m.session == nil {
		t.Fatal("the session did not take")
	}
}

// send gives the conversation a key, the way the terminal would.
func send(m *Model, key string) tea.Cmd {
	var message tea.Msg
	switch key {
	case "enter":
		message = tea.KeyMsg{Type: tea.KeyEnter}
	case "alt+enter":
		message = tea.KeyMsg{Type: tea.KeyEnter, Alt: true}
	case "esc":
		message = tea.KeyMsg{Type: tea.KeyEsc}
	case "pgup":
		message = tea.KeyMsg{Type: tea.KeyPgUp}
	case "ctrl+c":
		message = tea.KeyMsg{Type: tea.KeyCtrlC}
	default:
		message = tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(key)}
	}
	_, cmd := m.Update(message)
	return cmd
}

// ask types a question or a command and submits it.
func ask(m *Model, text string) tea.Cmd {
	m.input.SetValue(text)
	return send(m, "enter")
}

// step runs a command off the event loop the way the program does, and returns what it
// produced. A command still waiting — the heartbeat, or the next thing a silent session
// will say — is abandoned rather than waited out.
func step(cmd tea.Cmd) []tea.Msg {
	if cmd == nil {
		return nil
	}
	produced := make(chan tea.Msg, 1)
	go func() { produced <- cmd() }()
	select {
	case message := <-produced:
		if batch, ok := message.(tea.BatchMsg); ok {
			var all []tea.Msg
			for _, one := range batch {
				all = append(all, step(one)...)
			}
			return all
		}
		return []tea.Msg{message}
	case <-time.After(300 * time.Millisecond):
		return nil
	}
}

// settle runs a command and feeds everything it produced back in, until the conversation
// has nothing left to do. Heartbeats are dropped, since a test that waits for the clock is
// slow for no reason.
func settle(m *Model, cmd tea.Cmd) {
	for _, message := range step(cmd) {
		if _, beat := message.(tick); beat {
			continue
		}
		_, next := m.Update(message)
		settle(m, next)
	}
}

// plain is what is on screen with the styling taken off, which is what a test can assert
// about without asserting a palette.
func plain(s string) string { return ansi.Strip(s) }

// screen is the whole interface as lines of plain text.
func screen(m *Model) []string { return strings.Split(plain(m.View()), "\n") }

// shown says whether the conversation currently shows some text, ignoring styling and the
// wrapping the terminal width imposed.
func shown(m *Model, text string) bool {
	return strings.Contains(strings.Join(strings.Fields(plain(m.transcript())), " "), text)
}
