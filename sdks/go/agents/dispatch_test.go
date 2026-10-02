package agents

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// worked is a stand-in for the whole router a worker talks to: the dispatch socket work
// arrives on, the endpoint that opens a session, and the socket that session is held over.
// All three are real, so what is under test is a worker answering a message end to end.
type worked struct {
	*httptest.Server

	mu     sync.Mutex
	opened []acceleration.CreateSessionRequest
	asked  []string
	// answered is each command answered on a session someone else holds, as who it was
	// answered for and the command, which is what a worker handed one has to send back.
	answered []string
	sessions map[string]chan stream.Frame
	// sockets are the open session sockets, by session id, which answers are written to.
	sockets map[string]*socket

	// hand is what the router pushes down the dispatch socket once a worker is waiting.
	hand func(*websocket.Conn)
	// hold is what the router does with a session once it is open. Nil answers every
	// question with one piece of text.
	hold func(*testing.T, *socket, string)
}

func newWorked(t *testing.T, hand func(*websocket.Conn)) *worked {
	t.Helper()

	router := &worked{hand: hand, sessions: map[string]chan stream.Frame{}, sockets: map[string]*socket{}}
	var opened atomic.Int64
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/agents/sessions", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.CreateSessionRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		router.opened = append(router.opened, request)
		router.mu.Unlock()

		id := "session-" + string(rune('a'+opened.Add(1)-1))
		reply(w, http.StatusCreated, acceleration.Session{
			Id: id, AgentId: "agent-1", State: "running", CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("POST /v1/agents/sessions/{id}/responses", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.CreateResponseRequest
		_ = json.NewDecoder(r.Body).Decode(&request)

		router.mu.Lock()
		router.asked = append(router.asked, r.PathValue("id")+": "+request.Text)
		if request.CommandId != nil {
			router.answered = append(router.answered, r.Header.Get(stream.UserHeader)+": "+*request.CommandId)
		}
		router.mu.Unlock()
		// The answer arrives on the session's socket after the question is taken, as it does
		// from the backend. A command on a session somebody else holds is answered there.
		if request.CommandId == nil {
			go router.respond(t, r.PathValue("id"), request.Text)
		}

		reply(w, http.StatusAccepted, acceleration.AgentResponse{
			Id: "response-1", SessionId: r.PathValue("id"), Said: &request.Text,
			Status: "running", CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		router.answer(t, connection, r.PathValue("id"))
	})

	mux.HandleFunc("GET /v1/dispatch", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()

		_ = connection.WriteJSON(stream.Frame{"type": "ready", "worker_id": "worker-1"})
		if router.hand != nil {
			router.hand(connection)
		}
		for {
			var frame stream.Frame
			if err := connection.ReadJSON(&frame); err != nil {
				return
			}
		}
	})

	router.Server = httptest.NewServer(mux)
	t.Cleanup(router.Close)
	return router
}

// socket is one session's socket, written to by one answer at a time.
type socket struct {
	mu         sync.Mutex
	connection *websocket.Conn
}

func (s *socket) WriteJSON(frame any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connection.WriteJSON(frame)
}

// answer holds one session's socket, collecting the tool results sent back on it.
func (w *worked) answer(t *testing.T, connection *websocket.Conn, id string) {
	replies := make(chan stream.Frame, 8)
	w.mu.Lock()
	w.sessions[id] = replies
	w.sockets[id] = &socket{connection: connection}
	w.mu.Unlock()

	for {
		var frame stream.Frame
		if err := connection.ReadJSON(&frame); err != nil {
			return
		}
		switch frame.Type() {
		case "tool_result":
			select {
			case replies <- frame:
			default:
			}
		}
	}
}

// respond answers one question on its session's socket the way the backend would.
func (w *worked) respond(t *testing.T, id, text string) {
	var held *socket
	for held == nil {
		w.mu.Lock()
		held = w.sockets[id]
		hold := w.hold
		w.mu.Unlock()
		if held == nil {
			time.Sleep(time.Millisecond)
			continue
		}
		if hold != nil {
			hold(t, held, text)
			return
		}
	}
	_ = held.WriteJSON(stream.Frame{"type": "response_delta", "text": "answering " + text})
	_ = held.WriteJSON(stream.Frame{"type": "responded", "text": "answering " + text})
}

// questions is what the router was asked, in the order it was asked.
func (w *worked) questions() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.asked...)
}

// results is where one session's tool results arrive.
func (w *worked) results(t *testing.T, id string) chan stream.Frame {
	t.Helper()

	deadline := time.After(3 * time.Second)
	for {
		w.mu.Lock()
		found, open := w.sessions[id]
		w.mu.Unlock()
		if open {
			return found
		}
		select {
		case <-deadline:
			t.Fatalf("no session %s was ever held", id)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// answering is a worker waiting on this router, answering with an agent built by the
// factory. The number of agents built is returned, since reusing one is most of what a
// conversation is.
func answering(t *testing.T, router *worked, prepare func(*stream.Pipeline)) (*Dispatch, *atomic.Int64) {
	t.Helper()

	backend := stream.Backend{URL: router.URL, CustomerID: "acme"}
	dispatch, err := NewDispatch(DispatchOptions{Backend: backend, Logger: slog.New(slog.DiscardHandler)})
	if err != nil {
		t.Fatal(err)
	}

	var built atomic.Int64
	dispatch.OnMessage(func(ctx context.Context, message InboundMessage) error {
		conversation, err := dispatch.Conversation(ctx, message, func(context.Context, InboundMessage) (*Agent, error) {
			built.Add(1)
			llm := stream.Accelerated(stream.Config{Backend: backend, Logger: slog.New(slog.DiscardHandler)})
			if prepare != nil {
				prepare(llm)
			}
			return New(Options{Name: "Support", LLM: llm, Instructions: "help", Logger: slog.New(slog.DiscardHandler)})
		})
		if err != nil {
			return err
		}
		return conversation.Respond(message.Text)
	})
	return dispatch, &built
}

// waited runs the worker in the background and returns a function that stops it.
func waited(t *testing.T, dispatch *Dispatch) func() {
	t.Helper()

	ctx, stop := context.WithCancel(t.Context())
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		_ = dispatch.Run(ctx)
	}()
	return func() {
		stop()
		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			t.Fatal("the worker never stopped waiting")
		}
	}
}

// written is a message frame the router hands to a worker.
func written(channel, text string) stream.Frame {
	return stream.Frame{
		"type": "message", "channel_type": "agent", "channel_id": channel,
		"agent_id": channel, "config_id": "config-1", "text": text, "user_id": "sam",
	}
}

// eventually waits for a condition, so an assertion does not race the goroutines under it.
func eventually(t *testing.T, why string, settled func() bool) {
	t.Helper()

	deadline := time.After(3 * time.Second)
	for !settled() {
		select {
		case <-deadline:
			t.Fatal(why)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

func TestASessionIsOpenedOnTheChannelTheQuestionWasAskedIn(t *testing.T) {
	// The agent id is what names the channel replies are written into, and it is also how
	// the router finds this session when the next message arrives. Left as the agent's own
	// user id, every conversation this worker holds would be the same one.
	router := newWorked(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "hello"))
	})
	dispatch, _ := answering(t, router, nil)
	defer waited(t, dispatch)()

	eventually(t, "no session was ever opened", func() bool {
		router.mu.Lock()
		defer router.mu.Unlock()
		return len(router.opened) == 1
	})

	router.mu.Lock()
	defer router.mu.Unlock()
	opened := router.opened[0]
	if opened.AgentId == nil || *opened.AgentId != "support-42" {
		t.Errorf("the session was opened for agent %v, want support-42", opened.AgentId)
	}
	if opened.Incognito != nil && *opened.Incognito {
		t.Error("the session is incognito; the answer would be written nowhere the person can read it")
	}
}

func TestTheSecondMessageOnAChannelGoesToTheAgentAlreadyAnsweringThere(t *testing.T) {
	// That agent is still open and knows what has been said. Building another would answer
	// the follow-up as though the conversation had not happened.
	router := newWorked(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "first"))
		_ = connection.WriteJSON(written("support-42", "second"))
	})
	dispatch, built := answering(t, router, nil)
	defer waited(t, dispatch)()

	eventually(t, "the second question was never asked", func() bool { return len(router.questions()) == 2 })

	if built.Load() != 1 {
		t.Errorf("%d agents were built for one channel, want 1", built.Load())
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if len(router.opened) != 1 {
		t.Errorf("%d sessions were opened for one channel, want 1", len(router.opened))
	}
}

func TestAMessageOnAnotherChannelGetsAnAgentOfItsOwn(t *testing.T) {
	router := newWorked(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "hello"))
		_ = connection.WriteJSON(written("support-43", "hello"))
	})
	dispatch, built := answering(t, router, nil)
	defer waited(t, dispatch)()

	eventually(t, "both channels were never answered", func() bool {
		router.mu.Lock()
		defer router.mu.Unlock()
		return len(router.opened) == 2
	})

	if built.Load() != 2 {
		t.Errorf("%d agents were built for two channels, want 2", built.Load())
	}
}

func TestASecondQuestionWaitsForTheAnswerToTheFirst(t *testing.T) {
	// Two messages written in quick succession must not be answered on top of each other:
	// the person would watch two answers interleave in one channel.
	router := newWorked(t, nil)
	started := make(chan struct{})
	release := make(chan struct{})
	var overlapped atomic.Bool
	router.hold = func(_ *testing.T, connection *socket, text string) {
		if text == "first" {
			close(started)
			<-release
		} else if !isClosed(release) {
			overlapped.Store(true)
		}
		_ = connection.WriteJSON(stream.Frame{"type": "responded", "text": "answered " + text})
	}
	router.hand = func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "first"))
		<-started
		_ = connection.WriteJSON(written("support-42", "second"))
	}

	dispatch, _ := answering(t, router, nil)
	defer waited(t, dispatch)()

	<-started
	// Long enough that a second question on its way would have arrived.
	time.Sleep(150 * time.Millisecond)
	if asked := router.questions(); len(asked) != 1 {
		t.Fatalf("the router was asked %v while the first answer was still being written", asked)
	}

	close(release)
	eventually(t, "the second question was never asked", func() bool { return len(router.questions()) == 2 })
	if overlapped.Load() {
		t.Error("the second question was put while the first was still being answered")
	}
}

func TestAConversationKeepsRunningFunctionsAfterALongAnswer(t *testing.T) {
	// The session's events are a buffered channel the pipeline's socket reader writes
	// into, and that reader is also what delivers tool calls. A worker that answers a
	// message and walks away fills the buffer partway through the first long answer, and
	// from then on the model's tool calls are never read: local function calling stops,
	// silently, in the middle of a conversation.
	router := newWorked(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "how many?"))
	})
	router.hold = func(_ *testing.T, connection *socket, _ string) {
		for range 200 {
			_ = connection.WriteJSON(stream.Frame{"type": "response_delta", "text": "."})
		}
		_ = connection.WriteJSON(stream.Frame{
			"type": "tool_call", "id": "call-1", "name": "count", "arguments": `{}`,
		})
	}

	var ran atomic.Bool
	dispatch, _ := answering(t, router, func(llm *stream.Pipeline) {
		_ = llm.Functions().Add(count{ran: &ran})
	})
	defer waited(t, dispatch)()

	eventually(t, "the model's tool call was never read, so the function never ran", ran.Load)

	result := <-router.results(t, "session-a")
	if result.String("output") != "42" {
		t.Errorf("the model was answered %q, want 42", result.String("output"))
	}
}

func TestWhatTheBackendSaysAboutAConversationReachesTheWatcher(t *testing.T) {
	// The conversation reads its own session, so this is the only way to see any of it. A
	// worker with no way to watch can report that it answered and nothing about how: not
	// which tools ran, not how long they took, not that the model failed.
	router := newWorked(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "hello"))
	})

	var mu sync.Mutex
	var watched []string
	var channels []string
	backend := stream.Backend{URL: router.URL, CustomerID: "acme"}
	dispatch, err := NewDispatch(DispatchOptions{
		Backend: backend,
		Logger:  slog.New(slog.DiscardHandler),
		OnEvent: func(channel string, event stream.Event) {
			mu.Lock()
			defer mu.Unlock()
			watched = append(watched, event.Kind)
			channels = append(channels, channel)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	dispatch.OnMessage(func(ctx context.Context, message InboundMessage) error {
		conversation, err := dispatch.Conversation(ctx, message, func(context.Context, InboundMessage) (*Agent, error) {
			llm := stream.Accelerated(stream.Config{Backend: backend, Logger: slog.New(slog.DiscardHandler)})
			return New(Options{Name: "Support", LLM: llm, Instructions: "help", Logger: slog.New(slog.DiscardHandler)})
		})
		if err != nil {
			return err
		}
		return conversation.Respond(message.Text)
	})
	defer waited(t, dispatch)()

	eventually(t, "the answer was never watched", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return slices.Contains(watched, "responded")
	})

	mu.Lock()
	defer mu.Unlock()
	if !slices.Contains(watched, "response_delta") {
		t.Errorf("the watcher saw %v, and none of it is the answer being written", watched)
	}
	for _, channel := range channels {
		if channel != "support-42" {
			t.Errorf("an event was watched on %q; a worker holding two conversations could not tell them apart", channel)
		}
	}
}

func TestAnAnswerIsGivenAsLongAsTheWorkerAskedFor(t *testing.T) {
	// Five minutes suits an agent whose tools answer in seconds. One whose tools read a
	// source tree needs longer, and a turn abandoned underneath it is an answer the
	// person watched stop halfway.
	router := newWorked(t, nil)
	router.hold = func(_ *testing.T, connection *socket, text string) {
		if text == "slow" {
			// Neither responded nor error, which is the only case the timeout is
			// reached in: a turn the backend never ends.
			return
		}
		_ = connection.WriteJSON(stream.Frame{"type": "responded", "text": "answered " + text})
	}
	router.hand = func(connection *websocket.Conn) {
		_ = connection.WriteJSON(written("support-42", "slow"))
		_ = connection.WriteJSON(written("support-42", "next"))
	}

	backend := stream.Backend{URL: router.URL, CustomerID: "acme"}
	dispatch, err := NewDispatch(DispatchOptions{
		Backend:     backend,
		Logger:      slog.New(slog.DiscardHandler),
		TurnTimeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	dispatch.OnMessage(func(ctx context.Context, message InboundMessage) error {
		conversation, err := dispatch.Conversation(ctx, message, func(context.Context, InboundMessage) (*Agent, error) {
			llm := stream.Accelerated(stream.Config{Backend: backend, Logger: slog.New(slog.DiscardHandler)})
			return New(Options{Name: "Support", LLM: llm, Instructions: "help", Logger: slog.New(slog.DiscardHandler)})
		})
		if err != nil {
			return err
		}
		return conversation.Respond(message.Text)
	})
	defer waited(t, dispatch)()

	// The first answer never settles, so only a turn timeout this short lets the second
	// through. Under the five-minute default this conversation is stuck.
	eventually(t, "the turn was never abandoned, so the next question was never asked", func() bool {
		return len(router.questions()) == 2
	})
}

func TestATurnCannotBeGivenLessThanNoTimeToFinish(t *testing.T) {
	_, err := NewDispatch(DispatchOptions{TurnTimeout: -time.Second})
	if err == nil {
		t.Fatal("a negative turn timeout was accepted; every answer would be abandoned before it started")
	}
}

func TestAMessageWithNoChannelIsRefusedRatherThanAnsweredSomewhere(t *testing.T) {
	router := newWorked(t, nil)
	dispatch, _ := answering(t, router, nil)

	_, err := dispatch.Conversation(t.Context(), InboundMessage{Text: "hello"}, func(context.Context, InboundMessage) (*Agent, error) {
		t.Fatal("an agent was built for a message with nowhere to answer")
		return nil, nil
	})
	if err == nil {
		t.Fatal("a message with no channel was accepted")
	}
}

func TestAMessageWrittenToARunningSessionIsAnsweredThereForWhoeverWroteIt(t *testing.T) {
	// The session is the person's own, and only a request acting for them reaches it. The
	// command is what their screen is showing as being answered, so it is what the answer
	// has to land on.
	router := newWorked(t, nil)
	dispatch, built := answering(t, router, nil)

	err := dispatch.Answer(t.Context(), InboundMessage{
		AgentID: "support-42", SessionID: "session-9", CommandID: "command-1", Text: "Where is my order?", UserID: "sam",
	})
	if err != nil {
		t.Fatal(err)
	}

	router.mu.Lock()
	defer router.mu.Unlock()
	if !slices.Equal(router.asked, []string{"session-9: Where is my order?"}) {
		t.Errorf("the router was asked %v", router.asked)
	}
	if !slices.Equal(router.answered, []string{"sam: command-1"}) {
		t.Errorf("the command answered was %v, want command-1 for sam", router.answered)
	}
	if built.Load() != 0 || len(router.opened) != 0 {
		t.Error("an agent was started for a conversation a session is already holding")
	}
}

func TestAMessageARunningSessionHoldsIsNotGivenASecondAgent(t *testing.T) {
	router := newWorked(t, nil)
	dispatch, _ := answering(t, router, nil)

	_, err := dispatch.Conversation(t.Context(), InboundMessage{ChannelID: "support-42", SessionID: "session-9", Text: "hello"},
		func(context.Context, InboundMessage) (*Agent, error) {
			t.Fatal("an agent was built for a conversation a session is already holding")
			return nil, nil
		})
	if err == nil {
		t.Fatal("a message a session is holding was given a conversation of its own")
	}
}

func isClosed(done chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// count is a tool that says 42 and remembers it was asked.
type count struct {
	ran *atomic.Bool
}

func (count) Name() string        { return "count" }
func (count) Description() string { return "count things" }
func (c count) Run(context.Context) (any, error) {
	c.ran.Store(true)
	return 42, nil
}
