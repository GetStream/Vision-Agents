package stream

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
	"github.com/GetStream/Vision-Agents/agents-core-go/tools"
	"github.com/gorilla/websocket"
)

// router is a stand-in for the acceleration backend: the one endpoint that creates a
// session, and the socket it is then watched on. It is a real HTTP server with a real
// WebSocket upgrader, so what is under test is the whole exchange rather than a description
// of it.
type router struct {
	*httptest.Server

	mu       sync.Mutex
	requests []acceleration.CreateSessionRequest

	// configs are the stored agent configs a name is resolved against.
	configs []acceleration.AgentConfig

	// serve is what the socket does once a client is on it.
	serve func(t *testing.T, connection *websocket.Conn)
}

func newRouter(t *testing.T, serve func(*testing.T, *websocket.Conn)) *router {
	t.Helper()

	backend := &router{serve: serve}
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/agents/sessions", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.CreateSessionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		backend.mu.Lock()
		backend.requests = append(backend.requests, request)
		backend.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(acceleration.Session{
			Id: "session-1", AgentId: "agent-1", CallId: "call-1",
			CallType: "default", UserId: "jean", State: "running",
			CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("GET /v1/agents/configs", func(w http.ResponseWriter, _ *http.Request) {
		backend.mu.Lock()
		stored := backend.configs
		backend.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if stored == nil {
			stored = []acceleration.AgentConfig{}
		}
		_ = json.NewEncoder(w).Encode(stored)
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		if backend.serve != nil {
			backend.serve(t, connection)
		}
	})

	backend.Server = httptest.NewServer(mux)
	t.Cleanup(backend.Close)
	return backend
}

// created is the session request the router was sent.
func (r *router) created(t *testing.T) acceleration.CreateSessionRequest {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.requests) == 0 {
		t.Fatal("no session was created")
	}
	return r.requests[0]
}

func TestJoiningRendersTheAgentAsASessionToCreate(t *testing.T) {
	backend := newRouter(t, hold)
	pipeline := Accelerated(Config{
		LLM:     "gemma4",
		STT:     "parakeet",
		TTS:     "sonic",
		Agent:   "jean",
		Backend: Backend{URL: backend.URL, CustomerID: "acme"},
	})

	session, err := pipeline.Join(t.Context(), Call{
		ID: "call-1", UserID: "jean", Instructions: "Be brief.",
		Tags: map[string]string{"customer_id": "123"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	if session.Id != "session-1" {
		t.Errorf("the session is %q", session.Id)
	}

	request := backend.created(t)
	if *request.CallId != "call-1" || *request.Llm != "gemma4" || *request.ConfigId != "jean" {
		t.Errorf("the router was asked for %+v", request)
	}
	if (*request.Tags)["customer_id"] != "123" {
		t.Errorf("the cost labels did not travel: %v", request.Tags)
	}
}

func TestAStoredConfigCanBeNamedRatherThanIdentified(t *testing.T) {
	backend := newRouter(t, hold)
	backend.configs = []acceleration.AgentConfig{{Id: "config-7", Name: "jean"}}

	pipeline := Accelerated(Config{
		Agent:   "jean",
		Backend: Backend{URL: backend.URL, CustomerID: "acme"},
	})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	// The backend looks a config up by id, so the name has to have become one on the way.
	if request := backend.created(t); *request.ConfigId != "config-7" {
		t.Errorf("the session was created against %q", *request.ConfigId)
	}
}

func TestAConfigNameNothingMatchesIsPassedThroughAsAnID(t *testing.T) {
	backend := newRouter(t, hold)

	pipeline := Accelerated(Config{
		Agent:   "config-7",
		Backend: Backend{URL: backend.URL, CustomerID: "acme"},
	})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	if request := backend.created(t); *request.ConfigId != "config-7" {
		t.Errorf("the session was created against %q", *request.ConfigId)
	}
}

func TestACallWithNoIDIsHeldInWriting(t *testing.T) {
	backend := newRouter(t, hold)
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})

	if _, err := pipeline.Join(t.Context(), Call{UserID: "jean"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	request := backend.created(t)
	if request.Text == nil || !*request.Text {
		t.Error("a session with no call to join has to be a text one")
	}
	if request.CallId != nil {
		t.Errorf("there is no call, but the router was sent %q", *request.CallId)
	}
}

func TestTheModelIsOfferedEveryRegisteredFunction(t *testing.T) {
	backend := newRouter(t, hold)
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})

	err := tools.Register(pipeline.Functions(), "get_weather", "Get current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state"`
		}) (any, error) {
			return "sunny in " + in.Location, nil
		})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	request := backend.created(t)
	if request.Tools == nil || len(*request.Tools) != 1 {
		t.Fatalf("the model was offered %v", request.Tools)
	}
	offered := (*request.Tools)[0]
	if offered.Name != "get_weather" || offered.Parameters == nil {
		t.Fatalf("the tool went over as %+v", offered)
	}
	properties := (*offered.Parameters)["properties"].(map[string]any)
	if _, ok := properties["location"]; !ok {
		t.Errorf("the model was not told what to fill in: %v", properties)
	}
}

func TestAToolCallIsRunHereAndAnsweredOverTheSocket(t *testing.T) {
	answered := make(chan map[string]any, 1)
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-42", "name": "get_weather",
			"arguments": `{"location":"Boulder, CO"}`,
		}); err != nil {
			return
		}
		var frame map[string]any
		if err := connection.ReadJSON(&frame); err != nil {
			return
		}
		answered <- frame
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	err := tools.Register(pipeline.Functions(), "get_weather", "Get current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location"`
		}) (any, error) {
			return "sunny in " + in.Location, nil
		})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	select {
	case frame := <-answered:
		if frame["type"] != "tool_result" || frame["tool_call_id"] != "call-42" {
			t.Fatalf("the model was sent %v", frame)
		}
		if frame["output"] != "sunny in Boulder, CO" {
			t.Errorf("the model was told %v", frame["output"])
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the model is still waiting on the tool")
	}
}

func TestAToolThatFailsSaysSoRatherThanLeavingTheModelWaiting(t *testing.T) {
	answered := make(chan map[string]any, 1)
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-42", "name": "nothing", "arguments": "{}",
		}); err != nil {
			return
		}
		var frame map[string]any
		if err := connection.ReadJSON(&frame); err != nil {
			return
		}
		answered <- frame
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	select {
	case frame := <-answered:
		if frame["error"] == nil {
			t.Fatalf("the model was sent %v, and would wait forever for a result", frame)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the model is still waiting on the tool")
	}
}

func TestWhatTheBackendDidArrivesAsEvents(t *testing.T) {
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		_ = connection.WriteJSON(map[string]any{
			"type": "heard", "text": "what is the weather",
			"participant": map[string]any{"id": "p1", "user_id": "caller", "name": "Ada"},
		})
		_ = connection.WriteJSON(map[string]any{"type": "responded", "text": "It is sunny."})
		hold(t, connection)
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	heard := next(t, pipeline.Events())
	if heard.Kind != "heard" || heard.Text != "what is the weather" {
		t.Fatalf("the first event is %+v", heard)
	}
	if heard.Participant.Name != "Ada" {
		t.Errorf("who spoke did not travel: %+v", heard.Participant)
	}

	answered := next(t, pipeline.Events())
	if answered.Kind != "responded" || answered.Text != "It is sunny." {
		t.Errorf("the second event is %+v", answered)
	}
}

func TestSayingSomethingSendsItDownTheSocket(t *testing.T) {
	sent := make(chan map[string]any, 4)
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		for {
			var frame map[string]any
			if err := connection.ReadJSON(&frame); err != nil {
				return
			}
			sent <- frame
		}
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	if err := pipeline.Say("we are closing in five minutes", false); err != nil {
		t.Fatal(err)
	}
	if err := pipeline.Respond("answer them", true); err != nil {
		t.Fatal(err)
	}

	want := []string{"say", "interrupt", "respond"}
	for _, kind := range want {
		select {
		case frame := <-sent:
			if frame["type"] != kind {
				t.Fatalf("the router was sent %v, want %s", frame, kind)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("the router never got %s", kind)
		}
	}
}

func TestActingOnACallNobodyIsOnIsRefused(t *testing.T) {
	pipeline := Accelerated(Config{Backend: Backend{URL: "http://localhost:1", CustomerID: "acme"}})

	if err := pipeline.Say("hello", false); err == nil {
		t.Fatal("there is nowhere for that to be said")
	}
}

func TestTheEventsChannelClosesWhenTheConversationEnds(t *testing.T) {
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		_ = connection.WriteJSON(map[string]any{"type": "left"})
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	if _, err := pipeline.Join(t.Context(), Call{ID: "call-1"}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	events := pipeline.Events()
	if left := next(t, events); left.Kind != "left" {
		t.Fatalf("the last event is %+v", left)
	}
	select {
	case _, open := <-events:
		if open {
			t.Fatal("something arrived after the call ended")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the channel is still open with nothing on the other end")
	}
}

// hold keeps a socket open until the client goes away, which is what a call with nothing
// happening on it looks like.
func hold(_ *testing.T, connection *websocket.Conn) {
	for {
		if _, _, err := connection.ReadMessage(); err != nil {
			return
		}
	}
}

// next reads one event, failing rather than hanging when none arrives.
func next(t *testing.T, events <-chan Event) Event {
	t.Helper()
	select {
	case event := <-events:
		return event
	case <-time.After(5 * time.Second):
		t.Fatal("nothing arrived on the events channel")
		return Event{}
	}
}
