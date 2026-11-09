package stream

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// restartable is a stand-in router whose events socket can be stopped and started again
// on the same address, the way a rollout replaces the process behind it. The nth socket
// opened is served by the nth handler, a nil one refusing it as a session the router does
// not have, and the query each was opened with is recorded.
type restartable struct {
	t       *testing.T
	address string
	serve   []func(*websocket.Conn)

	mu     sync.Mutex
	server *httptest.Server
	asked  []url.Values
}

func newRestartable(t *testing.T, serve ...func(*websocket.Conn)) *restartable {
	t.Helper()
	r := &restartable{t: t, serve: serve}
	r.start()
	t.Cleanup(r.stop)
	return r
}

func (r *restartable) start() {
	r.mu.Lock()
	defer r.mu.Unlock()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", func(w http.ResponseWriter, req *http.Request) {
		r.mu.Lock()
		r.asked = append(r.asked, req.URL.Query())
		var serve func(*websocket.Conn)
		if opened := len(r.asked); opened <= len(r.serve) {
			serve = r.serve[opened-1]
		}
		r.mu.Unlock()
		if serve == nil {
			http.Error(w, `{"error":{"message":"no such session","type":"not_found","code":"session_not_found"}}`, http.StatusNotFound)
			return
		}
		connection, err := (&websocket.Upgrader{}).Upgrade(w, req, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		serve(connection)
	})

	server := httptest.NewUnstartedServer(mux)
	if r.address != "" {
		listener, err := net.Listen("tcp", r.address)
		if err != nil {
			r.t.Fatal(err)
		}
		server.Listener = listener
	}
	server.Start()
	r.address = server.Listener.Addr().String()
	r.server = server
}

// restart stops the router and starts the next one on the same address, a while later.
func (r *restartable) restart() {
	r.stop()
	time.Sleep(50 * time.Millisecond)
	r.start()
}

func (r *restartable) stop() {
	r.mu.Lock()
	server := r.server
	r.server = nil
	r.mu.Unlock()
	if server != nil {
		server.CloseClientConnections()
		server.Close()
	}
}

func (r *restartable) url() string { return "http://" + r.address }

// queries are what each socket was opened with, in order.
func (r *restartable) queries() []url.Values {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]url.Values(nil), r.asked...)
}

// closes ends a socket the way the router does, with code and reason.
func closes(connection *websocket.Conn, code int, reason string) {
	_ = connection.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(code, reason))
	// Wait for the far side to close, so the frame is read rather than reset.
	_ = connection.SetReadDeadline(time.Now().Add(time.Second))
	for {
		if _, _, err := connection.ReadMessage(); err != nil {
			return
		}
	}
}

// watching opens a pipeline onto the session, reconnecting fast enough for a test.
func watching(t *testing.T, router *restartable, session *acceleration.Session) *Pipeline {
	t.Helper()
	pipeline := Accelerated(Config{Backend: Backend{URL: router.url(), CustomerID: "acme"}})
	pipeline.firstRetry = 10 * time.Millisecond
	if err := pipeline.Watch(t.Context(), session); err != nil {
		t.Fatal(err)
	}
	return pipeline
}

// collect reads events until the channel closes.
func collect(t *testing.T, events <-chan Event) []Event {
	t.Helper()
	var got []Event
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-events:
			if !open {
				return got
			}
			got = append(got, event)
		case <-deadline:
			t.Fatalf("the events never ended; got %d", len(got))
		}
	}
}

func kinds(events []Event) []string {
	var named []string
	for _, event := range events {
		named = append(named, event.Kind)
	}
	return named
}

func TestATextSessionCarriesOnThroughARouterRestart(t *testing.T) {
	var router *restartable
	restarted := make(chan struct{})
	router = newRestartable(t,
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
			_ = connection.WriteJSON(Frame{"type": "responding", "turn_id": "turn-1", "cursor": "first.1"})
			closes(connection, websocket.CloseServiceRestart, "router restarting")
			go func() { router.restart(); close(restarted) }()
		},
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "second.0", "complete": false,
				"command": Frame{"request_id": "question-1", "state": "interrupted"}})
			_ = connection.WriteJSON(Frame{"type": "responded", "turn_id": "turn-2", "text": "Asked again.", "cursor": "second.1"})
			closes(connection, websocket.CloseNormalClosure, "the session ended")
		},
	)
	pipeline := watching(t, router, &acceleration.Session{Id: "session-1"})
	pipeline.Awaiting("question-1")

	got := collect(t, pipeline.Events())
	<-restarted

	if want := []string{"responding", "resumed", "responded"}; !equalStrings(kinds(got), want) {
		t.Fatalf("the events were %v, want %v: one stream across the restart", kinds(got), want)
	}
	resumed := got[1].Resumed
	if resumed == nil || resumed.Complete || resumed.Command == nil || resumed.Command.State != "interrupted" {
		t.Errorf("the reconnect was told %+v", resumed)
	}
	asked := router.queries()
	if len(asked) != 2 {
		t.Fatalf("the socket was opened %d times", len(asked))
	}
	if asked[0].Get("resumable") != "true" {
		t.Errorf("a text session's socket did not ask to be resumable: %v", asked[0])
	}
	if asked[1].Get("cursor") != "first.1" || asked[1].Get("request_id") != "question-1" {
		t.Errorf("the reconnect asked %v, want the last cursor and the question waited on", asked[1])
	}
}

func TestASocketDroppedWithoutAWordIsReconnectedToo(t *testing.T) {
	router := newRestartable(t,
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
			_ = connection.WriteJSON(Frame{"type": "responding", "cursor": "first.1"})
			// No close frame: the connection is cut.
		},
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.1", "complete": true})
			_ = connection.WriteJSON(Frame{"type": "responded", "text": "Done.", "cursor": "first.2"})
			closes(connection, websocket.CloseNormalClosure, "the session ended")
		},
	)
	pipeline := watching(t, router, &acceleration.Session{Id: "session-1"})

	got := collect(t, pipeline.Events())

	if want := []string{"responding", "resumed", "responded"}; !equalStrings(kinds(got), want) {
		t.Fatalf("the events were %v, want %v", kinds(got), want)
	}
	if !got[1].Resumed.Complete {
		t.Error("nothing was lost, and the reconnect should say so")
	}
}

func TestAToolCallSentAgainAfterAReconnectIsRunOnce(t *testing.T) {
	release := make(chan struct{})
	answered := make(chan Frame, 1)
	call := Frame{"type": "tool_call", "id": "call-1", "name": "count", "arguments": `{}`, "cursor": "first.1"}
	router := newRestartable(t,
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
			_ = connection.WriteJSON(call)
			closes(connection, websocket.CloseServiceRestart, "router restarting")
		},
		func(connection *websocket.Conn) {
			// The call is still waiting, so the router sends it again.
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.1", "complete": true})
			_ = connection.WriteJSON(call)
			close(release)
			var result Frame
			if err := connection.ReadJSON(&result); err == nil {
				answered <- result
			}
			closes(connection, websocket.CloseNormalClosure, "the session ended")
		},
	)
	pipeline := Accelerated(Config{Backend: Backend{URL: router.url(), CustomerID: "acme"}})
	pipeline.firstRetry = 10 * time.Millisecond
	var ran atomic.Int32
	if err := tools.Register(pipeline.Functions(), "count", "count",
		func(context.Context, struct{}) (any, error) {
			ran.Add(1)
			<-release
			return "counted", nil
		}); err != nil {
		t.Fatal(err)
	}
	if err := pipeline.Watch(t.Context(), &acceleration.Session{Id: "session-1"}); err != nil {
		t.Fatal(err)
	}

	collect(t, pipeline.Events())

	select {
	case result := <-answered:
		if result["tool_call_id"] != "call-1" || result["output"] != "counted" {
			t.Errorf("the reconnected socket was sent %v", result)
		}
	default:
		t.Error("the call was never answered on the socket that came back")
	}
	if ran.Load() != 1 {
		t.Errorf("the function ran %d times", ran.Load())
	}
}

func TestATextSessionTheRouterEndsIsNotReconnected(t *testing.T) {
	router := newRestartable(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
		closes(connection, websocket.CloseNormalClosure, "the session ended")
	})
	pipeline := watching(t, router, &acceleration.Session{Id: "session-1"})

	got := collect(t, pipeline.Events())

	if len(got) != 0 {
		t.Errorf("the first resumed frame only says where the stream starts, and was handed on: %v", kinds(got))
	}
	if len(router.queries()) != 1 {
		t.Errorf("a session that ended was watched again")
	}
}

func TestACallsSocketIsNeitherResumableNorReconnected(t *testing.T) {
	router := newRestartable(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "joined"})
		closes(connection, websocket.CloseServiceRestart, "router restarting")
	}, nil)
	pipeline := watching(t, router, &acceleration.Session{Id: "session-1", CallId: "call-1"})

	got := collect(t, pipeline.Events())

	if want := []string{"joined"}; !equalStrings(kinds(got), want) {
		t.Errorf("the events were %v", kinds(got))
	}
	asked := router.queries()
	if len(asked) != 1 || asked[0].Has("resumable") {
		t.Errorf("a call's socket was opened with %v", asked)
	}
}

func TestAReconnectTheRouterRefusesEndsTheEvents(t *testing.T) {
	var router *restartable
	router = newRestartable(t,
		func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
			closes(connection, websocket.CloseServiceRestart, "router restarting")
			go router.restart()
		},
		nil,
	)
	pipeline := watching(t, router, &acceleration.Session{Id: "session-1"})

	collect(t, pipeline.Events())

	if len(router.queries()) != 2 {
		t.Errorf("a session the router no longer has was asked for %d times", len(router.queries()))
	}
}

func TestLeavingWhileReconnectingStopsIt(t *testing.T) {
	router := newRestartable(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
		closes(connection, websocket.CloseServiceRestart, "router restarting")
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: router.url(), CustomerID: "acme"}})
	pipeline.firstRetry = time.Hour
	if err := pipeline.Watch(t.Context(), &acceleration.Session{Id: "session-1"}); err != nil {
		t.Fatal(err)
	}
	events := pipeline.Events()
	router.stop()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = pipeline.Leave(ctx)

	select {
	case _, open := <-events:
		if open {
			t.Error("an event arrived after leaving")
		}
	case <-time.After(time.Second):
		t.Error("leaving did not stop the reconnect")
	}
}

// reopened is a router whose events socket answers the nth attempt to open it with the nth
// answer, and every attempt past the last with the last. It counts the attempts.
func reopened(t *testing.T, answers ...http.HandlerFunc) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var attempts atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(attempts.Add(1))
		answers[min(n, len(answers))-1](w, r)
	}))
	t.Cleanup(server.Close)
	return server, &attempts
}

// served upgrades the socket and hands it to serve.
func served(serve func(*websocket.Conn)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		serve(connection)
	}
}

// refused answers the socket with status and the router's envelope.
func refused(status int, code string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"error":{"message":"not now","type":"unavailable","code":"` + code + `"}}`))
	}
}

// restarting is a socket the router drops while restarting, after saying where it got to.
func restarting(connection *websocket.Conn) {
	_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.0", "complete": true})
	_ = connection.WriteJSON(Frame{"type": "responding", "cursor": "first.1"})
	closes(connection, websocket.CloseServiceRestart, "router restarting")
}

func TestAReconnectTheRouterCannotTakeIsGivenUpWhenTheWindowEnds(t *testing.T) {
	server, attempts := reopened(t, served(restarting), refused(http.StatusServiceUnavailable, "unavailable"))
	pipeline := Accelerated(Config{Backend: Backend{URL: server.URL, CustomerID: "acme"}})
	pipeline.firstRetry = 10 * time.Millisecond
	pipeline.resumeFor = 150 * time.Millisecond
	if err := pipeline.Watch(t.Context(), &acceleration.Session{Id: "session-1"}); err != nil {
		t.Fatal(err)
	}

	started := time.Now()
	got := collect(t, pipeline.Events())

	if want := []string{"responding"}; !equalStrings(kinds(got), want) {
		t.Errorf("the events were %v, want %v", kinds(got), want)
	}
	if attempts.Load() < 3 {
		t.Errorf("a router that is only unavailable was tried %d times before giving up", attempts.Load())
	}
	if took := time.Since(started); took < 150*time.Millisecond {
		t.Errorf("the reconnect gave up after %v, before its window ended", took)
	}
}

func TestAReconnectTheRouterRateLimitsIsTriedAgain(t *testing.T) {
	server, attempts := reopened(t,
		served(restarting),
		refused(http.StatusTooManyRequests, "rate_limited"),
		refused(http.StatusTooManyRequests, "rate_limited"),
		served(func(connection *websocket.Conn) {
			_ = connection.WriteJSON(Frame{"type": "resumed", "cursor": "first.1", "complete": true})
			_ = connection.WriteJSON(Frame{"type": "responded", "text": "Done.", "cursor": "first.2"})
			closes(connection, websocket.CloseNormalClosure, "the session ended")
		}),
	)
	pipeline := Accelerated(Config{Backend: Backend{URL: server.URL, CustomerID: "acme"}})
	pipeline.firstRetry = 10 * time.Millisecond
	if err := pipeline.Watch(t.Context(), &acceleration.Session{Id: "session-1"}); err != nil {
		t.Fatal(err)
	}

	got := collect(t, pipeline.Events())

	if want := []string{"responding", "resumed", "responded"}; !equalStrings(kinds(got), want) {
		t.Errorf("the events were %v, want %v: a 429 is no refusal for good", kinds(got), want)
	}
	if attempts.Load() != 4 {
		t.Errorf("the socket was opened %d times, want 4", attempts.Load())
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
