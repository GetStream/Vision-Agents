package stream

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// pool is a stand-in for the router's dispatch socket: a real WebSocket a worker waits on,
// so what is under test is the exchange rather than a description of it.
type pool struct {
	*httptest.Server

	mu       sync.Mutex
	query    string
	received []Frame
	arrived  chan Frame

	// hand is what the router pushes once a worker is waiting.
	hand func(*websocket.Conn)
}

func newPool(t *testing.T, hand func(*websocket.Conn)) *pool {
	t.Helper()

	router := &pool{hand: hand, arrived: make(chan Frame, 8)}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		router.mu.Lock()
		router.query = r.URL.RawQuery
		router.mu.Unlock()

		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()

		if err := connection.WriteJSON(Frame{"type": "ready", "worker_id": "worker-7"}); err != nil {
			return
		}
		if router.hand != nil {
			router.hand(connection)
		}
		for {
			var frame Frame
			if err := connection.ReadJSON(&frame); err != nil {
				return
			}
			router.mu.Lock()
			router.received = append(router.received, frame)
			router.mu.Unlock()
			if frame.Type() == "ping" {
				_ = connection.WriteJSON(Frame{"type": "pong", "at": frame["at"]})
			}
			select {
			case router.arrived <- frame:
			default:
			}
		}
	})

	router.Server = httptest.NewServer(handler)
	t.Cleanup(router.Close)
	return router
}

// told waits for the worker to send a frame of this type, so an assertion about what the
// router heard does not race the goroutine that says it.
func (p *pool) told(t *testing.T, kind string) Frame {
	t.Helper()

	deadline := time.After(3 * time.Second)
	for {
		select {
		case frame := <-p.arrived:
			if frame.Type() == kind {
				return frame
			}
		case <-deadline:
			t.Fatalf("the router was never told %q", kind)
		}
	}
}

func waiting(t *testing.T, router *pool, options DispatchOptions) *Dispatch {
	t.Helper()

	options.Backend = Backend{URL: router.URL, CustomerID: "acme"}
	options.Logger = slog.New(slog.DiscardHandler)
	worker, err := NewDispatch(options)
	if err != nil {
		t.Fatal(err)
	}
	return worker
}

// run waits for work in the background and returns what Run said once it stops.
func run(t *testing.T, ctx context.Context, worker *Dispatch) func() error {
	t.Helper()

	stopped := make(chan error, 1)
	go func() { stopped <- worker.Run(ctx) }()
	return func() error {
		select {
		case err := <-stopped:
			return err
		case <-time.After(5 * time.Second):
			t.Fatal("the worker never stopped waiting")
			return nil
		}
	}
}

func TestAMessageWrittenToAnAgentReachesTheHandler(t *testing.T) {
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "message", "channel_type": "agent", "channel_id": "support-42",
			"agent_id": "support-42", "config_id": "config-1",
			"text":       "does the react sdk retry a failed upload?",
			"message_id": "message-1", "user_id": "sam", "user_name": "Sam",
			"at": "2026-09-08T12:00:00Z",
		})
	})

	answered := make(chan InboundMessage, 1)
	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, message InboundMessage) error {
		answered <- message
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	var message InboundMessage
	select {
	case message = <-answered:
	case <-time.After(3 * time.Second):
		t.Fatal("the message never reached the handler")
	}
	stop()
	_ = stopped()

	if message.ChannelID != "support-42" {
		t.Errorf("the message arrived on %q, want support-42", message.ChannelID)
	}
	if message.AgentID != "support-42" {
		t.Errorf("the agent is %q, want support-42; a session opened on anything else answers elsewhere", message.AgentID)
	}
	if message.ConfigID != "config-1" {
		t.Errorf("the config is %q, want config-1", message.ConfigID)
	}
	if message.Text != "does the react sdk retry a failed upload?" {
		t.Errorf("the text is %q", message.Text)
	}
	if message.UserID != "sam" || message.UserName != "Sam" {
		t.Errorf("it was written by %q/%q, want sam/Sam", message.UserID, message.UserName)
	}
	if !message.At.Equal(time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)) {
		t.Errorf("it was written at %s", message.At)
	}
}

func TestWhatTheChannelWasCreatedWithReachesTheWorker(t *testing.T) {
	// It is the only way a worker learns what a conversation is for. The router has no
	// opinion about an organization or a locale and should not need one; it carries what
	// the channel was created with and lets the worker decide what any of it means.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "message", "channel_id": "support-42", "text": "hello",
			"custom": map[string]any{
				"organization_id": "1234",
				"locale":          "en-GB",
				// Stream takes arbitrary JSON here and a worker reads strings, so
				// anything else is dropped rather than rendered into one.
				"seats": 12,
			},
		})
	})

	answered := make(chan InboundMessage, 1)
	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, message InboundMessage) error {
		answered <- message
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	message := <-answered
	stop()
	_ = stopped()

	if message.Custom["organization_id"] != "1234" {
		t.Errorf("the organization is %q, want 1234", message.Custom["organization_id"])
	}
	if message.Custom["locale"] != "en-GB" {
		t.Errorf("the locale is %q, want en-GB", message.Custom["locale"])
	}
	if seats, carried := message.Custom["seats"]; carried {
		t.Errorf("a number arrived as the string %q; a worker reading it would believe it", seats)
	}
}

func TestAMessageCarryingNothingCustomStillReadsAsEmpty(t *testing.T) {
	// A handler should be able to read the field without checking whether the router
	// bothered to send it.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "message", "channel_id": "support-42", "text": "hello"})
	})

	answered := make(chan InboundMessage, 1)
	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, message InboundMessage) error {
		answered <- message
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	message := <-answered
	stop()
	_ = stopped()

	if message.Custom == nil {
		t.Fatal("the custom data is nil; reading it would panic on a channel created with nothing")
	}
	if len(message.Custom) != 0 {
		t.Errorf("the custom data is %v, want empty", message.Custom)
	}
}

func TestAMessageNamingNoAgentFallsBackToItsChannel(t *testing.T) {
	// The two are the same thing at the router, and a worker that opened a session on an
	// empty agent id would answer in a conversation nobody is reading.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "message", "channel_id": "support-42", "text": "hello"})
	})

	answered := make(chan InboundMessage, 1)
	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, message InboundMessage) error {
		answered <- message
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	message := <-answered
	stop()
	_ = stopped()

	if message.AgentID != "support-42" {
		t.Errorf("the agent is %q, want the channel it was written in", message.AgentID)
	}
	if message.ChannelType != "agent" {
		t.Errorf("the channel type is %q, want agent", message.ChannelType)
	}
}

func TestACallThatWasAnsweredIsAcceptedAtTheRouter(t *testing.T) {
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "call", "call_id": "call-1", "call_type": "default",
			"called_number": "+13035550100", "caller_number": "+13035550111",
			"custom": map[string]any{"campaign": "spring"},
		})
	})

	worker := waiting(t, router, DispatchOptions{})
	var answered InboundCall
	worker.OnCall(func(_ context.Context, call InboundCall) error {
		answered = call
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	accepted := router.told(t, "accepted")
	stop()
	_ = stopped()

	if accepted.String("call_id") != "call-1" {
		t.Errorf("the router was told about %q, want call-1", accepted.String("call_id"))
	}
	if answered.CallerNumber != "+13035550111" {
		t.Errorf("the caller is %q", answered.CallerNumber)
	}
	if answered.Custom["campaign"] != "spring" {
		t.Errorf("what was on the call did not come through: %v", answered.Custom)
	}
}

func TestACallThatCouldNotBeAnsweredIsRejectedWithTheReason(t *testing.T) {
	// A call nobody answered has to show up at the router rather than only in this
	// process's log, because the router is where somebody is looking when a caller says
	// nobody picked up.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "call", "call_id": "call-1"})
	})

	worker := waiting(t, router, DispatchOptions{})
	worker.OnCall(func(_ context.Context, _ InboundCall) error {
		return context.DeadlineExceeded
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	rejected := router.told(t, "rejected")
	stop()
	_ = stopped()

	if rejected.String("call_id") != "call-1" {
		t.Errorf("the router was told about %q, want call-1", rejected.String("call_id"))
	}
	if rejected.String("reason") != context.DeadlineExceeded.Error() {
		t.Errorf("the reason given was %q", rejected.String("reason"))
	}
}

func TestAMessageThatCouldNotBeAnsweredIsNotReportedToTheRouter(t *testing.T) {
	// Accepting and rejecting are about a caller waiting on a line, and there is no line
	// here. Reporting one would have the router treat a failed answer as a failed
	// hand-over and say a worker refused work it took.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "message", "channel_id": "support-42", "text": "hello"})
	})

	failed := make(chan struct{})
	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, _ InboundMessage) error {
		close(failed)
		return context.DeadlineExceeded
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	<-failed
	// Long enough that a frame on its way would have arrived.
	time.Sleep(100 * time.Millisecond)
	stop()
	_ = stopped()

	router.mu.Lock()
	defer router.mu.Unlock()
	for _, frame := range router.received {
		if frame.Type() == "rejected" || frame.Type() == "accepted" {
			t.Errorf("the router was told %q about a message", frame.Type())
		}
	}
}

func TestTheWorkerTellsTheRouterWhatItCanHold(t *testing.T) {
	// The router passes over a full worker rather than queueing behind it, so this is a
	// promise about what this process can answer and it has to be on the handshake.
	router := newPool(t, nil)
	worker := waiting(t, router, DispatchOptions{Capacity: 9})
	worker.OnMessage(func(context.Context, InboundMessage) error { return nil })

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	deadline := time.After(3 * time.Second)
	for worker.WorkerID() == "" {
		select {
		case <-deadline:
			t.Fatal("the router never named the worker")
		case <-time.After(5 * time.Millisecond):
		}
	}
	stop()
	_ = stopped()

	router.mu.Lock()
	defer router.mu.Unlock()
	if router.query != "capacity=9" {
		t.Errorf("the worker waited with %q, want capacity=9", router.query)
	}
}

func TestAWorkerThatWasNeverToldWhatToDoRefusesToWait(t *testing.T) {
	// Work would arrive with nothing to do it, which at the router looks like a worker
	// that took the call.
	router := newPool(t, nil)
	worker := waiting(t, router, DispatchOptions{})

	if err := worker.Run(t.Context()); err == nil {
		t.Fatal("a worker with no handler waited anyway")
	}
}

func TestLoadIsReportedWithTheRoundTripTheWorkerMeasured(t *testing.T) {
	router := newPool(t, nil)
	worker := waiting(t, router, DispatchOptions{ReportEvery: 20 * time.Millisecond})
	worker.OnMessage(func(context.Context, InboundMessage) error { return nil })

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	load := router.told(t, "load")
	stop()
	_ = stopped()

	if _, said := load["active_agents"]; !said {
		t.Error("the report says nothing about how much the worker is holding")
	}
	if worker.LatencyMS() <= 0 {
		t.Errorf("the round trip measured %v ms; a report carrying zero reads at the router as no latency at all", worker.LatencyMS())
	}
}

func TestStoppingWaitsForWorkAlreadyBeingAnswered(t *testing.T) {
	// Dropping a call on the way out would hang up on whoever is talking.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "call", "call_id": "call-1"})
	})

	started := make(chan struct{})
	finished := make(chan struct{})
	worker := waiting(t, router, DispatchOptions{})
	worker.OnCall(func(context.Context, InboundCall) error {
		close(started)
		time.Sleep(150 * time.Millisecond)
		close(finished)
		return nil
	})

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	stopped := run(t, ctx, worker)

	<-started
	stop()
	_ = stopped()

	select {
	case <-finished:
	default:
		t.Error("the worker stopped while it was still answering a call")
	}
}

func TestARouterThatStopsDispatchingIsNotAFailure(t *testing.T) {
	// The router closes the socket normally when it stops handing out work, and a worker
	// that reported that as an error would have a supervisor restarting it in a loop.
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, "dispatch stopped"))
	})

	worker := waiting(t, router, DispatchOptions{})
	worker.OnMessage(func(context.Context, InboundMessage) error { return nil })

	if err := worker.Run(t.Context()); err != nil {
		t.Errorf("a router that stopped dispatching was reported as %v", err)
	}
}

func TestAHostedFunctionIsDeclaredAndAnsweredOverTheDispatchSocket(t *testing.T) {
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "tool_call", "id": "call-1", "session_id": "s", "name": "investigate_sdk", "arguments": `{"sdk":"android"}`})
	})
	functions := tools.NewRegistry()
	if err := tools.Register(functions, "investigate_sdk", "Read SDK source", func(_ context.Context, in struct {
		SDK string `json:"sdk"`
	}) (any, error) {
		return "read " + in.SDK, nil
	}); err != nil {
		t.Fatal(err)
	}
	worker := waiting(t, router, DispatchOptions{})
	worker.Host("support", functions, time.Minute)

	ctx, cancel := context.WithCancel(t.Context())
	stopped := run(t, ctx, worker)

	declared := router.told(t, "host_tools")
	if declared.String("config_id") != "support" || declared.Int("timeout_ms") != 60000 {
		t.Errorf("the router was told %v", declared)
	}
	answered := router.told(t, "tool_result")
	if answered.String("id") != "call-1" || answered.String("output") != "read android" {
		t.Errorf("the call was answered %v", answered)
	}
	cancel()
	stopped()
}

func TestAWorkerTheRouterDropsReconnectsAndHostsAgain(t *testing.T) {
	// The first connection is cut without a close frame, the way a router pod being
	// replaced ends it, and the second goes away with one. A worker that stopped at
	// either would leave every session on the config without its tools.
	var connections sync.Mutex
	opened := 0
	router := newPool(t, func(connection *websocket.Conn) {
		connections.Lock()
		opened++
		this := opened
		connections.Unlock()
		switch this {
		case 1:
			_ = connection.UnderlyingConn().Close()
		case 2:
			_ = connection.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseGoingAway, "shutting down"))
		}
	})
	functions := tools.NewRegistry()
	if err := tools.Register(functions, "investigate_sdk", "Read SDK source", func(context.Context, struct{}) (any, error) { return "", nil }); err != nil {
		t.Fatal(err)
	}
	worker := waiting(t, router, DispatchOptions{})
	worker.firstRetry = 10 * time.Millisecond
	worker.Host("support", functions, 0)

	ctx, cancel := context.WithCancel(t.Context())
	stopped := run(t, ctx, worker)

	deadline := time.After(3 * time.Second)
	for {
		connections.Lock()
		reached := opened
		connections.Unlock()
		if reached >= 3 {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("the worker connected %d times, want it back after each drop", reached)
		case <-time.After(10 * time.Millisecond):
		}
	}
	if declared := router.told(t, "host_tools"); declared.String("config_id") != "support" {
		t.Errorf("the reconnected worker declared %v", declared)
	}
	cancel()
	if err := stopped(); err != nil && err != context.Canceled {
		t.Errorf("a cancelled worker stopped with %v", err)
	}
}

func TestAWorkerWhoseToolsAreRefusedStopsWaiting(t *testing.T) {
	router := newPool(t, func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "hosting_refused", "config_id": "support", "reason": "there is no such config"})
	})
	functions := tools.NewRegistry()
	if err := tools.Register(functions, "investigate_sdk", "Read SDK source", func(context.Context, struct{}) (any, error) { return "", nil }); err != nil {
		t.Fatal(err)
	}
	worker := waiting(t, router, DispatchOptions{})
	worker.Host("support", functions, 0)

	if err := run(t, t.Context(), worker)(); err == nil {
		t.Fatal("a worker nobody will call kept waiting")
	}
}
