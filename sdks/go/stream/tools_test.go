package stream

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
	"github.com/gorilla/websocket"
)

func TestACancelledToolCallStopsTheFunction(t *testing.T) {
	started := make(chan struct{})
	stopped := make(chan error, 1)
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-1", "name": "search_flights", "arguments": "{}",
		}); err != nil {
			return
		}
		<-started
		if err := connection.WriteJSON(map[string]any{"type": "tool_cancel", "id": "call-1"}); err != nil {
			return
		}
		var frame map[string]any
		_ = connection.ReadJSON(&frame)
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	err := tools.Register(pipeline.Functions(), "search_flights", "Search for flights",
		func(ctx context.Context, _ struct{}) (any, error) {
			close(started)
			<-ctx.Done()
			stopped <- ctx.Err()
			return nil, ctx.Err()
		})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pipeline.Join(t.Context(), Call{Voice: true}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	select {
	case err := <-stopped:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("the function stopped with %v, want context.Canceled", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the function is still running after the router cancelled its call")
	}
}

func TestACallTheRouterSendsAgainRunsOnce(t *testing.T) {
	answers := make(chan map[string]any, 2)
	release := make(chan struct{})
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		call := map[string]any{"type": "tool_call", "id": "call-1", "name": "book_flight", "arguments": "{}"}
		if connection.WriteJSON(call) != nil || connection.WriteJSON(call) != nil {
			return
		}
		close(release)
		for range 2 {
			var frame map[string]any
			if connection.ReadJSON(&frame) != nil {
				return
			}
			answers <- frame
		}
	})

	var ran atomic.Int32
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	err := tools.Register(pipeline.Functions(), "book_flight", "Book a flight",
		func(context.Context, struct{}) (any, error) {
			ran.Add(1)
			<-release
			return "booked", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pipeline.Join(t.Context(), Call{Voice: true}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	for range 2 {
		select {
		case frame := <-answers:
			if frame["tool_call_id"] != "call-1" || frame["output"] != "booked" {
				t.Errorf("the router was sent %v", frame)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("a copy of the call was never answered")
		}
	}
	if got := ran.Load(); got != 1 {
		t.Errorf("the booking ran %d times for one call", got)
	}
}

func TestACallForAToolHeldElsewhereIsLeftForWhoeverHoldsIt(t *testing.T) {
	answered := make(chan map[string]any, 1)
	backend := newRouter(t, func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-9", "name": "share_location", "arguments": "{}",
		}); err != nil {
			return
		}
		var frame map[string]any
		if connection.ReadJSON(&frame) == nil {
			answered <- frame
		}
	})

	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	if _, err := pipeline.Join(t.Context(), Call{Voice: true}); err != nil {
		t.Fatal(err)
	}
	defer pipeline.Leave(context.Background())

	select {
	case frame := <-answered:
		t.Fatalf("answered a call for a tool this process does not hold: %v", frame)
	case <-time.After(500 * time.Millisecond):
	}
}
