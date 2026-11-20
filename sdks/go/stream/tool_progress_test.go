package stream

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
	"github.com/gorilla/websocket"
)

// toldTheRouter reads what the session's socket sends until n frames other than pings have
// come, and returns them.
func toldTheRouter(connection *websocket.Conn, n int) []map[string]any {
	var told []map[string]any
	for len(told) < n {
		var frame map[string]any
		if connection.ReadJSON(&frame) != nil {
			return told
		}
		told = append(told, frame)
	}
	return told
}

func (s *ToolCallSuite) TestAHeldCallOutlivesTheRouterCancellingIt() {
	answered := make(chan map[string]any, 1)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
		if connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-1", "name": "rebook_flight", "arguments": "{}",
		}) != nil {
			return
		}
		if hold := toldTheRouter(connection, 1); len(hold) != 1 || hold[0]["type"] != "tool_hold" || hold[0]["tool_call_id"] != "call-1" {
			t.Errorf("the router was told %v rather than to hold the call", hold)
			return
		}
		if connection.WriteJSON(map[string]any{"type": "tool_held", "id": "call-1"}) != nil ||
			connection.WriteJSON(map[string]any{"type": "tool_cancel", "id": "call-1"}) != nil {
			return
		}
		if result := toldTheRouter(connection, 1); len(result) == 1 {
			answered <- result[0]
		}
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "rebook_flight", "Move a booking",
		func(ctx context.Context, _ struct{}) (any, error) {
			writing := tools.ToolCallContextFrom(ctx).DisallowInterruptions(ctx)
			<-ctx.Done()
			if writing.Err() != nil {
				return nil, writing.Err()
			}
			return "rebooked", nil
		}))

	s.join(pipeline)

	select {
	case result := <-answered:
		s.Equal("rebooked", result["output"], "the write was stopped after the call was held: %v", result)
	case <-time.After(3 * time.Second):
		s.Fail("the held call never answered")
	}
}

func (s *ToolCallSuite) TestACallTooLateToHoldDoesNotStartItsWrite() {
	answered := make(chan map[string]any, 1)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
		if connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-1", "name": "rebook_flight", "arguments": "{}",
		}) != nil {
			return
		}
		toldTheRouter(connection, 1)
		if connection.WriteJSON(map[string]any{"type": "tool_cancel", "id": "call-1"}) != nil {
			return
		}
		if result := toldTheRouter(connection, 1); len(result) == 1 {
			answered <- result[0]
		}
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "rebook_flight", "Move a booking",
		func(ctx context.Context, _ struct{}) (any, error) {
			writing := tools.ToolCallContextFrom(ctx).DisallowInterruptions(ctx)
			if writing.Err() != nil {
				return nil, writing.Err()
			}
			return "rebooked", nil
		}))

	s.join(pipeline)

	select {
	case result := <-answered:
		s.NotEmpty(result["error"], "the write ran for a call already cancelled: %v", result)
	case <-time.After(3 * time.Second):
		s.Fail("the cancelled call never answered")
	}
}
