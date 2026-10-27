package stream

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
)

// ToolCallSuite is how a session's tool calls are run, cancelled and answered over its socket.
type ToolCallSuite struct {
	suite.Suite
}

func TestToolCallSuite(t *testing.T) {
	suite.Run(t, new(ToolCallSuite))
}

// join opens a voice session against backend with the functions registered on pipeline.
func (s *ToolCallSuite) join(pipeline *Pipeline) {
	_, err := pipeline.Join(s.T().Context(), Call{Voice: true})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = pipeline.Leave(context.Background()) })
}

func (s *ToolCallSuite) TestASessionDeclaresEachToolsPolicies() {
	backend := newRouter(s.T(), func(*testing.T, *websocket.Conn) {})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	noop := func(context.Context, struct{}) (any, error) { return nil, nil }
	s.Require().NoError(tools.Register(pipeline.Functions(), "search_flights", "Search for flights", noop,
		tools.Interruptible(true), tools.Timeout(5*time.Second)))
	s.Require().NoError(tools.Register(pipeline.Functions(), "book_flight", "Book a flight", noop))

	s.join(pipeline)

	declared := *backend.created(s.T()).Tools
	s.Require().Len(declared, 2)
	byName := map[string]acceleration.SessionTool{declared[0].Name: declared[0], declared[1].Name: declared[1]}
	s.Require().NotNil(byName["search_flights"].Interruptible)
	s.True(*byName["search_flights"].Interruptible)
	s.Require().NotNil(byName["search_flights"].TimeoutMs)
	s.EqualValues(5000, *byName["search_flights"].TimeoutMs)
	s.Nil(byName["book_flight"].Interruptible, "a tool that said nothing leaves the router's default")
	s.Nil(byName["book_flight"].TimeoutMs)
	s.Nil(byName["book_flight"].Approval)
}

func (s *ToolCallSuite) TestASessionDeclaresTheQuestionAToolAsksBeforeItRuns() {
	backend := newRouter(s.T(), func(*testing.T, *websocket.Conn) {})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "book_flight", "Book a flight",
		func(context.Context, struct {
			FlightID string `json:"flight_id"`
		}) (any, error) {
			return nil, nil
		}, tools.NeedsApproval("Book flight {flight_id}?")))

	s.join(pipeline)

	declared := *backend.created(s.T()).Tools
	s.Require().Len(declared, 1)
	s.Equal(&acceleration.SessionToolApproval{Title: "Book flight {flight_id}?"}, declared[0].Approval)
}

func (s *ToolCallSuite) TestACancelledToolCallStopsTheFunction() {
	started := make(chan struct{})
	stopped := make(chan error, 1)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
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
	s.Require().NoError(tools.Register(pipeline.Functions(), "search_flights", "Search for flights",
		func(ctx context.Context, _ struct{}) (any, error) {
			close(started)
			<-ctx.Done()
			stopped <- ctx.Err()
			return nil, ctx.Err()
		}))

	s.join(pipeline)

	select {
	case err := <-stopped:
		s.ErrorIs(err, context.Canceled)
	case <-time.After(3 * time.Second):
		s.Fail("the function is still running after the router cancelled its call")
	}
}

func (s *ToolCallSuite) TestACallTheRouterSendsAgainRunsOnce() {
	answers := make(chan map[string]any, 2)
	release := make(chan struct{})
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
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
	s.Require().NoError(tools.Register(pipeline.Functions(), "book_flight", "Book a flight",
		func(context.Context, struct{}) (any, error) {
			ran.Add(1)
			<-release
			return "booked", nil
		}))

	s.join(pipeline)

	for range 2 {
		select {
		case frame := <-answers:
			s.Equal("call-1", frame["tool_call_id"])
			s.Equal("booked", frame["output"])
		case <-time.After(3 * time.Second):
			s.FailNow("a copy of the call was never answered")
		}
	}
	s.EqualValues(1, ran.Load(), "the booking ran more than once for one call")
}

func (s *ToolCallSuite) TestACallForAToolHeldElsewhereIsLeftForWhoeverHoldsIt() {
	answered := make(chan map[string]any, 1)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
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

	s.join(pipeline)

	select {
	case frame := <-answered:
		s.Failf("answered a call for a tool this process does not hold", "%v", frame)
	case <-time.After(500 * time.Millisecond):
	}
}

func (s *ToolCallSuite) TestTheSocketSaysWhichToolsItRuns() {
	backend := newRouter(s.T(), func(*testing.T, *websocket.Conn) {})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	noop := func(context.Context, struct{}) (any, error) { return nil, nil }
	s.Require().NoError(tools.Register(pipeline.Functions(), "search_flights", "Search for flights", noop))
	s.Require().NoError(tools.Register(pipeline.Functions(), "book_flight", "Book a flight", noop))

	s.join(pipeline)

	backend.mu.Lock()
	defer backend.mu.Unlock()
	s.Require().True(backend.watchedWith.Has("tools"))
	s.Equal("search_flights,book_flight", backend.watchedWith.Get("tools"))
}

func (s *ToolCallSuite) TestACallAddressedToAnotherWatcherIsLeftAlone() {
	answered := make(chan map[string]any, 1)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-5", "name": "lookup_order", "arguments": "{}", "run": false,
		}); err != nil {
			return
		}
		var frame map[string]any
		if connection.ReadJSON(&frame) == nil {
			answered <- frame
		}
	})
	var ran atomic.Bool
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "lookup_order", "Look up an order",
		func(context.Context, struct{}) (any, error) { ran.Store(true); return "found", nil }))

	s.join(pipeline)

	select {
	case frame := <-answered:
		s.Failf("answered a call the router addressed to another watcher", "%v", frame)
	case <-time.After(500 * time.Millisecond):
	}
	s.False(ran.Load())
}

func (s *ToolCallSuite) TestAFunctionReadsTheCallItIsRunningFor() {
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(map[string]any{
			"type": "tool_call", "id": "call-3", "name": "lookup_order", "arguments": "{}", "turn_id": "turn-2", "approved": true,
			"session_id": "session-1", "user": map[string]any{"id": "jean", "name": "Jean"},
			"conversation": map[string]any{"cid": "messaging:support-1", "custom": map[string]any{"plan": "gold"}},
		}); err != nil {
			return
		}
		var frame map[string]any
		_ = connection.ReadJSON(&frame)
	})
	seen := make(chan tools.Call, 1)
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "lookup_order", "Look up an order",
		func(ctx context.Context, _ struct{}) (any, error) {
			seen <- *tools.CallFrom(ctx)
			return "found", nil
		}))

	s.join(pipeline)

	select {
	case toolCallContext := <-seen:
		s.Equal("call-3", toolCallContext.ID)
		s.Equal("session-1", toolCallContext.SessionID)
		s.Equal("turn-2", toolCallContext.TurnID)
		s.Equal(tools.User{ID: "jean", Name: "Jean"}, toolCallContext.User)
		s.Equal(tools.Conversation{CID: "messaging:support-1", Custom: map[string]any{"plan": "gold"}}, toolCallContext.Conversation)
		s.True(toolCallContext.Approved)
	case <-time.After(3 * time.Second):
		s.Fail("the function never ran")
	}
}

func (s *ToolCallSuite) TestAFailingToolOnlyFailsItsOwnCall() {
	answers := make(chan map[string]any, 2)
	backend := newRouter(s.T(), func(t *testing.T, connection *websocket.Conn) {
		for _, id := range []string{"call-1", "call-2"} {
			if connection.WriteJSON(map[string]any{"type": "tool_call", "id": id, "name": "flaky", "arguments": "{}"}) != nil {
				return
			}
		}
		for range 2 {
			var frame map[string]any
			if connection.ReadJSON(&frame) != nil {
				return
			}
			answers <- frame
		}
	})
	var calls atomic.Int32
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.Require().NoError(tools.Register(pipeline.Functions(), "flaky", "Fails the first time",
		func(context.Context, struct{}) (any, error) {
			if calls.Add(1) == 1 {
				return nil, errors.New("try again")
			}
			return "fine", nil
		}))

	s.join(pipeline)

	results := map[string]map[string]any{}
	for range 2 {
		select {
		case frame := <-answers:
			results[frame["tool_call_id"].(string)] = frame
		case <-time.After(3 * time.Second):
			s.FailNow("a call was never answered")
		}
	}
	s.Len(results, 2, "each call is answered on its own")
}
