package stream

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
)

// HostedCallSuite is how a worker answers the calls it runs for sessions it did not open.
type HostedCallSuite struct {
	suite.Suite
}

func TestHostedCallSuite(t *testing.T) {
	suite.Run(t, new(HostedCallSuite))
}

// answer hosts photograph as photograph_room, sends the router one call to it, and returns the
// result the router is sent.
func (s *HostedCallSuite) answer(photograph func(context.Context, struct{}) (any, error)) Frame {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "tool_call", "id": "call-1", "session_id": "s", "name": "photograph_room", "arguments": "{}"})
	})
	functions := tools.NewRegistry()
	s.Require().NoError(tools.Register(functions, "photograph_room", "Photograph the room", photograph))
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"stream-support", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	s.T().Cleanup(func() {
		cancel()
		stopped()
	})
	return router.told(s.T(), "tool_result")
}

func (s *HostedCallSuite) TestAWorkerDeclaresEachToolsPolicies() {
	router := newPool(s.T(), func(*websocket.Conn) {})
	functions := tools.NewRegistry()
	noop := func(context.Context, struct{}) (any, error) { return nil, nil }
	s.Require().NoError(tools.Register(functions, "investigate_sdk", "Read SDK source", noop,
		tools.Interruptible(true), tools.Timeout(90*time.Second)))
	s.Require().NoError(tools.Register(functions, "search_docs", "Search the docs", noop))
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"stream-support", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	defer func() {
		cancel()
		stopped()
	}()

	declared := router.told(s.T(), "host_tools")["tools"].([]any)

	s.Require().Len(declared, 2)
	byName := map[string]map[string]any{}
	for _, tool := range declared {
		byName[tool.(map[string]any)["name"].(string)] = tool.(map[string]any)
	}
	s.Equal(true, byName["investigate_sdk"]["interruptible"])
	s.EqualValues(90000, byName["investigate_sdk"]["timeout_ms"])
	s.NotContains(byName["search_docs"], "interruptible")
	s.NotContains(byName["search_docs"], "timeout_ms")
}

func (s *HostedCallSuite) TestAFunctionReadsTheCallItIsRunningFor() {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "tool_call", "id": "call-1", "session_id": "s", "turn_id": "turn-1", "name": "photograph_room", "arguments": "{}",
			"user":         Frame{"id": "jean", "name": "Jean"},
			"conversation": Frame{"cid": "messaging:support-1", "custom": Frame{"plan": "gold"}},
		})
	})
	seen := make(chan tools.Call, 1)
	functions := tools.NewRegistry()
	s.Require().NoError(tools.Register(functions, "photograph_room", "Photograph the room", func(ctx context.Context, _ struct{}) (any, error) {
		seen <- *tools.CallFrom(ctx)
		return "the kitchen", nil
	}))
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"stream-support", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	defer func() {
		cancel()
		stopped()
	}()

	select {
	case call := <-seen:
		s.Equal(tools.Call{
			ID: "call-1", SessionID: "s", TurnID: "turn-1",
			User:         tools.User{ID: "jean", Name: "Jean"},
			Conversation: tools.Conversation{CID: "messaging:support-1", Custom: map[string]any{"plan": "gold"}},
		}, call)
	case <-time.After(3 * time.Second):
		s.Fail("the function never ran")
	}
}

func (s *HostedCallSuite) TestAResultNamesItsCallAsASessionsDoes() {
	answered := s.answer(func(context.Context, struct{}) (any, error) { return "the kitchen", nil })

	s.Equal("call-1", answered.String("tool_call_id"))
	s.Equal("call-1", answered.String("id"), "a router from before tool_call_id reads id")
	s.Equal("the kitchen", answered.String("output"))
}

func (s *HostedCallSuite) TestAPictureReachesTheRouter() {
	answered := s.answer(func(context.Context, struct{}) (any, error) {
		return tools.Output{Text: "the kitchen", Images: []tools.Image{{URL: "https://example.com/kitchen.jpg"}}}, nil
	})

	s.Empty(answered.String("error"))
	parts, ok := answered["output"].([]any)
	s.Require().True(ok, "the output is %v, want parts", answered["output"])
	s.Require().Len(parts, 2)
	s.Equal("https://example.com/kitchen.jpg", parts[1].(map[string]any)["image_url"].(map[string]any)["url"])
}
