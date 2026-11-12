package stream

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// ToolFailureSuite is what the model is told about a call that failed, by a session and by
// a dispatch worker.
type ToolFailureSuite struct {
	suite.Suite
}

func TestToolFailureSuite(t *testing.T) {
	suite.Run(t, new(ToolFailureSuite))
}

// errDatabase is a failure of the kind a function passes up without thinking who reads it.
var errDatabase = errors.New("pq: password authentication failed for user orders_rw at 10.4.0.12")

// errNotStaff is a refusal whose words are the caller's own business.
var errNotStaff = errors.New("user jean has role customer, lookup_order needs role staff")

// lookupOrder registers lookup_order, which takes a number from 1 to 9 and fails with
// failure. Order 7 is one only staff may look up, and nobody calling here is staff.
func (s *ToolFailureSuite) lookupOrder(functions *tools.Registry, failure error) {
	s.Require().NoError(tools.Register(functions, "lookup_order", "Look up an order",
		func(context.Context, struct {
			Number int `json:"number" minimum:"1" maximum:"9"`
		}) (any, error) {
			return nil, failure
		}, tools.Authorize(func(_ context.Context, call tools.Call) error {
			if call.User.ID == "jean" {
				return errNotStaff
			}
			return nil
		})))
}

// inSession calls lookup_order in a session and returns the result its router is sent.
func (s *ToolFailureSuite) inSession(arguments string, failure error) Frame {
	answered := make(chan Frame, 1)
	backend := newRouter(s.T(), func(_ *testing.T, connection *websocket.Conn) {
		if err := connection.WriteJSON(Frame{"type": "tool_call", "id": "call-42", "name": "lookup_order", "arguments": arguments}); err != nil {
			return
		}
		var frame Frame
		if err := connection.ReadJSON(&frame); err != nil {
			return
		}
		answered <- frame
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.lookupOrder(pipeline.Functions(), failure)
	_, err := pipeline.Join(s.T().Context(), Call{Voice: true})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = pipeline.Leave(context.Background()) })

	select {
	case frame := <-answered:
		return frame
	case <-time.After(5 * time.Second):
		s.FailNow("the model is still waiting on the tool")
		return nil
	}
}

// onWorker calls lookup_order on a worker hosting it and returns the result its router is sent.
func (s *ToolFailureSuite) onWorker(arguments string, failure error) Frame {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{"type": "tool_call", "id": "call-42", "session_id": "session-1", "name": "lookup_order", "arguments": arguments})
	})
	functions := tools.NewRegistry()
	s.lookupOrder(functions, failure)
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"orders", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	result := router.told(s.T(), "tool_result")
	cancel()
	_ = stopped()
	return result
}

func (s *ToolFailureSuite) TestAFailuresOwnWordsAreNotSentToTheModel() {
	result := s.inSession(`{"number": 4}`, errDatabase)

	s.Equal("the tool failed", result.String("error"))
	s.NotContains(result, "output")
}

func (s *ToolFailureSuite) TestAHostedFailuresOwnWordsAreNotSentToTheModel() {
	result := s.onWorker(`{"number": 4}`, errDatabase)

	s.Equal("the tool failed", result.String("error"))
	s.NotContains(result, "output")
}

func (s *ToolFailureSuite) TestArgumentsTheSchemaRefusesAreExplainedToTheModel() {
	result := s.inSession(`{"number": 12}`, nil)

	s.Contains(result.String("error"), "/number: maximum: got 12, want 9")
}

func (s *ToolFailureSuite) TestHostedArgumentsTheSchemaRefusesAreExplainedToTheModel() {
	result := s.onWorker(`{"number": 12}`, nil)

	s.Contains(result.String("error"), "/number: maximum: got 12, want 9")
}

func (s *ToolFailureSuite) TestAFailureWrittenForTheModelIsSentAsWritten() {
	result := s.inSession(`{"number": 4}`, fmt.Errorf("orders: %w", tools.UserError("there is no order 4")))

	s.Equal("there is no order 4", result.String("error"))
}

func (s *ToolFailureSuite) TestAHostedFailureWrittenForTheModelIsSentAsWritten() {
	result := s.onWorker(`{"number": 4}`, fmt.Errorf("orders: %w", tools.UserError("there is no order 4")))

	s.Equal("there is no order 4", result.String("error"))
}

func (s *ToolFailureSuite) TestACallItsAuthorizeRefusesTellsTheModelOnlyThatItWasRefused() {
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteJSON(Frame{
			"type": "tool_call", "id": "call-42", "session_id": "session-1", "name": "lookup_order", "arguments": `{"number": 4}`,
			"user": Frame{"id": "jean"},
		})
	})
	functions := tools.NewRegistry()
	s.lookupOrder(functions, nil)
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"orders", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	result := router.told(s.T(), "tool_result")
	cancel()
	_ = stopped()

	s.Equal("lookup_order was refused", result.String("error"))
	s.NotContains(result, "output")
}
