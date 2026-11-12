package stream

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// ToolApprovalSuite is how a call to a function that asks first (tools.NeedsApproval) waits
// for a person, in a session and on a dispatch worker.
type ToolApprovalSuite struct {
	suite.Suite

	// ran counts the runs of book_flight.
	ran atomic.Int64
	// sent are the frames the session's router was sent, in order.
	sent chan Frame
}

func TestToolApprovalSuite(t *testing.T) {
	suite.Run(t, new(ToolApprovalSuite))
}

func (s *ToolApprovalSuite) SetupTest() {
	s.ran.Store(0)
	s.sent = make(chan Frame, 8)
}

// bookFlight registers book_flight, which asks before it runs and counts each run.
func (s *ToolApprovalSuite) bookFlight(functions *tools.Registry) {
	s.Require().NoError(tools.Register(functions, "book_flight", "Book a flight",
		func(_ context.Context, in struct {
			FlightID string `json:"flight_id"`
		}) (any, error) {
			s.ran.Add(1)
			return "booked " + in.FlightID, nil
		}, tools.NeedsApproval("Book flight {flight_id}?")))
}

// asked opens a session held in writing whose router sends frames and then records what it
// is sent, and returns the pipeline once it is watching.
func (s *ToolApprovalSuite) asked(frames ...Frame) *Pipeline {
	sent := s.sent
	backend := newRouter(s.T(), func(_ *testing.T, connection *websocket.Conn) {
		for _, frame := range frames {
			if err := connection.WriteJSON(frame); err != nil {
				return
			}
		}
		for {
			var frame Frame
			if err := connection.ReadJSON(&frame); err != nil || frame.Type() == "close" {
				return
			}
			sent <- frame
		}
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: backend.URL, CustomerID: "acme"}})
	s.bookFlight(pipeline.Functions())
	_, err := pipeline.Join(s.T().Context(), Call{})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = pipeline.Leave(context.Background()) })
	return pipeline
}

// told is the next frame the session's router was sent.
func (s *ToolApprovalSuite) told() Frame {
	select {
	case frame := <-s.sent:
		return frame
	case <-time.After(5 * time.Second):
		s.FailNow("the router was sent nothing")
		return nil
	}
}

// booking is a call to book_flight as the router sends it in writing: nobody was asked.
func booking() Frame {
	return Frame{
		"type": "tool_call", "id": "call-42", "name": "book_flight", "arguments": `{"flight_id":"UA 12"}`,
		"request_id": "request-1", "turn_id": "turn-1",
	}
}

func (s *ToolApprovalSuite) TestACallThatAsksFirstDoesNotRunUntilAPersonAllowsIt() {
	pipeline := s.asked(booking())

	waiting := next(s.T(), pipeline.Events())
	s.Equal("tool_call", waiting.Kind)
	s.Equal("call-42", waiting.Frame.String("id"))
	s.Equal(`{"flight_id":"UA 12"}`, waiting.Frame.String("arguments"))
	s.Zero(s.ran.Load(), "the call ran before anybody was asked")

	s.Require().NoError(pipeline.AnswerApproval("call-42", "request-1", true, ""))

	// The answer comes first, so a result ahead of it is a call that did not wait.
	s.Equal(Frame{
		"type": "tool_approval", "tool_call_id": "call-42", "request_id": "request-1", "turn_id": "turn-1", "allowed": true,
	}, s.told())
	s.Equal(Frame{
		"type": "tool_result", "tool_call_id": "call-42", "request_id": "request-1", "turn_id": "turn-1", "output": "booked UA 12",
	}, s.told())
	s.EqualValues(1, s.ran.Load())
}

func (s *ToolApprovalSuite) TestACallAPersonDeclinesNeverRuns() {
	pipeline := s.asked(booking())
	next(s.T(), pipeline.Events())

	s.Require().NoError(pipeline.AnswerApproval("call-42", "request-1", false, "Not now"))

	s.Equal(Frame{
		"type": "tool_approval", "tool_call_id": "call-42", "request_id": "request-1", "turn_id": "turn-1",
		"allowed": false, "summary": "Not now",
	}, s.told())
	s.Equal(Frame{
		"type": "tool_result", "tool_call_id": "call-42", "request_id": "request-1", "turn_id": "turn-1",
		"error": "The person declined this call.",
	}, s.told())
	// Leaving waits for every call still running, so a run started by the answer is counted.
	s.Require().NoError(pipeline.Leave(context.Background()))
	s.Zero(s.ran.Load())
}

func (s *ToolApprovalSuite) TestAnAnswerGivenTwiceRunsTheCallOnce() {
	pipeline := s.asked(booking())
	next(s.T(), pipeline.Events())

	s.Require().NoError(pipeline.AnswerApproval("call-42", "request-1", true, ""))
	s.Require().NoError(pipeline.AnswerApproval("call-42", "request-1", true, ""))

	s.Equal("tool_approval", s.told().Type())
	second, third := s.told(), s.told()
	s.ElementsMatch([]string{"tool_approval", "tool_result"}, []string{second.Type(), third.Type()})
	s.Require().NoError(pipeline.Leave(context.Background()))
	s.EqualValues(1, s.ran.Load())
}

func (s *ToolApprovalSuite) TestACallTheCallerSaidYesToOutLoudRunsAsItComes() {
	call := booking()
	call["approved"] = true
	s.asked(call)

	s.Equal(Frame{
		"type": "tool_result", "tool_call_id": "call-42", "request_id": "request-1", "turn_id": "turn-1", "output": "booked UA 12",
	}, s.told())
}

func (s *ToolApprovalSuite) TestACallTheRouterCancelledNoLongerWaitsForAnAnswer() {
	pipeline := s.asked(booking(),
		Frame{"type": "tool_cancel", "id": "call-42"},
		Frame{"type": "heard", "text": "never mind"})
	for event := next(s.T(), pipeline.Events()); event.Kind != "heard"; event = next(s.T(), pipeline.Events()) {
	}

	s.Require().NoError(pipeline.AnswerApproval("call-42", "request-1", true, ""))

	s.Equal("tool_approval", s.told().Type())
	s.Require().NoError(pipeline.Leave(context.Background()))
	s.Zero(s.ran.Load(), "a call the router gave up on ran once it was allowed")
}

// hosted runs a worker hosting book_flight, sends it one call, and returns the result the
// router is sent.
func (s *ToolApprovalSuite) hosted(call Frame) Frame {
	router := newPool(s.T(), func(connection *websocket.Conn) { _ = connection.WriteJSON(call) })
	functions := tools.NewRegistry()
	s.bookFlight(functions)
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.Host(hostedAgent{"travel", functions}, time.Minute)
	ctx, cancel := context.WithCancel(s.T().Context())
	stopped := run(s.T(), ctx, worker)
	result := router.told(s.T(), "tool_result")
	cancel()
	_ = stopped()
	return result
}

func (s *ToolApprovalSuite) TestAHostedCallThatAsksFirstIsRefusedWhenNobodySaidYes() {
	result := s.hosted(Frame{
		"type": "tool_call", "id": "call-42", "session_id": "session-1", "name": "book_flight", "arguments": `{"flight_id":"UA 12"}`,
	})

	s.Equal("call-42", result.String("tool_call_id"))
	s.Equal("book_flight asks a person before it runs, and nobody has allowed this call", result.String("error"))
	s.NotContains(result, "output")
	s.Zero(s.ran.Load())
}

func (s *ToolApprovalSuite) TestAHostedCallTheCallerSaidYesToRuns() {
	result := s.hosted(Frame{
		"type": "tool_call", "id": "call-42", "session_id": "session-1", "name": "book_flight", "arguments": `{"flight_id":"UA 12"}`,
		"approved": true,
	})

	s.Equal("booked UA 12", result.String("output"))
	s.EqualValues(1, s.ran.Load())
}
