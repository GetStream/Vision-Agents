package stream

import (
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// ToolCallStateSuite is how a session's tool call states reach the program watching it.
type ToolCallStateSuite struct {
	suite.Suite
}

func TestToolCallStateSuite(t *testing.T) {
	suite.Run(t, new(ToolCallStateSuite))
}

// watch opens a pipeline on a router that sends frames and ends the session, calling before
// on it first so nothing is missed, and returns it with what reached Events.
func (s *ToolCallStateSuite) watch(before func(*Pipeline), frames ...Frame) (*Pipeline, []Event) {
	router := newRestartable(s.T(), func(connection *websocket.Conn) {
		for _, frame := range frames {
			_ = connection.WriteJSON(frame)
		}
		closes(connection, websocket.CloseNormalClosure, "the session ended")
	})
	pipeline := Accelerated(Config{Backend: Backend{URL: router.url(), CustomerID: "acme"}})
	before(pipeline)
	s.Require().NoError(pipeline.Watch(s.T().Context(), &acceleration.Session{Id: "session-1"}))
	return pipeline, collect(s.T(), pipeline.Events())
}

func (s *ToolCallStateSuite) TestTheLastStateOfEachCallIsKept() {
	pipeline, events := s.watch(func(*Pipeline) {},
		Frame{"type": "tool_call.started", "tool_call_id": "call-1", "tool": "book_flight", "turn_id": "turn-1",
			"status": "waiting", "started_at": "2026-11-18T12:00:00Z", "duration_ms": 0},
		Frame{"type": "tool_call.started", "tool_call_id": "call-2", "tool": "search_flights", "turn_id": "turn-1",
			"status": "in_progress", "started_at": "2026-11-18T12:00:00Z", "duration_ms": 0},
		Frame{"type": "tool_call.progress", "tool_call_id": "call-1", "tool": "book_flight", "turn_id": "turn-1",
			"status": "in_progress", "started_at": "2026-11-18T12:00:00Z", "duration_ms": 400,
			"approved_at": "2026-11-18T12:00:00.4Z"},
		Frame{"type": "tool_call.finished", "tool_call_id": "call-1", "tool": "book_flight", "turn_id": "turn-1",
			"status": "success", "started_at": "2026-11-18T12:00:00Z", "duration_ms": 1200,
			"approved_at": "2026-11-18T12:00:00.4Z"},
	)

	booked, ok := pipeline.ToolCall("call-1")
	s.Require().True(ok)
	s.Equal(acceleration.ToolCallStatus("success"), booked.Status)
	s.EqualValues(1200, booked.DurationMs)
	s.Require().NotNil(booked.ApprovedAt)
	searching, ok := pipeline.ToolCall("call-2")
	s.Require().True(ok)
	s.Equal(acceleration.ToolCallStatus("in_progress"), searching.Status)
	s.Len(pipeline.ToolCalls(), 2)
	s.Require().Len(events, 4)
	s.Require().NotNil(events[3].ToolCall)
	s.Equal("call-1", events[3].ToolCall.ToolCallId)
}

func (s *ToolCallStateSuite) TestAHandlerHearsItsKindUntilItIsTurnedOff() {
	var heard []string
	var finished int
	s.watch(func(pipeline *Pipeline) {
		var off func()
		off = pipeline.On("tool_call.progress", func(event Event) {
			heard = append(heard, *event.ToolCall.Progress)
			off()
		})
		pipeline.On("tool_call.finished", func(Event) { finished++ })
	},
		Frame{"type": "tool_call.progress", "tool_call_id": "call-1", "tool": "search_flights", "status": "in_progress",
			"started_at": "2026-11-18T12:00:00Z", "duration_ms": 100, "progress": "Searching for flights"},
		Frame{"type": "tool_call.progress", "tool_call_id": "call-1", "tool": "search_flights", "status": "in_progress",
			"started_at": "2026-11-18T12:00:00Z", "duration_ms": 200, "progress": "Comparing prices"},
		Frame{"type": "tool_call.finished", "tool_call_id": "call-1", "tool": "search_flights", "status": "success",
			"started_at": "2026-11-18T12:00:00Z", "duration_ms": 300},
	)

	s.Equal([]string{"Searching for flights"}, heard)
	s.Equal(1, finished)
}
