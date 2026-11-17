//go:build integration

package api

import "time"

func (s *SessionEventsSuite) TestHoldingARunningCallIsAnsweredHeld() {
	opened := s.withATool()
	watching := s.watch(opened.Id)
	s.respond(opened.Id)
	asked := s.await(watching, "tool_call")

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_hold", "tool_call_id": asked["id"]}))

	s.Equal(asked["id"], s.await(watching, "tool_held")["id"])
	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))
	s.Equal("it ships tomorrow", s.await(watching, "tool_ran")["result"], "a held call is answered as any other")
}

func (s *SessionEventsSuite) TestHoldingACallThatIsNotRunningIsAnsweredCancelled() {
	opened := s.withATool()
	watching := s.watch(opened.Id)

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "tool_hold", "tool_call_id": "call-nobody-made"}))

	s.Equal("call-nobody-made", s.await(watching, "tool_cancel")["id"])
}

func (s *SessionEventsSuite) TestAWatcherThatDoesNotRunTheCallCannotHoldIt() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("")
	s.respondFor(opened.Id)
	asked := s.await(browser, "tool_call")
	s.await(backend, "tool_call")

	s.Require().NoError(browser.WriteJSON(map[string]any{"type": "tool_hold", "tool_call_id": asked["id"]}))
	s.Require().NoError(backend.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))

	for {
		frame := s.next(backend)
		s.NotEqual("tool_held", frame["type"], "a browser held a call it does not run")
		if frame["type"] == "tool_ran" {
			return
		}
	}
}

// next is the next frame on the socket, whatever its type.
func (s *SessionEventsSuite) next(connection interface {
	SetReadDeadline(time.Time) error
	ReadJSON(any) error
}) map[string]any {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	var received map[string]any
	s.Require().NoError(connection.ReadJSON(&received), "the socket closed")
	return received
}
