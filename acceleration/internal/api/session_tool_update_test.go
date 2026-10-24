//go:build integration

package api

func (s *SessionEventsSuite) TestAStatusForARunningCallReachesEveryWatcher() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("")
	s.respondFor(opened.Id)
	asked := s.await(backend, "tool_call")

	s.Require().NoError(backend.WriteJSON(map[string]any{
		"type": "tool_update", "tool_call_id": asked["id"], "text": "searching 3 airlines",
	}))

	shown := s.await(browser, "tool_updated")
	s.Equal(asked["id"], shown["id"])
	s.Equal("searching 3 airlines", shown["text"])
	s.Equal(false, shown["verbatim"])
}

func (s *SessionEventsSuite) TestWordsARunningCallAsksToSayArePostedInWriting() {
	opened := s.withATool()
	watching := s.watch(opened.Id)
	s.respond(opened.Id)
	asked := s.await(watching, "tool_call")

	s.Require().NoError(watching.WriteJSON(map[string]any{
		"type": "tool_update", "tool_call_id": asked["id"], "text": "Got it, one moment.", "verbatim": true,
	}))

	s.Equal(true, s.await(watching, "tool_updated")["verbatim"])
	s.Equal("Got it, one moment.", s.await(watching, "responded")["text"])
}

func (s *SessionEventsSuite) TestAWatcherThatDoesNotRunTheCallCannotUpdateIt() {
	opened, backend, browser := s.backendsToolWatchedByItsUser("")
	s.respondFor(opened.Id)
	asked := s.await(browser, "tool_call")
	s.await(backend, "tool_call")

	s.Require().NoError(browser.WriteJSON(map[string]any{
		"type": "tool_update", "tool_call_id": asked["id"], "text": "from the browser",
	}))
	s.Require().NoError(backend.WriteJSON(map[string]any{
		"type": "tool_result", "tool_call_id": asked["id"], "output": "it ships tomorrow",
	}))

	for {
		frame := s.next(backend)
		s.NotEqual("tool_updated", frame["type"], "a browser updated a call it does not run")
		if frame["type"] == "tool_ran" {
			return
		}
	}
}
