//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"
)

// SessionUpdateSuite covers what an end user's device may change about a session.
type SessionUpdateSuite struct {
	RouterSuite
}

func TestSessionUpdateSuite(t *testing.T) {
	runSuite(t, new(SessionUpdateSuite))
}

func (s *SessionUpdateSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *SessionUpdateSuite) TestAUserRenamesTheirOwnSession() {
	opened := s.client.createSession(textSession(nil))

	updated := s.client.updateSession(opened.Id, UpdateSessionRequest{
		Title:       pointerTo("Pricing"),
		Description: pointerTo("Asked twice"),
		Custom:      &map[string]any{"pinned": true},
	})

	s.Equal("Pricing", value(updated.Title))
	read := s.client.getSession(opened.Id)
	s.Equal("Asked twice", value(read.Description))
	s.Equal(map[string]any{"pinned": true}, value(read.Custom))
}

func (s *SessionUpdateSuite) TestAUserRenamesASessionThatEnded() {
	opened := s.client.createSession(textSession(nil))
	s.client.stopSession(opened.Id)

	s.Equal("Kept", value(s.client.updateSession(opened.Id, UpdateSessionRequest{Title: pointerTo("Kept")}).Title))
}

// stallTheRecordWriter holds another session's row FOR UPDATE so the writer stalls on that
// session's rename, ahead of every row queued after it, until the returned release runs
// (or recordWriteTimeout, 5s in internal/session/records.go, gives the writer its own way out).
func (s *RouterSuite) stallTheRecordWriter() (release func()) {
	ctx := context.Background()
	earlier := s.client.createSession(textSession(nil))
	s.client.querySessions(SessionQuery{}) // lists only once the writer has written earlier's row
	tx, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = tx.Rollback() })
	_, err = tx.ExecContext(ctx, "SELECT 1 FROM agent_sessions WHERE id = ? FOR UPDATE", earlier.Id)
	s.Require().NoError(err)
	s.client.updateSession(earlier.Id, UpdateSessionRequest{Title: pointerTo("Stalls the writer")})
	s.Require().Eventually(func() bool {
		var waiting int
		_ = s.store.DB().QueryRowContext(ctx, "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "+
			"AND wait_event_type = 'Lock' AND query LIKE 'UPDATE \"agent_sessions\"%'").Scan(&waiting)
		return waiting > 0
	}, 10*time.Second, 10*time.Millisecond)
	return func() { s.Require().NoError(tx.Commit()) }
}

// backgroundAnswer is what a request run by askInBackground was answered with.
type backgroundAnswer struct {
	status int
	body   Session
}

// askInBackground sends the request off the test goroutine; a test waits on the channel with a
// timeout, since a request that fails its Require there never sends.
func (s *RouterSuite) askInBackground(method, path string, body any) <-chan backgroundAnswer {
	answered := make(chan backgroundAnswer, 1)
	go func() {
		var got Session
		status := s.client.do(method, path, body, &got)
		answered <- backgroundAnswer{status, got}
	}()
	return answered
}

// The row is written behind the conversation, so a session stopped while the writer is behind
// has left the live set before its row exists. The writer is stalled ahead of the stopped
// session's row, and a live session still answers at once.
func (s *SessionUpdateSuite) TestAUserRenamesASessionThatEndedBeforeItsRowWasWritten() {
	release := s.stallTheRecordWriter()
	live := s.client.createSession(textSession(nil))
	opened := s.client.createSession(textSession(nil))

	began := time.Now()
	s.Equal("Live", value(s.client.updateSession(live.Id, UpdateSessionRequest{Title: pointerTo("Live")}).Title))
	// Well under recordWriteTimeout, so only a live rename waiting for the writer takes this long.
	s.Less(time.Since(began), time.Second, "a live session's rename waited for the writer")

	s.client.stopSession(opened.Id)
	answered := s.askInBackground(http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
		UpdateSessionRequest{Title: pointerTo("Kept")})
	// Well under recordWriteTimeout, the writer's own way out of the lock, so only the rename
	// waiting for the writer keeps it unanswered this long.
	s.Never(func() bool { return len(answered) > 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the rename answered before the writer had written the session's row")
	release()

	select {
	case got := <-answered:
		s.Equal(http.StatusOK, got.status)
		s.Equal("Kept", value(got.body.Title))
	case <-time.After(10 * time.Second):
		s.FailNow("the rename never answered")
	}
}

// A session that ended with its row already written is read at once: only a read that missed
// waits for the writer.
func (s *SessionUpdateSuite) TestAUserRenamesASessionThatEndedWhileTheWriterIsBehind() {
	opened := s.client.createSession(textSession(nil))
	s.client.querySessions(SessionQuery{}) // lists only once the writer has written opened's row
	release := s.stallTheRecordWriter()
	defer release()
	s.client.stopSession(opened.Id)

	began := time.Now()
	s.Equal("Kept", value(s.client.updateSession(opened.Id, UpdateSessionRequest{Title: pointerTo("Kept")}).Title))
	s.Less(time.Since(began), time.Second, "the rename waited for the writer although the row was there")
}

func (s *SessionUpdateSuite) TestADeviceCannotMoveASessionOntoAnotherModel() {
	opened := s.client.createSession(textSession(nil))

	s.Equal(http.StatusForbidden, s.client.do(http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
		UpdateSessionRequest{Title: pointerTo("Renamed"), Llm: pointerTo("vision/vision-model")}, nil))
	s.Nil(s.client.getSession(opened.Id).Title, "a refused update changes nothing")
}

// A device renaming its session is not given its brief, and the backend still is.
func (s *SessionUpdateSuite) TestOnlyTheBackendReadsTheBriefBackFromARename() {
	owner := s.data.createUser()
	request, brief := s.data.briefed(textSession(nil))
	opened := s.serverClient.actingFor(owner).createSession(request)
	renamed := UpdateSessionRequest{Title: pointerTo("Practice")}

	s.Nil(owner.updateSession(opened.Id, renamed).Instructions)
	s.Equal(brief, value(s.serverClient.actingFor(owner).updateSession(opened.Id, renamed).Instructions))
}

func (s *SessionUpdateSuite) TestAnotherUserCannotRenameASession() {
	opened := s.client.createSession(textSession(nil))

	s.Equal(http.StatusNotFound, s.data.createUser().do(http.MethodPatch,
		"/v1/agents/sessions/"+opened.Id, UpdateSessionRequest{Title: pointerTo("Mine now")}, nil))
}
