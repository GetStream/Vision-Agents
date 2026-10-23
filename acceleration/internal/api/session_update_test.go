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

// The row is written behind the conversation, so a session stopped while the writer is behind
// has left the live set before its row exists. The test holds another session's row FOR
// UPDATE so the writer stalls on that session's rename, ahead of the stopped one's row.
func (s *SessionUpdateSuite) TestAUserRenamesASessionThatEndedBeforeItsRowWasWritten() {
	ctx := context.Background()
	earlier := s.client.createSession(textSession(nil))
	s.client.querySessions(SessionQuery{}) // lists only once the writer has written earlier's row
	tx, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = tx.Rollback() }()
	_, err = tx.ExecContext(ctx, "SELECT 1 FROM agent_sessions WHERE id = ? FOR UPDATE", earlier.Id)
	s.Require().NoError(err)
	s.client.updateSession(earlier.Id, UpdateSessionRequest{Title: pointerTo("Stalls the writer")})
	s.Require().Eventually(func() bool {
		var waiting int
		_ = s.store.DB().QueryRowContext(ctx, "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "+
			"AND wait_event_type = 'Lock' AND query LIKE 'UPDATE \"agent_sessions\"%'").Scan(&waiting)
		return waiting > 0
	}, 10*time.Second, 10*time.Millisecond)

	opened := s.client.createSession(textSession(nil))
	s.client.stopSession(opened.Id)
	type answer struct {
		status  int
		renamed Session
	}
	answered := make(chan answer, 1)
	go func() {
		var renamed Session
		status := s.client.do(http.MethodPatch, "/v1/agents/sessions/"+opened.Id,
			UpdateSessionRequest{Title: pointerTo("Kept")}, &renamed)
		answered <- answer{status, renamed}
	}()
	// Well under recordWriteTimeout (5s, internal/session/records.go), the writer's own way out
	// of the lock, so only the rename waiting for the writer keeps it unanswered this long.
	s.Never(func() bool { return len(answered) > 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the rename answered before the writer had written the session's row")
	s.Require().NoError(tx.Commit())

	got := <-answered
	s.Equal(http.StatusOK, got.status)
	s.Equal("Kept", value(got.renamed.Title))
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
