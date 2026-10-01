//go:build integration

package api

import (
	"net/http"
	"testing"
)

// SessionDeleteSuite covers deleting a session, which takes it away rather than stopping it.
type SessionDeleteSuite struct {
	RouterSuite
}

func TestSessionDeleteSuite(t *testing.T) {
	runSuite(t, new(SessionDeleteSuite))
}

func (s *SessionDeleteSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionDeleteSuite) TestADeletedRunningSessionIsGone() {
	opened := s.client.createSession(textSession(nil))

	s.client.deleteSession(opened.Id)

	s.assertDoesNotReach(s.client, opened.Id)
	s.NotContains(ids(s.client.querySessions(SessionQuery{}).Items), opened.Id)
}

func (s *SessionDeleteSuite) TestAStoppedSessionCanStillBeDeleted() {
	opened := s.client.createSession(textSession(nil))
	s.client.stopSession(opened.Id)
	s.Require().Contains(ids(s.client.querySessions(SessionQuery{}).Items), opened.Id)

	s.client.deleteSession(opened.Id)

	s.NotContains(ids(s.client.querySessions(SessionQuery{}).Items), opened.Id)
}

func (s *SessionDeleteSuite) TestADeletedSessionCannotBeDeletedTwice() {
	opened := s.client.createSession(textSession(nil))
	s.client.deleteSession(opened.Id)

	s.Equal(http.StatusNotFound, s.client.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))
}

func (s *SessionDeleteSuite) TestAnotherUserCannotDeleteASession() {
	opened := s.client.createSession(textSession(nil))
	s.client.stopSession(opened.Id)

	s.Equal(http.StatusNotFound,
		s.data.createUser().do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil))
	s.Contains(ids(s.client.querySessions(SessionQuery{}).Items), opened.Id)
}

func (s *SessionDeleteSuite) TestAnotherAppCannotDeleteASession() {
	opened := s.client.createSession(textSession(nil))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id, nil, nil)
	})
	s.assertReaches(s.client, opened.Id)
}

func (s *SessionDeleteSuite) TestTheAppsBackendMayDeleteAUsersSession() {
	opened := s.client.createSession(textSession(nil))

	s.serverClient.deleteSession(opened.Id)

	s.assertDoesNotReach(s.client, opened.Id)
}
