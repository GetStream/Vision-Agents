//go:build integration

package api

import (
	"net/http"
	"testing"
	"time"

	"github.com/google/uuid"
)

type SessionCreateSuite struct {
	RouterSuite
}

func TestSessionCreateSuite(t *testing.T) {
	runSuite(t, new(SessionCreateSuite))
}

func (s *SessionCreateSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *SessionCreateSuite) TestASessionIsHeldByTheIdTheDeviceChose() {
	id := s.utils.uuid()

	s.Equal(id, s.client.createSession(textSession(&id)).Id)
	s.Equal(id, s.client.getSession(id).Id)
}

func (s *SessionCreateSuite) TestASessionIsHeldByTheIdTheBackendChose() {
	id := s.utils.uuid()

	s.Equal(id, s.serverClient.createSession(textSession(&id)).Id)
	s.Equal(id, s.serverClient.getSession(id).Id)
}

func (s *SessionCreateSuite) TestASessionWithoutAnIdIsGivenAUUIDv7() {
	created := s.client.createSession(textSession(nil))

	parsed, err := uuid.Parse(created.Id)
	s.Require().NoError(err)
	s.Equal(uuid.Version(7), parsed.Version())
}

func (s *SessionCreateSuite) TestAnIdThatIsNotAUUIDIsRefused() {
	id := "my-session"

	s.Equal(http.StatusBadRequest, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil))
}

func (s *SessionCreateSuite) TestAnIdAnotherSessionHasIsRefused() {
	id, project := s.utils.uuid(), s.utils.uuid()
	request := textSession(&id)
	request.ProjectId = &project
	s.client.createSession(request)

	s.Equal(http.StatusConflict, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil),
		"the session is running")

	s.client.stopSession(id)
	s.Require().Eventually(func() bool {
		for _, one := range s.client.querySessions(inProject(project)).Items {
			if one.Id == id {
				return one.ClosedAt != nil
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond, "the closed session was never written down")

	s.Equal(http.StatusConflict, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil),
		"the session has ended and its row still holds the id")
}

func (s *SessionCreateSuite) TestASessionIsCreatedListedUpdatedAndRead() {
	id, project, title := s.utils.uuid(), s.utils.uuid(), "Refunds"
	request := textSession(&id)
	request.Title, request.ProjectId = &title, &project
	s.Equal(title, value(s.serverClient.createSession(request).Title))

	s.Equal([]string{id}, ids(s.serverClient.querySessions(inProject(project)).Items))

	renamed, description := "Refunds, resolved", "The customer was owed two pennies."
	updated := s.serverClient.updateSession(id, UpdateSessionRequest{Title: &renamed, Description: &description})
	s.Equal(renamed, value(updated.Title))
	s.Equal(description, value(updated.Description))

	read := s.serverClient.getSession(id)
	s.Equal(renamed, value(read.Title))
	s.Equal(description, value(read.Description))
	s.Equal(Live, read.State)
}

func (s *SessionCreateSuite) TestAnyoneHoldingTheAppsKeyMayCreateASession() {
	s.assertPosture(anyAppCaller, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil)
	})
}

func (s *SessionCreateSuite) TestASessionBelongsToTheUserWhoCreatedIt() {
	for _, owner := range []*testClient{s.client, s.guestClient, s.anonymousClient} {
		s.Run(string(owner.kind), func() {
			s.assertOwnedBy(owner.createSession(textSession(nil)).Id, owner)
		})
	}
}

func (s *SessionCreateSuite) TestABackendCreatesASessionForTheUserItNames() {
	owner := s.data.createUser()

	s.assertOwnedBy(s.serverClient.actingFor(owner).createSession(textSession(nil)).Id, owner)
}

func (s *SessionCreateSuite) TestABackendNamingNobodyCreatesASessionNoUserOwns() {
	s.assertOwnedByNobody(s.serverClient.createSession(textSession(nil)).Id)
}
