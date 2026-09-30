//go:build integration

package api

import (
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
)

type SessionCreateSuite struct {
	RouterSuite
}

func TestSessionCreateSuite(t *testing.T) {
	suite.Run(t, new(SessionCreateSuite))
}

func (s *SessionCreateSuite) TestASessionIsHeldByTheIdTheCallerChose() {
	for name, as := range map[string]testClient{"client": s.client, "server": s.backend} {
		s.Run(name, func() {
			id := uuid.Must(uuid.NewV7()).String()

			created := s.createSession(as, textSession(&id))
			s.Equal(id, created.Id)

			var read Session
			s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &read))
			s.Equal(id, read.Id)
		})
	}
}

func (s *SessionCreateSuite) TestASessionWithoutAnIdIsGivenAUUIDv7() {
	created := s.createSession(s.backend, textSession(nil))

	parsed, err := uuid.Parse(created.Id)
	s.Require().NoError(err)
	s.Equal(uuid.Version(7), parsed.Version())
}

func (s *SessionCreateSuite) TestAnIdThatIsNotAUUIDIsRefused() {
	id := "my-session"

	s.Equal(http.StatusBadRequest, s.backend.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil))
}

func (s *SessionCreateSuite) TestAnIdAnotherSessionHasIsRefused() {
	id := uuid.Must(uuid.NewV7()).String()
	s.createSession(s.backend, textSession(&id))

	s.Equal(http.StatusConflict, s.backend.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil),
		"the session is running")

	s.Require().Equal(http.StatusNoContent, s.backend.do(http.MethodDelete, "/v1/agents/sessions/"+id, nil, nil))
	s.Require().Eventually(func() bool {
		return slices.Contains(ids(s.listSessions(s.backend, "?state=closed").Items), id)
	}, 5*time.Second, 20*time.Millisecond, "the closed session was never written down")

	s.Equal(http.StatusConflict, s.backend.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil),
		"the session has ended and its row still holds the id")
}

func (s *SessionCreateSuite) TestASessionIsCreatedListedUpdatedAndRead() {
	id := uuid.Must(uuid.NewV7()).String()
	request := textSession(&id)
	title := "Refunds"
	request.Title = &title
	created := s.createSession(s.backend, request)
	s.Equal("Refunds", value(created.Title))

	s.Contains(ids(s.listSessions(s.backend, "").Items), id)

	renamed, description := "Refunds, resolved", "The customer was owed two pennies."
	var updated Session
	s.Require().Equal(http.StatusOK, s.backend.do(http.MethodPatch, "/v1/agents/sessions/"+id,
		UpdateSessionRequest{Title: &renamed, Description: &description}, &updated))
	s.Equal(renamed, value(updated.Title))
	s.Equal(description, value(updated.Description))

	var read Session
	s.Require().Equal(http.StatusOK, s.backend.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &read))
	s.Equal(id, read.Id)
	s.Equal(renamed, value(read.Title))
	s.Equal(description, value(read.Description))
	s.Equal(Live, read.State)
}

func (s *SessionCreateSuite) TestSessionsAreCreatedFromTheClientAndTheServerButNotAnonymously() {
	s.createSession(s.client, textSession(nil))
	s.createSession(s.backend, textSession(nil))

	s.Equal(http.StatusUnauthorized, s.anonymous.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil))
}
