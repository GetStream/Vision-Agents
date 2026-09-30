//go:build integration

package api

import (
	"net/http"
	"net/url"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type SessionListSuite struct {
	RouterSuite
}

func TestSessionListSuite(t *testing.T) {
	suite.Run(t, new(SessionListSuite))
}

func (s *SessionListSuite) TestAUserIsListedTheirOwnSessionsAndTheBackendEveryone() {
	alice := s.createSession(s.client, textSession(nil))
	bob := s.createSession(s.user("bob"), textSession(nil))
	server := s.createSession(s.backend, textSession(nil))

	s.Equal([]string{alice.Id}, ids(s.listSessions(s.client, "").Items))
	s.ElementsMatch([]string{alice.Id, bob.Id, server.Id}, ids(s.listSessions(s.backend, "").Items))
}

func (s *SessionListSuite) TestPagesFollowTheCursorNewestFirstWithoutRepeatsOrGaps() {
	var created []string
	for range 5 {
		created = append(created, s.createSession(s.backend, textSession(nil)).Id)
	}
	slices.Reverse(created)

	var listed []string
	var sizes []int
	query := "?limit=2"
	for {
		page := s.listSessions(s.backend, query)
		listed = append(listed, ids(page.Items)...)
		sizes = append(sizes, len(page.Items))
		if !page.HasMore {
			s.Nil(page.NextCursor, "the last page has nowhere to go next")
			break
		}
		s.Require().NotNil(page.NextCursor)
		query = "?limit=2&cursor=" + url.QueryEscape(*page.NextCursor)
	}

	s.Equal([]int{2, 2, 1}, sizes)
	s.Equal(created, listed)
}

func (s *SessionListSuite) TestFiltersNarrowTheList() {
	support, sales := "support", "sales"
	ticketed := textSession(nil)
	ticketed.Project = &support
	ticketed.Custom = &map[string]any{"ticket": "4721"}
	wanted := s.createSession(s.backend, ticketed)

	other := textSession(nil)
	other.Project = &sales
	s.createSession(s.backend, other)

	s.Equal([]string{wanted.Id}, ids(s.listSessions(s.backend, "?project=support").Items))
	s.Equal([]string{wanted.Id},
		ids(s.listSessions(s.backend, "?custom="+url.QueryEscape(`{"ticket":"4721"}`)).Items))
}

func (s *SessionListSuite) TestAClosedSessionIsListedAsClosedRatherThanRunning() {
	closing := s.createSession(s.backend, textSession(nil))
	running := s.createSession(s.backend, textSession(nil))

	s.Require().Equal(http.StatusNoContent,
		s.backend.do(http.MethodDelete, "/v1/agents/sessions/"+closing.Id, nil, nil))

	s.Require().Eventually(func() bool {
		return slices.Equal([]string{closing.Id}, ids(s.listSessions(s.backend, "?state=closed").Items))
	}, 5*time.Second, 20*time.Millisecond, "the closed session was never listed as closed")
	s.Equal([]string{running.Id}, ids(s.listSessions(s.backend, "?state=running").Items))
}

func (s *SessionListSuite) TestACursorTheRouterNeverIssuedIsRefused() {
	s.Equal(http.StatusBadRequest, s.backend.do(http.MethodGet, "/v1/agents/sessions?cursor=nonsense", nil, nil))
}

func (s *SessionListSuite) TestNobodyIsListedAnythingWithoutCredentials() {
	s.createSession(s.backend, textSession(nil))

	s.Equal(http.StatusUnauthorized, s.anonymous.do(http.MethodGet, "/v1/agents/sessions", nil, nil))
}
