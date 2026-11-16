//go:build integration

package api

import (
	"database/sql"
	"errors"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// UsersSuite covers who gets written down as an end user of an app, which every
// authenticated request decides before the handler it was sent to runs.
type UsersSuite struct {
	RouterSuite
}

func TestUsersSuite(t *testing.T) {
	runSuite(t, new(UsersSuite))
}

func (s *UsersSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// recorded is the kind a caller was written down as, and whether they were at all.
func (s *UsersSuite) recorded(userID string) (string, bool) {
	var kind string
	err := s.store.DB().QueryRowContext(s.T().Context(),
		"SELECT kind FROM users WHERE customer_id = ? AND id = ?", s.customerID(), userID).Scan(&kind)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false
	}
	s.Require().NoError(err)
	return kind, true
}

func (s *UsersSuite) TestASignedInCallerIsWrittenDownAsAUserOfTheApp() {
	caller := s.data.createUser()
	caller.querySessions(SessionQuery{})

	kind, found := s.recorded(caller.userID)
	s.True(found, "an app cannot be told who its users are if only its guests are recorded")
	s.Equal(store.UserKindAuthenticated, kind)
}

func (s *UsersSuite) TestAGuestIsWrittenDownAsOne() {
	guest := s.data.createGuest()
	guest.querySessions(SessionQuery{})

	kind, found := s.recorded(guest.userID)
	s.True(found)
	s.Equal(store.UserKindGuest, kind, "only a guest is claimable, so the kind has to survive")
}

func (s *UsersSuite) TestAnAnonymousCallerIsNotWrittenDown() {
	// The name they go by is a claim nobody checked, so a row would record what the
	// caller asked to be called rather than who they are.
	anonymous := s.data.createAnonymous()
	anonymous.querySessions(SessionQuery{})

	_, found := s.recorded(anonymous.userID)
	s.False(found)
}

func (s *UsersSuite) TestTheUserABackendActsForIsWrittenDown() {
	owner := s.data.createUser()
	s.serverClient.actingFor(owner).querySessions(SessionQuery{})

	kind, found := s.recorded(owner.userID)
	s.True(found, "a backend naming one of its users holds the secret, so it has nothing to gain by lying")
	s.Equal(store.UserKindAuthenticated, kind)
}

func (s *UsersSuite) TestAnAccountIsNotRecordedAsAUserOfAnotherApp() {
	caller := s.data.createUser()
	caller.querySessions(SessionQuery{})
	mine := s.customerID()

	s.useApp(s.data.createApp())
	_, found := s.recorded(caller.userID)

	s.False(found, "a user id belongs to the app that named it")
	s.NotEqual(mine, s.customerID())
}
