//go:build integration

package api

import (
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// MemoriesSuite covers taking back what an agent remembers: by user, by session, and what
// ending a session leaves behind.
type MemoriesSuite struct {
	RouterSuite
}

func TestMemoriesSuite(t *testing.T) {
	runSuite(t, new(MemoriesSuite))
}

func (s *MemoriesSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *MemoriesSuite) TestTruncatingAUserForgetsOnlyThatUsersMemories() {
	theirs := s.remembered(s.utils.uuid())
	somebodyElses := s.remembered(s.utils.uuid())

	s.Equal(http.StatusNoContent, s.serverClient.do(
		http.MethodDelete, "/v1/agents/users/"+theirs.UserID+"/memories", nil, nil))

	s.False(s.isRemembered(theirs))
	s.True(s.isRemembered(somebodyElses), "another user's memories are kept")
}

func (s *MemoriesSuite) TestAnotherAppsSameUserIsNotForgotten() {
	// A user id is one app's own name for somebody, so two apps may use the same one.
	user := s.utils.uuid()
	mine := s.remembered(user)
	elsewhere := memory.Scope{AppID: s.utils.uuid(), UserID: user, RunID: s.utils.uuid()}
	s.remember(elsewhere)

	s.Equal(http.StatusNoContent, s.serverClient.do(
		http.MethodDelete, "/v1/agents/users/"+user+"/memories", nil, nil))

	s.False(s.isRemembered(mine))
	s.True(s.isRemembered(elsewhere))
}

func (s *MemoriesSuite) TestForgettingASessionKeepsTheUsersOtherMemories() {
	opened := s.client.createSession(textSession(nil))
	inTheSession := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: opened.Id}
	earlier := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: s.utils.uuid()}
	s.remember(inTheSession)
	s.remember(earlier)

	s.Equal(http.StatusNoContent, s.serverClient.do(
		http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/memories", nil, nil))

	s.False(s.isRemembered(inTheSession))
	s.True(s.isRemembered(earlier))
}

func (s *MemoriesSuite) TestAnotherAppsSessionMemoriesCannotBeForgotten() {
	opened := s.client.createSession(textSession(nil))
	learned := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: opened.Id}
	s.remember(learned)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/sessions/"+opened.Id+"/memories", nil, nil)
	})

	s.True(s.isRemembered(learned), "a session id says nothing about whose it is")
}

func (s *MemoriesSuite) TestStoppingASessionKeepsWhatItRemembered() {
	// Memory is what the next conversation starts from, so stopping one is not forgetting
	// it.
	opened := s.client.createSession(textSession(nil))
	learned := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: opened.Id}
	s.remember(learned)

	s.client.stopSession(opened.Id)

	s.True(s.isRemembered(learned))
}

func (s *MemoriesSuite) TestDeletingASessionForgetsOnlyWhatItRemembered() {
	opened := s.client.createSession(textSession(nil))
	learned := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: opened.Id}
	earlier := memory.Scope{AppID: s.customerID(), UserID: s.client.userID, RunID: s.utils.uuid()}
	s.remember(learned)
	s.remember(earlier)

	s.client.deleteSession(opened.Id)

	s.False(s.isRemembered(learned))
	s.True(s.isRemembered(earlier))
}

func (s *MemoriesSuite) TestOnlyTheAppsOwnBackendMayForgetAUser() {
	user := s.utils.uuid()

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/users/"+user+"/memories", nil, nil)
	})
}

// remembered is a memory about a user of the suite's app, as one of its sessions would
// have left behind.
func (s *MemoriesSuite) remembered(user string) memory.Scope {
	scope := memory.Scope{AppID: s.customerID(), UserID: user, RunID: s.utils.uuid()}
	s.remember(scope)
	return scope
}

func (s *MemoriesSuite) remember(scope memory.Scope) {
	s.memories.mu.Lock()
	defer s.memories.mu.Unlock()
	s.memories.kept = append(s.memories.kept, scope)
}

func (s *MemoriesSuite) isRemembered(scope memory.Scope) bool {
	for _, kept := range s.memories.remaining() {
		if kept.AppID == scope.AppID && kept.UserID == scope.UserID && kept.RunID == scope.RunID {
			return true
		}
	}
	return false
}
