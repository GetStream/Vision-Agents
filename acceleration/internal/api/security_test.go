//go:build integration

package api

import (
	"net/http"
	"slices"
)

// callerKind is one sort of caller an endpoint can be asked by.
type callerKind string

const (
	unauthenticated callerKind = "unauthenticated"
	anonymous       callerKind = "anonymous"
	guest           callerKind = "guest"
	user            callerKind = "user"
	server          callerKind = "server"
)

// posture is which sorts of caller an endpoint admits.
type posture []callerKind

var (
	// anyAppCaller is an endpoint anyone holding the app's key may call, including a device
	// that has not signed anybody in.
	anyAppCaller = posture{anonymous, guest, user, server}
	// serverOnly is an endpoint only the app's own backend may call.
	serverOnly = posture{server}
)

// assertPosture calls an endpoint as every sort of caller. The ones the posture admits
// must get through; the rest are refused, an unauthenticated caller with a 401 and anyone
// else with a 403.
func (s *RouterSuite) assertPosture(admits posture, call func(as *testClient) int) {
	for _, as := range []*testClient{
		s.unauthenticatedClient, s.anonymousClient, s.guestClient, s.client, s.serverClient,
	} {
		status := call(as)
		switch {
		case slices.Contains(admits, as.kind):
			s.True(status >= 200 && status < 300, "%s is admitted, and was answered %d", as.kind, status)
		case as.kind == unauthenticated:
			s.Equal(http.StatusUnauthorized, status, "%s is refused", as.kind)
		default:
			s.Equal(http.StatusForbidden, status, "%s is refused", as.kind)
		}
	}
}

// assertHiddenFromOtherApps checks a row one customer made does not reach another's
// backend, which is told there is no such row rather than that it may not have it.
//
// The call is made as the server of an app this test has nothing to do with, and it must
// answer 404: anything else, including a 403, tells a stranger the id is real.
func (s *RouterSuite) assertHiddenFromOtherApps(call func(as *testClient) int) {
	status := call(s.data.backendOfAnotherApp())
	s.Equal(http.StatusNotFound, status, "another app is told there is no such row")
}

// assertOwnedBy checks a session belongs to owner: the owner reaches it, and neither another
// user nor an anonymous caller claiming the owner's name does.
func (s *RouterSuite) assertOwnedBy(sessionID string, owner *testClient) {
	s.assertReaches(owner, sessionID)
	s.assertDoesNotReach(s.data.createUser(), sessionID)
	if owner.kind != anonymous {
		s.assertDoesNotReach(s.data.claiming(owner.userID), sessionID)
	}
}

// assertOwnedByNobody checks a session belongs to no user: the backend reaches it and an
// end user does not.
func (s *RouterSuite) assertOwnedByNobody(sessionID string) {
	s.assertReaches(s.serverClient, sessionID)
	s.assertDoesNotReach(s.client, sessionID)
}

func (s *RouterSuite) assertReaches(as *testClient, sessionID string) {
	s.Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/sessions/"+sessionID, nil, nil),
		"the %s reaches it", as.kind)
}

// assertDoesNotReach checks the caller is told there is no such session, which is also
// what an id nobody has used gets, so the answer does not confirm the id is real.
func (s *RouterSuite) assertDoesNotReach(as *testClient, sessionID string) {
	s.Equal(http.StatusNotFound, as.do(http.MethodGet, "/v1/agents/sessions/"+sessionID, nil, nil),
		"the %s is told there is no such session", as.kind)
}
