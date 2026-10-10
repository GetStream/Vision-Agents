//go:build integration

package api

import (
	"net/http"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// SessionOwnershipSuite covers whose conversation a session is: who reaches it, who is
// told there is no such thing, and what a name is worth without a token behind it.
type SessionOwnershipSuite struct {
	RouterSuite
}

func TestSessionOwnershipSuite(t *testing.T) {
	runSuite(t, new(SessionOwnershipSuite))
}

func (s *SessionOwnershipSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionOwnershipSuite) TestTheUserWhoOpenedASessionReachesItAndMayEndIt() {
	alice := s.data.createUser()

	opened := alice.createSession(textSession(nil))

	s.assertReaches(alice, opened.Id)
	alice.stopSession(opened.Id)
}

func (s *SessionOwnershipSuite) TestASessionClosedOverItsSocketStopsBeingOneOfTheirs() {
	// The socket is how every SDK closes a conversation it is holding. Closing this way
	// once ended the conversation and left it in the manager, which listed it as live for
	// as long as the router ran.
	alice := s.data.createUser()
	opened := alice.createSession(s.onACall())
	watching := alice.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "close"}))

	// The close is applied on the reading goroutine, so this waits for it rather than
	// assuming the write was enough.
	s.Require().Eventually(func() bool {
		for _, listed := range alice.querySessions(SessionQuery{}).Items {
			if listed.Id == opened.Id {
				return listed.State != Live
			}
		}
		return false
	}, settleFor, 20*time.Millisecond, "it is still listed as a call they are on")

	s.Equal(http.StatusNotFound,
		alice.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/stop", nil, nil),
		"a session that closed itself is over rather than stoppable twice")
}

func (s *SessionOwnershipSuite) TestWritingIntoASessionIsNotForDevicesEvenTheOwners() {
	// Ownership is not the only thing between a device and a session. Everything the spec
	// does not open is server-side whoever asks, so the owner is refused these too, and
	// with a 403 rather than the 404 a stranger gets: what is being withheld is the
	// operation and not the session.
	alice := s.data.createUser()
	opened := alice.createSession(textSession(nil))

	s.assertReaches(alice, opened.Id)
	s.Equal(http.StatusForbidden,
		alice.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/say",
			SayRequest{Text: "hello"}, nil),
		"putting words in the agent's mouth")
	s.Equal(http.StatusForbidden,
		alice.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
			SayRequest{Text: "hello"}, nil),
		"making the agent answer")
}

func (s *SessionOwnershipSuite) TestOneUsersConversationIsOutOfAnothersReach() {
	// The whole point of the token: two signed-in users of the same app, and neither can
	// read what the other said.
	opened := s.data.createUser().createSession(textSession(nil))

	s.assertDoesNotReach(s.data.createUser(), opened.Id)
}

func (s *SessionOwnershipSuite) TestClaimingAUsersNameWithoutATokenReachesNothing() {
	// A user id is only worth what proved it. An anonymous caller may name itself
	// anything, so if the name were the whole of the identity, typing somebody else's
	// into a header would be enough to read their conversation.
	alice := s.data.createUser()
	opened := alice.createSession(textSession(nil))

	s.assertDoesNotReach(s.data.claiming(alice.userID), opened.Id)
}

func (s *SessionOwnershipSuite) TestAGuestIsNotTheSignedInUserOfTheSameName() {
	// Guest ids are issued per device and a customer's user ids are their own, so the two
	// namespaces can collide. Nothing in a token tells them apart, which is why the
	// declaration does.
	alice := s.data.createUser()
	opened := alice.createSession(textSession(nil))

	s.assertDoesNotReach(s.guestNamed(alice.userID), opened.Id)
}

func (s *SessionOwnershipSuite) TestAGuestReachesItsOwnConversationAndNobodyElsesReachesIt() {
	guest := s.guestNamed(s.utils.uuid())

	opened := guest.createSession(textSession(nil))

	s.assertReaches(guest, opened.Id)
	s.assertDoesNotReach(s.guestNamed(s.utils.uuid()), opened.Id)
	s.assertDoesNotReach(s.data.signedInAs(guest.userID), opened.Id)
}

func (s *SessionOwnershipSuite) TestAnAnonymousCallerReachesTheSessionItOpenedAndNotAnothers() {
	// Anonymous is a supported way to hold a conversation, not a way to be refused one:
	// the name is unverified, so it is kept apart from every verified user of the same
	// name, and from other anonymous callers by being a different name.
	someone := s.data.createAnonymous()

	opened := someone.createSession(textSession(nil))

	s.assertReaches(someone, opened.Id)
	s.assertDoesNotReach(s.data.createAnonymous(), opened.Id)
}

func (s *SessionOwnershipSuite) TestAnAnonymousCallerThatNamesNobodyIsListedNothing() {
	// It has the session id, so it can go on with the conversation it opened. What it
	// cannot have is a listing, because every caller naming nobody is the same owner and
	// a listing would hand one stranger another's conversation.
	nameless := s.data.claiming("")

	opened := nameless.createSession(textSession(nil))

	s.assertReaches(nameless, opened.Id)
	s.Empty(nameless.querySessions(SessionQuery{}).Items)
}

func (s *SessionOwnershipSuite) TestAServerTokenInASocketsQueryStringIsNotABackend() {
	// A socket's credentials travel in the query string, where a URL is copied into places
	// a header never goes. If that were enough to be server-side, a leaked server token
	// would be every session the customer has; without the header it is a caller naming
	// nobody, which reaches nothing it did not open.
	opened := s.data.createUser().createSession(textSession(nil))
	quiet := *s.serverClient
	quiet.header = s.serverClient.header.Clone()
	quiet.header.Del(auth.AuthTypeHeader)

	_, status := quiet.watch("/v1/agents/sessions/" + opened.Id + "/events")

	s.Equal(http.StatusNotFound, status)
}

func (s *SessionOwnershipSuite) TestTheAppsBackendEndsAUsersCallWithoutReadingIt() {
	// A call is kept like a conversation in writing: it is the user's to read, and the
	// application's to clean up after a device that went away.
	alice := s.data.createUser()
	opened := alice.createSession(s.onACall())

	s.assertDoesNotReach(s.serverClient, opened.Id)
	s.serverClient.deleteSession(opened.Id)
	s.assertDoesNotReach(alice, opened.Id)
}

func (s *SessionOwnershipSuite) TestAnotherAppsUserReachesNothingEvenByTheSameName() {
	alice := s.data.createUser()
	opened := alice.createSession(textSession(nil))

	mine := s.app
	s.app = s.data.createApp()
	stranger := s.data.signedInAs(alice.userID)
	s.app = mine

	s.assertDoesNotReach(stranger, opened.Id)
}

func (s *SessionOwnershipSuite) TestASessionWithNoTokenAtAllIsRefusedBeforeItIsOwned() {
	// The ownership rules are about telling verified callers apart. A caller with no
	// credential never gets far enough to have any.
	status, _ := s.unauthenticatedClient.call(http.MethodPost, "/v1/agents/sessions/query", nil)

	s.Equal(http.StatusUnauthorized, status)
}

// onACall is a session with an agent in a call, which records no conversation of its own.
func (s *SessionOwnershipSuite) onACall() CreateSessionRequest {
	return CreateSessionRequest{StartVoice: pointerTo(true)}
}

// guestNamed is a client holding a guest token that goes by name, for the collisions
// between a guest id and one of the customer's own user ids.
func (s *SessionOwnershipSuite) guestNamed(name string) *testClient {
	return s.signedIn(jwt.MapClaims{"user_id": name, "role": "guest"}, auth.AuthTypeJWT, guest, name)
}
