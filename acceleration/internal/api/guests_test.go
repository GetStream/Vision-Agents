//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// GuestsSuite covers minting a guest so somebody can talk to an agent before they sign up,
// and moving their conversations onto the account they turn out to be.
//
// Minting the token itself reaches Stream, which this deployment has no real keys for, so
// what is asserted about minting is everything the router decides before it calls out.
type GuestsSuite struct {
	RouterSuite
}

func TestGuestsSuite(t *testing.T) {
	runSuite(t, new(GuestsSuite))
}

func (s *GuestsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *GuestsSuite) TestAnAppThatDoesNotAdmitGuestsRefusesToMintOne() {
	no := false
	s.useApp(s.data.createAppAdmitting(store.AppSettings{AllowGuest: &no}))

	status, _ := s.data.createAnonymous().call(http.MethodPost, "/v1/agents/guests", nil)

	s.Equal(http.StatusForbidden, status)
}

func (s *GuestsSuite) TestAnIdThatIsNotAGuestOfThisAppIsRefused() {
	// A guest naming an id could otherwise be handed a real user's conversations, and
	// whether a name belongs to one is not something this can tell apart.
	status, failure := s.data.createAnonymous().failure(http.MethodPost, "/v1/agents/guests",
		GuestUserRequest{Id: pointerTo(s.utils.uuid())})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not a guest of this app")
}

func (s *GuestsSuite) TestAGuestWhoHasSignedUpIsNotMintedAgain() {
	guest := s.recordGuest()
	s.claim(guest, s.utils.uuid())

	status, failure := s.data.createAnonymous().failure(http.MethodPost, "/v1/agents/guests",
		GuestUserRequest{Id: pointerTo(guest)})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "has been claimed")
}

func (s *GuestsSuite) TestClaimingAGuestMovesTheirConversationsOntoTheAccount() {
	guest, account := s.recordGuest(), s.data.createUser()
	talked := s.talked(guest)

	claimed := s.claim(guest, account.userID)

	s.Equal(1, claimed.SessionsMoved)
	s.Contains(ids(account.querySessions(SessionQuery{}).Items), talked,
		"the conversation is the account's now")
}

func (s *GuestsSuite) TestTheGuestNoLongerOwnsWhatTheyTalkedAbout() {
	guest := s.recordGuest()
	talked := s.talked(guest)

	s.claim(guest, s.data.createUser().userID)

	s.NotContains(ids(s.data.signedInAs(guest).querySessions(SessionQuery{}).Items), talked)
}

func (s *GuestsSuite) TestTheSameClaimArrivingTwiceIsNotAConflict() {
	// A retried request must not read as somebody else claiming the same guest.
	guest := s.recordGuest()
	account := s.utils.uuid()
	s.claim(guest, account)

	again := s.claim(guest, account)

	s.Equal(0, again.SessionsMoved, "the conversations moved the first time")
}

func (s *GuestsSuite) TestAGuestSomebodyElseClaimedIsAConflict() {
	guest := s.recordGuest()
	s.claim(guest, s.utils.uuid())

	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: guest, UserId: s.utils.uuid()})

	s.Equal(http.StatusConflict, status)
}

func (s *GuestsSuite) TestAGuestNobodyMintedCannotBeClaimed() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: s.utils.uuid(), UserId: s.utils.uuid()})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "no such guest")
}

func (s *GuestsSuite) TestAClaimHasToSayWhoIsClaimingWhom() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: s.recordGuest()})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "user id")
}

func (s *GuestsSuite) TestAnotherAppsGuestIsNotTheirsToClaim() {
	guest := s.recordGuest()

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/guests/claim",
			ClaimGuestRequest{GuestId: guest, UserId: s.utils.uuid()}, nil)
	})
}

func (s *GuestsSuite) TestOnlyTheAppsOwnBackendMayClaimAGuest() {
	// A page allowed to ask this could claim anybody's conversations by guessing an id.
	guest := s.recordGuest()

	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/agents/guests/claim",
			ClaimGuestRequest{GuestId: guest, UserId: s.utils.uuid()})
		return status
	})
}

// recordGuest is a guest of the suite's app, written the way minting one writes it. Minting
// over HTTP would reach Stream for the token.
func (s *GuestsSuite) recordGuest() string {
	id := guestPrefix + s.utils.uuid()
	s.Require().NoError(s.store.RecordGuest(context.Background(), &store.GuestUser{
		ID: id, CustomerID: s.customerID(), Name: "Guest",
	}))
	return id
}

// talked is a conversation the guest had and has finished, which is the history a claim
// moves. The row is written off the request path, so it is waited for: a claim that
// arrives before it moves nothing.
func (s *GuestsSuite) talked(guest string) string {
	talking := s.data.signedInAs(guest)
	opened := talking.createSession(textSession(nil))
	talking.stopSession(opened.Id)
	s.Require().Eventually(func() bool {
		return slices.Contains(ids(talking.querySessions(SessionQuery{}).Items), opened.Id)
	}, settleFor, 20*time.Millisecond, "the conversation was never written down")
	return opened.Id
}

func (s *GuestsSuite) claim(guest, account string) ClaimGuestResult {
	var claimed ClaimGuestResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/guests/claim", ClaimGuestRequest{GuestId: guest, UserId: account}, &claimed))
	return claimed
}
