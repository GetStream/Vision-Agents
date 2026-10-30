//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
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

func (s *GuestsSuite) TestAGuestOnlySeenInTheAppsOwnTokenIsNotMintedAgain() {
	// The app signs its guest tokens itself, for any id it likes, so a guest seen in one is
	// no sign the id is nobody else's.
	seen := s.guestSeenInAToken()

	status, failure := s.data.createAnonymous().failure(http.MethodPost, "/v1/agents/guests",
		GuestUserRequest{Id: pointerTo(seen)})

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

func (s *GuestsSuite) TestAClaimMovesTheGuestsScheduledTasks() {
	guest := s.recordGuest()
	ctx := context.Background()
	task := store.ScheduledTask{
		CustomerID: s.customerID(), UserID: guest, CallerKind: string(auth.KindServer), APIKey: s.app.key,
		Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: time.Now().UTC().Add(48 * time.Hour).Truncate(time.Second), Timezone: "UTC",
	}
	s.Require().NoError(s.store.CreateScheduledTask(ctx, &task, 0))
	s.T().Cleanup(func() { _ = s.store.DeleteScheduledTask(ctx, task.CustomerID, task.ID) })
	account := s.utils.uuid()

	s.claim(guest, account)

	moved, err := s.store.ScheduledTask(ctx, s.customerID(), task.ID)
	s.Require().NoError(err)
	s.Equal(account, moved.UserID, "the task follows up for the account the guest turned out to be")
}

func (s *GuestsSuite) TestAClaimedGuestsConversationIsTheAccountsByListAndById() {
	// A guest talks with a guest's token, so the row says a guest wrote it. Once claimed it
	// is the account's, reached the way the account reaches its own.
	guestID, account := s.recordGuest(), s.data.createUser()
	talked := s.talkedAs(s.asGuest(guestID))

	s.claim(guestID, account.userID)

	s.Contains(ids(account.querySessions(SessionQuery{}).Items), talked, "the account lists it")
	s.Equal(http.StatusOK, account.do(http.MethodGet, "/v1/agents/sessions/"+talked, nil, nil), "and opens it by id")
}

func (s *GuestsSuite) TestAClaimedGuestsScheduledTaskIsListedToTheAccount() {
	guestID, account := s.recordGuest(), s.data.createUser()
	task := s.scheduledFor(guestID, auth.KindGuest)

	s.claim(guestID, account.userID)

	var page ScheduledTaskPage
	s.Require().Equal(http.StatusOK, account.do(http.MethodGet, "/v1/agents/scheduled-tasks", nil, &page))
	s.Require().Len(page.Items, 1)
	s.Equal(task, page.Items[0].Id)
}

func (s *GuestsSuite) TestAClaimLeavesWhatAnAnonymousNamesakeSaidWhereItWas() {
	// A caller with no token picks its own name, so going by the guest's id or the account's
	// makes it neither: its conversations are not moved, and the account is not let into them.
	guestID, account := s.recordGuest(), s.data.createUser()
	genuine := s.talkedAs(s.asGuest(guestID))
	underTheGuestsName := s.talkedAs(s.data.claiming(guestID))
	underTheAccountsName := s.underTheName(account.userID)
	task := s.scheduledFor(guestID, auth.KindAnonymous)

	claimed := s.claim(guestID, account.userID)

	s.Equal(1, claimed.SessionsMoved, "only the guest's own conversation moved")
	s.Equal([]string{genuine}, ids(account.querySessions(SessionQuery{}).Items))
	s.Equal([]string{underTheGuestsName}, ids(s.data.claiming(guestID).querySessions(SessionQuery{}).Items),
		"the namesake keeps its conversation")
	s.Contains(s.chat.Members(s.transcriptOf(genuine)), account.userID, "the account reads what the guest said")
	s.NotContains(s.chat.Members(s.transcriptOf(underTheGuestsName)), account.userID)
	s.NotContains(s.chat.Members(s.transcriptOf(underTheAccountsName)), account.userID)
	left, err := s.store.ScheduledTask(context.Background(), s.customerID(), task)
	s.Require().NoError(err)
	s.Equal(guestID, left.UserID, "the namesake's task is not moved")
}

// recordGuest is a guest of the suite's app, written the way minting one writes it. Minting
// over HTTP would reach Stream for the token.
func (s *GuestsSuite) recordGuest() string {
	id := guestPrefix + s.utils.uuid()
	s.Require().NoError(s.store.RecordGuest(context.Background(), &store.User{
		ID: id, CustomerID: s.customerID(), Name: "Guest",
	}))
	return id
}

// guestSeenInAToken is an id the suite's app called as once, with a guest token it signed
// itself: recorded as its guest, though the router never minted it.
func (s *RouterSuite) guestSeenInAToken() string {
	id := "visitor-" + s.utils.uuid()
	s.signedIn(jwt.MapClaims{"user_id": id, "role": "guest"}, auth.AuthTypeJWT, guest, id).
		do(http.MethodGet, "/v1/agents/calls", nil, nil)
	seen, err := s.store.Guest(context.Background(), s.customerID(), id)
	s.Require().NoError(err, "the call is what records the guest")
	s.Require().False(seen.Minted)
	return id
}

// talked is a conversation the guest had and has finished, which is the history a claim
// moves. The row is written off the request path, so it is waited for: a claim that
// arrives before it moves nothing.
func (s *GuestsSuite) talked(guest string) string {
	return s.talkedAs(s.data.signedInAs(guest))
}

// talkedAs is a finished conversation the caller had, waited for the same way as talked.
func (s *GuestsSuite) talkedAs(talking *testClient) string {
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

// asGuest is a client holding a guest's token for the id, as a minted guest does.
func (s *GuestsSuite) asGuest(id string) *testClient {
	return s.signedIn(jwt.MapClaims{"user_id": id, "role": "guest"}, auth.AuthTypeJWT, guest, id)
}

// scheduledFor is a task for the user, written as a caller of that kind would have.
func (s *GuestsSuite) scheduledFor(userID string, kind auth.Kind) string {
	ctx := context.Background()
	task := store.ScheduledTask{
		CustomerID: s.customerID(), UserID: userID, CallerKind: string(kind), APIKey: s.app.key,
		Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: time.Now().UTC().Add(48 * time.Hour).Truncate(time.Second), Timezone: "UTC",
	}
	s.Require().NoError(s.store.CreateScheduledTask(ctx, &task, 0))
	s.T().Cleanup(func() { _ = s.store.DeleteScheduledTask(ctx, task.CustomerID, task.ID) })
	return task.ID
}

// transcriptOf is the Stream Chat channel the session's words were written to.
func (s *GuestsSuite) transcriptOf(sessionID string) string {
	var row store.AgentSession
	s.Require().NoError(s.store.DB().NewSelect().Model(&row).Where("id = ?", sessionID).Scan(context.Background()))
	channel := transcriptChannel(row)
	_, kept := s.chat.Channel(channel)
	s.Require().True(kept, "the session wrote a transcript")
	return channel
}

// underTheName is an ended conversation an anonymous caller had that the row files under the
// name, in a channel the name's owner was never put in. An anonymous caller opening one under
// a name would put the name in its channel, so the row is moved under it afterwards.
func (s *GuestsSuite) underTheName(name string) string {
	talked := s.talkedAs(s.data.createAnonymous())
	_, err := s.store.DB().NewUpdate().Model((*store.AgentSession)(nil)).
		Set("user_id = ?", name).Where("id = ?", talked).Exec(context.Background())
	s.Require().NoError(err)
	s.Require().NotContains(s.chat.Members(s.transcriptOf(talked)), name)
	return talked
}
