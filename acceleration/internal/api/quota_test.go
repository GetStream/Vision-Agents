//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// QuotaSuite runs against a router that allows one message a day by default, so what a
// caller who has spent their day may still do is a question the HTTP surface answers.
type QuotaSuite struct {
	RouterSuite

	// backend is the app's own server, naming the suite's user as who it acts for. What it
	// asks for is server-side, and is charged to nobody's day.
	backend *testClient
}

func TestQuotaSuite(t *testing.T) {
	runSuite(t, new(QuotaSuite))
}

func (s *QuotaSuite) SetupSuite() {
	s.messagesPerDay = 1
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test a user of its own and an unspent day. A day is spent by
// address as well as by user, and every test here calls from the same address.
func (s *QuotaSuite) SetupTest() {
	s.useFixture("standard")
	s.client = s.data.createUser()
	s.backend = s.serverClient.actingFor(s.client)
	s.clearAllowances()
}

func (s *QuotaSuite) TestAUserWhoHasSpentTheirDayIsRefusedAnotherSession() {
	s.spendTheDay()

	status, failure := s.client.failure(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Equal(http.StatusTooManyRequests, status)
	s.NotEmpty(failure, "a caller at the cap is told what happened")
}

func (s *QuotaSuite) TestAUserWhoHasSpentTheirDayMayStillStopTheSessionTheyHave() {
	// Stopping one cannot create work, and a session nobody can stop is a session that
	// stays open until it times out.
	opened := s.spendTheDay()

	s.Equal(http.StatusNoContent,
		s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened+"/stop", nil, nil))
}

func (s *QuotaSuite) TestAUserWhoHasSpentTheirDayMayStillDeleteASession() {
	opened := s.spendTheDay()

	s.Equal(http.StatusNoContent,
		s.client.do(http.MethodDelete, "/v1/agents/sessions/"+opened, nil, nil))
}

func (s *QuotaSuite) TestASessionSomebodyElseHoldsIsStillNotTheirsToStop() {
	// The allowance is not a way round the ownership check: stopping is exempt from the
	// cap, not from the answer about whose session it is.
	opened := s.spendTheDay()

	s.Equal(http.StatusNotFound,
		s.data.createUser().do(http.MethodPost, "/v1/agents/sessions/"+opened+"/stop", nil, nil))
}

func (s *QuotaSuite) TestABackendIsNotCountedAgainstADaysAllowance() {
	// The cap is what one person may ask for. A customer's own backend is billed rather
	// than limited, so a busy one is not cut off after a message.
	s.spendTheDay()

	for range 3 {
		s.backend.createSession(textSession(nil))
	}
}

func (s *QuotaSuite) TestABackendsRepliesForAUserAreNotChargedToTheirDay() {
	// The backend asks for a reply on the user's behalf, which it is trusted with. The
	// counted model reports what each reply cost, so the reply's row says it was answered.
	// An app of its own keeps other suites' answers from the same address out of its day.
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.backend = s.serverClient.actingFor(s.client)
	request := textSession(nil)
	request.Llm = pointerTo("counted/counted-model")
	opened := s.backend.createSession(request)
	command := s.utils.uuid()
	status, failure := s.backend.failure(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "how much does a call cost", RequestId: &command})
	s.Require().Equal(http.StatusOK, status, failure)
	// The day is debited just before the reply's request row is recorded, so a row means
	// the debit, if there was going to be one, has landed.
	s.Require().Eventually(func() bool {
		var spent CallTokens
		return s.backend.do(http.MethodGet, "/v1/agents/calls/"+opened.Id+"/tokens", nil, &spent) == http.StatusOK &&
			spent.InputTokens+spent.OutputTokens > 0
	}, settleFor, 20*time.Millisecond, "the reply was never recorded")

	s.NotEqual(http.StatusTooManyRequests, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil))
}

func (s *QuotaSuite) TestAUsersOwnTurnsInASessionTheirBackendOpenedAreCountedOnTheirDay() {
	// The backend opens the conversation in the user's name and hands it over; what the user
	// then sends with their own token is theirs, and the router's default is one a day.
	opened := s.backendOpensACountedSession()

	status, failure := s.client.failure(http.MethodPost, "/v1/agents/sessions/"+opened+"/responses",
		CreateResponseRequest{Text: "how much does a call cost", RequestId: pointerTo(s.utils.uuid())})
	s.Require().Equal(http.StatusAccepted, status, failure)
	s.awaitTheDaySpent()

	status, _ = s.client.failure(http.MethodPost, "/v1/agents/sessions/"+opened+"/responses",
		CreateResponseRequest{Text: "and a text", RequestId: pointerTo(s.utils.uuid())})
	s.Equal(http.StatusTooManyRequests, status, "the user's next turn is refused")
	status, failure = s.backend.failure(http.MethodPost, "/v1/agents/sessions/"+opened+"/respond",
		RespondRequest{Text: "and a text", RequestId: pointerTo(s.utils.uuid())})
	s.Equal(http.StatusOK, status, "the backend's own turn in the same session is not counted: %s", failure)
}

func (s *QuotaSuite) TestAUsersTurnsOverTheSocketOfASessionTheirBackendOpenedAreCountedOnTheirDay() {
	opened := s.backendOpensACountedSession()
	connection := s.client.opens("/v1/agents/sessions/" + opened + "/events")

	s.Require().NoError(connection.WriteJSON(frame{"type": "respond", "request_id": s.utils.uuid(),
		"text": "how much does a call cost"}))

	s.awaitTheDaySpent()
}

func (s *QuotaSuite) TestAUsersTurnsOverTheSocketOfAnUnrecordedSessionTheirBackendOpenedAreCountedOnTheirDay() {
	// Off the record, a turn has no request id to be taken once by, and is answered as sent.
	request := textSession(nil)
	request.Llm = pointerTo("counted/counted-model")
	request.Incognito = pointerTo(true)
	opened := s.backend.createSession(request).Id
	connection := s.client.opens("/v1/agents/sessions/" + opened + "/events")

	s.Require().NoError(connection.WriteJSON(frame{"type": "respond", "text": "how much does a call cost"}))

	s.awaitTheDaySpent()
}

func (s *QuotaSuite) TestABackendsOwnTurnsInASessionItOpenedForAUserAreNeverCounted() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.backend = s.serverClient.actingFor(s.client)
	s.putQuota(2)
	opened := s.backendOpensACountedSession()

	for turn := range 3 {
		status, failure := s.backend.failure(http.MethodPost, "/v1/agents/sessions/"+opened+"/respond",
			RespondRequest{Text: "how much does a call cost", RequestId: pointerTo(s.utils.uuid())})
		s.Require().Equal(http.StatusOK, status, failure)
		s.awaitReplies(opened, turn+1)
	}

	response := s.client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))
	s.Require().Equal(http.StatusCreated, response.StatusCode)
	s.Equal("2", response.Header.Get("X-Quota-Remaining"), "none of the backend's turns was the user's")
}

func (s *QuotaSuite) TestAUsersOwnSessionIsCountedOnTheirDayWhoeverSendsItsTurns() {
	// Unchanged by who sends a turn: a backend answering for the user in the user's own
	// session spends the user's day, as it always has.
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.backend = s.serverClient.actingFor(s.client)
	request := textSession(nil)
	request.Llm = pointerTo("counted/counted-model")
	opened := s.client.createSession(request)

	status, failure := s.backend.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "how much does a call cost", RequestId: pointerTo(s.utils.uuid())})
	s.Require().Equal(http.StatusOK, status, failure)

	s.awaitTheDaySpent()
}

func (s *QuotaSuite) TestAnAppWithNoPolicyIsHeldToTheDeploymentsLimit() {
	// An app that set nothing is refused at the deployment's default.
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()

	s.Equal(1, s.messagesBeforeRefusal())
}

func (s *QuotaSuite) TestAnAppsOwnLimitIsWhatItsUsersAreRefusedAt() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.putQuota(3)

	s.Equal(3, s.messagesBeforeRefusal())

	response := s.client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	retryAfter, err := strconv.Atoi(response.Header.Get("Retry-After"))
	s.Require().NoError(err)
	reset, err := strconv.ParseInt(response.Header.Get("X-Quota-Reset"), 10, 64)
	s.Require().NoError(err)
	// Both headers name the same midnight: the wait is what is left of it, to the second.
	s.InDelta(time.Until(time.Unix(reset, 0)).Seconds(), float64(retryAfter), 2)
	s.Equal("3", response.Header.Get("X-Quota-Limit"))
	s.Equal("0", response.Header.Get("X-Quota-Remaining"))
	var refused ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&refused))
	s.Equal("rate_limited", refused.Error.Code)
}

func (s *QuotaSuite) TestAChangedLimitAppliesAtOnceOnTheNodeThatStoredIt() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.backend = s.serverClient.actingFor(s.client)
	s.spendTheDay()
	s.Equal(http.StatusTooManyRequests, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil),
		"the router's default of one message is spent, and the decision is now cached")

	s.putQuota(5)

	s.Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil))
}

func (s *QuotaSuite) TestAnAppsLimitOfZeroRefusesNobody() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.putQuota(0)

	for range 3 {
		s.spend(s.client)
	}

	response := s.client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))
	s.Equal(http.StatusCreated, response.StatusCode)
	s.Empty(response.Header.Get("X-Quota-Limit"), "a caller with no cap is told nothing")
}

func (s *QuotaSuite) TestACountedAnswerSaysWhatIsLeftOfTheDay() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.putQuota(3)
	s.spend(s.client)

	response := s.client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Require().Equal(http.StatusCreated, response.StatusCode)
	s.Equal("3", response.Header.Get("X-Quota-Limit"))
	s.Equal("2", response.Header.Get("X-Quota-Remaining"))
	now := time.Now().UTC()
	midnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	s.Equal(strconv.FormatInt(midnight.Unix(), 10), response.Header.Get("X-Quota-Reset"))
}

func (s *QuotaSuite) TestABackendsAnswerCarriesNoRateLimit() {
	response := s.backend.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Require().Equal(http.StatusCreated, response.StatusCode)
	s.Empty(response.Header.Get("X-Quota-Limit"))
}

// putQuota sets, through the app's own policy, the messages one of its users may ask for in
// a day.
func (s *QuotaSuite) putQuota(messages int64) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/app",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": messages}}, nil))
}

// messagesBeforeRefusal spends the client's day one message at a time until a new session
// is refused, and says how many it took, up to ten.
func (s *QuotaSuite) messagesBeforeRefusal() int {
	spent := 0
	for ; spent < 10; spent++ {
		if s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil) == http.StatusTooManyRequests {
			return spent
		}
		s.spend(s.client)
	}
	return spent
}

// spend charges one message to a user's day, the way one of their own replies is.
func (s *QuotaSuite) spend(user *testClient) {
	s.limiter.Debit(s.T().Context(), s.customerID(), routing.Caller{UserID: user.userID}, 1)
}

// spendTheDay opens a session the app's backend holds for the client, and spends the
// client's day. It returns the session, which is still the client's to close.
func (s *QuotaSuite) spendTheDay() string {
	opened := s.backend.createSession(textSession(nil))
	s.spend(s.client)
	s.Require().Equal(http.StatusTooManyRequests,
		s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil))
	return opened.Id
}

// backendOpensACountedSession opens, as the app's backend acting for the client, a session
// whose model reports what each reply cost, and returns its id.
func (s *QuotaSuite) backendOpensACountedSession() string {
	request := textSession(nil)
	request.Llm = pointerTo("counted/counted-model")
	return s.backend.createSession(request).Id
}

// awaitTheDaySpent waits until the client is refused a new session, which a reply debited
// to their day after it settled is what does.
func (s *QuotaSuite) awaitTheDaySpent() {
	s.Require().Eventually(func() bool {
		return s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil) == http.StatusTooManyRequests
	}, settleFor, 50*time.Millisecond, "the client's day was never spent")
}

// awaitReplies waits until a session's replies have all been recorded. The day is debited
// just before a reply's request row is, so a row means its debit has landed.
func (s *QuotaSuite) awaitReplies(sessionID string, replies int) {
	s.Require().Eventually(func() bool {
		var rows []store.Request
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).Scan(s.T().Context())
		return err == nil && len(rows) >= replies
	}, settleFor, 50*time.Millisecond, "the replies of %s were never recorded", sessionID)
}

// clearAllowances forgets what has been spent today, so that one test's message is not
// still on the books for the next.
func (s *QuotaSuite) clearAllowances() {
	redis := s.live.Redis()
	spent, err := redis.Do(s.T().Context(), redis.B().Keys().Pattern("quota:*").Build()).AsStrSlice()
	s.Require().NoError(err)
	if len(spent) == 0 {
		return
	}
	s.Require().NoError(redis.Do(s.T().Context(), redis.B().Del().Key(spent...).Build()).Error())
}
