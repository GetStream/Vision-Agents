//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
	reset, err := strconv.ParseInt(response.Header.Get("X-RateLimit-Reset"), 10, 64)
	s.Require().NoError(err)
	// Both headers name the same midnight: the wait is what is left of it, to the second.
	s.InDelta(time.Until(time.Unix(reset, 0)).Seconds(), float64(retryAfter), 2)
	s.Equal("3", response.Header.Get("X-RateLimit-Limit"))
	s.Equal("0", response.Header.Get("X-RateLimit-Remaining"))
	var refused ErrorResponse
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&refused))
	s.Equal("rate_limited", refused.Error.Code)
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
	s.Empty(response.Header.Get("X-RateLimit-Limit"), "a caller with no cap is told nothing")
}

func (s *QuotaSuite) TestACountedAnswerSaysWhatIsLeftOfTheDay() {
	s.useApp(s.data.createApp())
	s.client = s.data.createUser()
	s.putQuota(3)
	s.spend(s.client)

	response := s.client.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Require().Equal(http.StatusCreated, response.StatusCode)
	s.Equal("3", response.Header.Get("X-RateLimit-Limit"))
	s.Equal("2", response.Header.Get("X-RateLimit-Remaining"))
	now := time.Now().UTC()
	midnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	s.Equal(strconv.FormatInt(midnight.Unix(), 10), response.Header.Get("X-RateLimit-Reset"))
}

func (s *QuotaSuite) TestABackendsAnswerCarriesNoRateLimit() {
	response := s.backend.raw(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Require().Equal(http.StatusCreated, response.StatusCode)
	s.Empty(response.Header.Get("X-RateLimit-Limit"))
}

// putQuota sets, as Stream staff, the messages one user of the app may ask for in a day.
func (s *QuotaSuite) putQuota(messages int64) {
	staff := &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
	s.Require().Equal(http.StatusOK, staff.do(http.MethodPut, "/v1/ops/customers/"+s.customerID()+"/quota",
		map[string]any{"messages_per_user_per_day": messages}, nil))
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
