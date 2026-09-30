//go:build integration

package api

import (
	"net/http"
	"testing"
	"time"
)

// QuotaSuite runs against a router that allows one message a day, so what a caller who has
// spent their day may still do is a question the HTTP surface answers.
type QuotaSuite struct {
	RouterSuite

	// backend is the app's own server, naming the suite's user as who it acts for. Asking
	// for a reply is server-side, and the reply is charged to the user it was for.
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

// spendTheDay has the app's backend ask for a reply on the client's behalf, which is what
// the allowance counts, and waits until the client is refused a new session. It returns
// the session it opened, which is still the client's to close.
func (s *QuotaSuite) spendTheDay() string {
	opened := s.backend.createSession(textSession(nil))

	command := s.utils.uuid()
	status, failure := s.backend.failure(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "how much does a call cost", CommandId: &command})
	s.Require().Equal(http.StatusOK, status, failure)

	s.Require().Eventually(func() bool {
		status, _ := s.client.call(http.MethodPost, "/v1/agents/sessions", textSession(nil))
		return status == http.StatusTooManyRequests
	}, settleFor, 20*time.Millisecond, "the reply was never counted against the day")
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
