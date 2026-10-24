//go:build integration

package api

import (
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// BalanceSuite reads what every answer to a backend says of its organization's credit, against
// what the real usage service holds.
type BalanceSuite struct {
	RouterSuite
}

func TestBalanceSuite(t *testing.T) {
	runSuite(t, new(BalanceSuite))
}

func (s *BalanceSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an organization of its own.
func (s *BalanceSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *BalanceSuite) organization() string { return s.app.organization.ID }

// balanceSaid sends a request that records nothing until its answer says the balance is
// micros, and returns that answer.
func (s *BalanceSuite) balanceSaid(micros int64) *http.Response {
	var response *http.Response
	s.Require().Eventually(func() bool {
		response = s.serverClient.raw(http.MethodGet, "/v1/policies/app", nil)
		_ = response.Body.Close()
		return response.Header.Get(headerBalanceMicros) == strconv.FormatInt(micros, 10)
	}, 10*time.Second, 50*time.Millisecond, "the answer says the balance is %d", micros)
	return response
}

func (s *BalanceSuite) TestTheBalanceIsOnEveryBackendAnswer() {
	s.usage.Credit(s.T(), s.organization(), 100_000_000)
	s.balanceSaid(100_000_000)

	// A search holds what it cost from the moment it is recorded until the usage service has its
	// row, which this suite never ships: the answer is the standing less that hold.
	searched := s.answered(s.serverClient, http.MethodPost, "/v1/search", SearchRequest{Query: "anything"}, http.StatusOK)
	cost := s.costOf(searched)
	standing, ok, err := s.enforcer.Standing(s.T().Context(), s.organization())
	s.Require().NoError(err)
	s.Require().True(ok)
	s.EqualValues(100_000_000, standing.BalanceMicros, "the usage service has not had the row")
	s.Equal(strconv.FormatInt(standing.BalanceMicros-cost, 10), searched.Header.Get(headerBalanceMicros),
		"the search's own cost is held against it at once")

	answer := s.balanceSaid(standing.BalanceMicros - cost)
	s.Empty(answer.Header.Get(headerBalanceWarning), "the default tier sets no threshold here")
}

func (s *BalanceSuite) TestBelowTheThresholdTheWarningIsLow() {
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{LowBalanceMicros: pointerTo(int64(5_000_000))})
	s.usage.Credit(s.T(), s.organization(), 4_000_000)

	answer := s.balanceSaid(4_000_000)

	s.Equal(balanceLow, answer.Header.Get(headerBalanceWarning))
}

func (s *BalanceSuite) TestAtTheThresholdTheWarningIsLowAndAboveItThereIsNone() {
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{LowBalanceMicros: pointerTo(int64(5_000_000))})
	s.usage.Credit(s.T(), s.organization(), 5_000_000)
	s.Equal(balanceLow, s.balanceSaid(5_000_000).Header.Get(headerBalanceWarning), "at it is low")

	s.usage.Credit(s.T(), s.organization(), 1)

	s.Empty(s.balanceSaid(5_000_001).Header.Get(headerBalanceWarning), "a top-up lifts it")
}

func (s *BalanceSuite) TestA402SaysExhausted() {
	var response *http.Response
	s.Require().Eventually(func() bool {
		response = s.serverClient.raw(http.MethodPost, "/v1/search", SearchRequest{Query: "anything"})
		_ = response.Body.Close()
		return response.StatusCode == http.StatusPaymentRequired
	}, 10*time.Second, 50*time.Millisecond, "a search with no credit is refused")

	s.Equal("0", response.Header.Get(headerBalanceMicros))
	s.Equal(balanceExhausted, response.Header.Get(headerBalanceWarning))
	s.Empty(response.Header.Get(headerCostMicros), "refused, it recorded nothing")
}

func (s *BalanceSuite) TestTheCustomersThresholdWins() {
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{LowBalanceMicros: pointerTo(int64(1_000_000))})
	s.usage.Credit(s.T(), s.organization(), 10_000_000)
	s.Empty(s.balanceSaid(10_000_000).Header.Get(headerBalanceWarning), "above the tier's threshold")

	var stored Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization",
		Policy{BalanceWarningMicros: pointerTo(int64(20_000_000))}, &stored))
	s.EqualValues(20_000_000, *stored.BalanceWarningMicros)

	s.Require().Eventually(func() bool {
		response := s.serverClient.raw(http.MethodGet, "/v1/policies/app", nil)
		_ = response.Body.Close()
		return response.Header.Get(headerBalanceWarning) == balanceLow
	}, 10*time.Second, 50*time.Millisecond, "under the customer's own threshold, once the policy is read again")
}

func (s *BalanceSuite) TestAnAppHasNoThresholdOfItsOwn() {
	status, message := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		Policy{BalanceWarningMicros: pointerTo(int64(20_000_000))})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, "organization's policy")
}
