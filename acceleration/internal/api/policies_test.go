//go:build integration

package api

import (
	"net/http"
	"testing"
)

type PoliciesSuite struct {
	RouterSuite
}

func TestPoliciesSuite(t *testing.T) {
	runSuite(t, new(PoliciesSuite))
}

// SetupTest gives every test an app of its own, because an app has one policy.
func (s *PoliciesSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *PoliciesSuite) TestAnAppPolicyIsStoredAndReadBackWithWhatItsBudgetSpent() {
	stored := s.put(map[string]any{
		"budget":           map[string]any{"limit_micros": 5_000_000, "interval": "daily"},
		"prompt_injection": true,
	})
	s.Require().NotNil(stored.Budget)
	s.Equal(int64(5_000_000), stored.Budget.LimitMicros)
	s.Equal(BudgetIntervalDaily, stored.Budget.Interval)

	var read Policy
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.Budget)
	s.Require().NotNil(read.Budget.SpentMicros)
	s.Zero(*read.Budget.SpentMicros)
	s.NotNil(read.Budget.ResetsAt)
	s.Require().NotNil(read.PromptInjection)
	s.True(*read.PromptInjection)
}

func (s *PoliciesSuite) TestABudgetOfNothingIsRefusedNamingTheFieldThatIsWrong() {
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"budget": map[string]any{"limit_micros": 0, "interval": "daily"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "limit_micros")
}

func (s *PoliciesSuite) TestOnlyTheAppsOwnBackendMaySetItsPolicy() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/policies/app",
			map[string]any{"budget": map[string]any{"limit_micros": 1_000, "interval": "daily"}}, nil)
	})
}

func (s *PoliciesSuite) TestAnAppsPolicyIsNotAnotherAppsPolicy() {
	s.put(map[string]any{"budget": map[string]any{"limit_micros": 7_000, "interval": "daily"}})

	stranger := s.data.backendOfAnotherApp()
	var read Policy
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Budget, "the other app has a policy of its own, and it is empty")
}

// put stores the policy the body describes.
func (s *PoliciesSuite) put(body map[string]any) Policy {
	var stored Policy
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPut, "/v1/policies/app", body, &stored))
	return stored
}
