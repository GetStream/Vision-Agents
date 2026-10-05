//go:build integration

package api

import (
	"context"
	"maps"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

func (s *PoliciesSuite) TestAllowedModelsAndTagsAreStoredAndReadBack() {
	s.put(map[string]any{
		"allowed_models": []string{"vision/vision-model"},
		"tags":           map[string]string{"application": "athena"},
	})

	var read Policy
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.AllowedModels)
	s.Equal([]string{"vision/vision-model"}, *read.AllowedModels)
	s.Equal(map[string]string{"application": "athena"}, read.Tags)
}

func (s *PoliciesSuite) TestAnEmptyAllowlistReadsBackAsAllowingNothing() {
	stored := s.put(map[string]any{"allowed_models": []string{}})

	s.Require().NotNil(stored.AllowedModels, "an empty list is not the same as no list")
	s.Empty(*stored.AllowedModels)
}

func (s *PoliciesSuite) TestAnAllowlistNamingAModelNothingRoutesIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"allowed_models": []string{"nobody/no-model"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nobody/no-model")
}

func (s *PoliciesSuite) TestTagsTheRollupsCannotCarryAreRefused() {
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"tags": map[string]string{"not a key": "value"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not a key")
}

func (s *PoliciesSuite) TestARequestForAModelThePolicyDoesNotAllowIsRefused() {
	s.put(map[string]any{"allowed_models": []string{"vision/vision-model"}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/search", SearchRequest{Query: "what is the time"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "does not allow")
}

func (s *PoliciesSuite) TestUsageIsRecordedUnderThePolicysTagsOverTheRequests() {
	s.put(map[string]any{"tags": map[string]string{"application": "athena"}})
	sent := map[string]string{"application": "spoofed", "feature": s.utils.uuid()}

	status, payload := s.serverClient.call(http.MethodPost, "/v1/search",
		SearchRequest{Query: "what is the time", Tags: &sent})
	s.Require().Equal(http.StatusOK, status, string(payload))

	want := map[string]string{"application": "athena", "feature": sent["feature"]}
	s.Require().Eventually(func() bool {
		var rows []store.Request
		err := s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).Scan(context.Background())
		return err == nil && len(rows) == 1 && maps.Equal(rows[0].Tags, want)
	}, 5*time.Second, 50*time.Millisecond)
}

// put stores the policy the body describes.
func (s *PoliciesSuite) put(body map[string]any) Policy {
	var stored Policy
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPut, "/v1/policies/app", body, &stored))
	return stored
}

func (s *PoliciesSuite) TestTheOrganizationRequirementIsNotWritableOverHTTP() {
	// Any app's backend can write its organization's policy, and this setting would let one
	// app keep all its siblings out of the shared app. It is the operator's.
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/organization",
		map[string]any{"require_own_stream_app": true})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "operator")

	required := true
	s.Require().NoError(s.store.SavePolicy(context.Background(), store.ScopeOrganization,
		s.app.organization.ID, store.PolicyDocument{RequireOwnStreamApp: &required}))
	var written Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization",
		map[string]any{"prompt_injection": true}, &written))
	s.Require().NotNil(written.RequireOwnStreamApp, "a write over HTTP keeps what the operator set")
	s.True(*written.RequireOwnStreamApp)
}

func (s *PoliciesSuite) TestAnOrganizationPolicyReadAndWrittenBackKeepsTheOperatorsRequirement() {
	required := true
	s.Require().NoError(s.store.SavePolicy(context.Background(), store.ScopeOrganization,
		s.app.organization.ID, store.PolicyDocument{RequireOwnStreamApp: &required}))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/organization",
		map[string]any{"budget": map[string]any{"limit_micros": 1_000, "interval": "daily"}}, nil))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &read))
	read.Budget.LimitMicros = 2_000
	var written Policy
	status := s.serverClient.do(http.MethodPut, "/v1/policies/organization", read, &written)

	s.Require().Equal(http.StatusOK, status, "a backend writes back what it read with only its budget changed")
	s.Equal(int64(2_000), written.Budget.LimitMicros)
	s.Require().NotNil(written.RequireOwnStreamApp)
	s.True(*written.RequireOwnStreamApp)
}

func (s *PoliciesSuite) TestChangingTheOrganizationRequirementOverHTTPIsRefused() {
	required := true
	s.Require().NoError(s.store.SavePolicy(context.Background(), store.ScopeOrganization,
		s.app.organization.ID, store.PolicyDocument{RequireOwnStreamApp: &required}))

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/organization",
		map[string]any{"require_own_stream_app": false})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "operator")
	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &read))
	s.Require().NotNil(read.RequireOwnStreamApp)
	s.True(*read.RequireOwnStreamApp)
}

func (s *PoliciesSuite) TestAnAppMayRequireItsOwnStreamApp() {
	stored := s.put(map[string]any{"require_own_stream_app": true})

	s.Require().NotNil(stored.RequireOwnStreamApp)
	s.True(*stored.RequireOwnStreamApp)
}
