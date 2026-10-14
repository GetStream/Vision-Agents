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

func (s *PoliciesSuite) TestAnAppCannotChangeTheDailyLimitStreamSetForIt() {
	// An app that could set its own users' limit could give them any number.
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": 20_000}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "quota is set by Stream")

	s.setQuota(map[string]any{"messages_per_user_per_day": 3})
	status, failure = s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": 20_000}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "quota is set by Stream")
	status, _ = s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": 3, "tokens_per_user_per_day": 0}})
	s.Equal(http.StatusBadRequest, status, "a number Stream left out is not the app's to set")

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(3)), read.Quota.MessagesPerUserPerDay)
}

func (s *PoliciesSuite) TestAnOrganizationHasNoDailyLimitToSet() {
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/organization",
		map[string]any{"quota": map[string]any{"messages_per_user_per_day": 5_000}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "quota is set by Stream")
}

func (s *PoliciesSuite) TestAnAppsPolicyReadAndWrittenBackKeepsItsDailyLimit() {
	s.setQuota(map[string]any{"messages_per_user_per_day": 3, "tokens_per_user_per_day": 0})
	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(0)), read.Quota.TokensPerUserPerDay, "zero is no cap, kept apart from left out")

	written := s.put(map[string]any{"quota": read.Quota, "prompt_injection": true})
	s.Equal(read.Quota, written.Quota, "sending back what was read is fine")

	written = s.put(map[string]any{"prompt_injection": false})
	s.Equal(read.Quota, written.Quota, "a policy that leaves the limit out keeps Stream's")
}

func (s *PoliciesSuite) TestOnlyStreamStaffMaySetAnAppsDailyLimit() {
	path := "/v1/ops/customers/" + s.customerID() + "/quota"
	body := map[string]any{"messages_per_user_per_day": 20_000}
	wrong := &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {"guess"}}, kind: noCredential}

	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodPut, path, body, nil), "the app's own backend")
	s.Equal(http.StatusUnauthorized, wrong.do(http.MethodPut, path, body, nil))
	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, path, nil, nil))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Quota)
}

func (s *PoliciesSuite) TestADailyLimitOutsideItsBoundsIsRefused() {
	path := "/v1/ops/customers/" + s.customerID() + "/quota"
	for field, over := range map[string]int64{
		"messages_per_user_per_day": 10_000_001,
		"tokens_per_user_per_day":   1_000_000_000_001,
	} {
		for _, sent := range []int64{-1, over} {
			status, body := s.staff().call(http.MethodPut, path, map[string]any{field: sent})

			s.Equal(http.StatusBadRequest, status, field, sent)
			s.Contains(string(body), `"code":"validation_failed"`, field, sent)
			s.Contains(string(body), field, sent)
		}
	}
	var quota Quota
	s.Require().Equal(http.StatusOK, s.staff().do(http.MethodGet, path, nil, &quota))
	s.Equal(Quota{}, quota)
}

func (s *PoliciesSuite) TestSettingAnAppsDailyLimitLeavesTheRestOfItsPolicy() {
	s.put(map[string]any{
		"budget":           map[string]any{"limit_micros": 5_000_000, "interval": "daily"},
		"prompt_injection": true,
		"tags":             map[string]string{"application": "support"},
	})

	set := s.setQuota(map[string]any{"messages_per_user_per_day": 3})
	s.Equal(pointerTo(int64(3)), set.MessagesPerUserPerDay)

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.Budget)
	s.Equal(int64(5_000_000), read.Budget.LimitMicros)
	s.Equal(pointerTo(true), read.PromptInjection)
	s.Equal(map[string]string{"application": "support"}, read.Tags)
	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(3)), read.Quota.MessagesPerUserPerDay)
	var staffRead Quota
	s.Require().Equal(http.StatusOK, s.staff().do(http.MethodGet,
		"/v1/ops/customers/"+s.customerID()+"/quota", nil, &staffRead))
	s.Equal(*read.Quota, staffRead)
}

func (s *PoliciesSuite) TestAQuotaSettingNeitherNumberIsStoredAsNoQuota() {
	s.setQuota(map[string]any{"messages_per_user_per_day": 3})
	s.setQuota(map[string]any{})

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Quota, "a quota with no number is no opinion")
}

// staff is Stream's own tool, holding the ops key and no app's credentials.
func (s *PoliciesSuite) staff() *testClient {
	return &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

// setQuota stores, as Stream staff, the app's daily limit per end user.
func (s *PoliciesSuite) setQuota(body map[string]any) Quota {
	var stored Quota
	s.Require().Equal(http.StatusOK, s.staff().do(http.MethodPut,
		"/v1/ops/customers/"+s.customerID()+"/quota", body, &stored))
	return stored
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
	status := s.serverClient.do(http.MethodPut, "/v1/policies/organization", read, nil)

	s.Require().Equal(http.StatusOK, status, "a backend writes back what it read with only its budget changed")
	// A replica's cached policy is dropped when Redis says so, a moment after the write.
	var written Policy
	s.Require().Eventually(func() bool {
		written = Policy{}
		return s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &written) == http.StatusOK &&
			written.Budget != nil && written.Budget.LimitMicros == 2_000
	}, settleFor, 10*time.Millisecond)
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
