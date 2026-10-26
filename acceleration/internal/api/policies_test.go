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

func (s *PoliciesSuite) TestAnAppPolicyIsStoredAndReadBack() {
	stored := s.put(map[string]any{"prompt_injection": true})
	s.Require().NotNil(stored.PromptInjection)

	var read Policy
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.PromptInjection)
	s.True(*read.PromptInjection)
	s.Nil(read.Budget, "a policy reads back no budget")
}

func (s *PoliciesSuite) TestAPolicyBudgetIsRefused() {
	for _, path := range []string{"/v1/policies/app", "/v1/policies/organization"} {
		status, body := s.serverClient.call(http.MethodPut, path,
			map[string]any{"budget": map[string]any{"limit_micros": 1_000, "interval": "daily"}, "prompt_injection": true})

		s.Equal(http.StatusBadRequest, status, path)
		s.Contains(string(body), `"code":"`+codeBudgetMoved+`"`, path)
		s.Contains(string(body), "/v1/spend/limits", path)
	}
	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.PromptInjection, "nothing of a refused policy is written")
}

func (s *PoliciesSuite) TestOnlyTheAppsOwnBackendMaySetItsPolicy() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/policies/app", map[string]any{"prompt_injection": true}, nil)
	})
}

func (s *PoliciesSuite) TestAnAppsPolicyIsNotAnotherAppsPolicy() {
	s.put(map[string]any{"prompt_injection": true})

	stranger := s.data.backendOfAnotherApp()
	var read Policy
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.PromptInjection, "the other app has a policy of its own, and it is empty")
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
		err := s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).Scan(context.Background())
		return err == nil && len(rows) == 1 && maps.Equal(rows[0].Tags, want)
	}, 5*time.Second, 50*time.Millisecond)
}

func (s *PoliciesSuite) TestAnAppsDailyLimitIsStoredAndReadBack() {
	stored := s.put(map[string]any{"quota": map[string]any{
		"messages_per_user_per_day": 20_000, "tokens_per_user_per_day": 0,
	}})
	s.Require().NotNil(stored.Quota)

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Require().NotNil(read.Quota)
	s.Equal(pointerTo(int64(20_000)), read.Quota.MessagesPerUserPerDay)
	s.Equal(pointerTo(int64(0)), read.Quota.TokensPerUserPerDay, "zero is no cap, kept apart from left out")
}

func (s *PoliciesSuite) TestAPolicyThatLeavesTheDailyLimitOutClearsIt() {
	// A PUT replaces the whole policy, the daily limit as much as the rest.
	s.put(map[string]any{"quota": map[string]any{"messages_per_user_per_day": 3}})

	written := s.put(map[string]any{"prompt_injection": true})
	s.Nil(written.Quota)

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Quota)
}

func (s *PoliciesSuite) TestADailyLimitOutsideItsBoundsIsRefused() {
	for field, over := range map[string]int64{
		"messages_per_user_per_day": 10_000_001,
		"tokens_per_user_per_day":   1_000_000_000_001,
	} {
		for _, sent := range []int64{-1, over} {
			status, body := s.serverClient.call(http.MethodPut, "/v1/policies/app",
				map[string]any{"quota": map[string]any{field: sent}})

			s.Equal(http.StatusBadRequest, status, field, sent)
			s.Contains(string(body), `"code":"validation_failed"`, field, sent)
			s.Contains(string(body), field, sent)
		}
	}
	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Quota)
}

func (s *PoliciesSuite) TestAQuotaSettingNeitherNumberIsStoredAsNoQuota() {
	stored := s.put(map[string]any{"quota": map[string]any{}})
	s.Nil(stored.Quota)

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.Quota, "a quota with no number is no opinion")
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
		map[string]any{"tags": map[string]string{"team": "a"}}, nil))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &read))
	read.Tags["team"] = "b"
	status := s.serverClient.do(http.MethodPut, "/v1/policies/organization", read, nil)

	s.Require().Equal(http.StatusOK, status, "a backend writes back what it read with only its tags changed")
	// A replica's cached policy is dropped when Redis says so, a moment after the write.
	var written Policy
	s.Require().Eventually(func() bool {
		written = Policy{}
		return s.serverClient.do(http.MethodGet, "/v1/policies/organization", nil, &written) == http.StatusOK &&
			written.Tags["team"] == "b"
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

func (s *PoliciesSuite) TestAToolOutputRetentionIsStoredAndReadBack() {
	stored := s.put(map[string]any{"tool_output_retention": "30d"})
	s.Equal("30d", value(stored.ToolOutputRetention))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Equal("30d", value(read.ToolOutputRetention))

	cleared := s.put(map[string]any{})
	s.Nil(cleared.ToolOutputRetention, "a policy left out keeps tool output as long as the session again")
}

func (s *PoliciesSuite) TestAToolOutputRetentionOutsideItsBoundsIsRefused() {
	for _, retention := range []string{"3651d", "87601h"} {
		status, failure := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
			map[string]any{"tool_output_retention": retention})
		s.Equal(http.StatusBadRequest, status, retention)
		s.Contains(failure, "tool_output_retention", retention)
	}
	for _, retention := range []string{"0d", "30m", "forever", "1w"} {
		status, body := s.serverClient.call(http.MethodPut, "/v1/policies/app",
			map[string]any{"tool_output_retention": retention})
		s.Equal(http.StatusBadRequest, status, retention)
		s.Contains(string(body), `"code":"validation_failed"`, retention)
	}
}
