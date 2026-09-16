package routing

import (
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func (s *RoutingSuite) policyRouter(allowed []string, failing bool) *Router[*stubProvider] {
	config := s.config()
	config.CustomerPolicies = map[string]CustomerPolicy{
		"athena-app": {AllowedModels: allowed, Tags: Tags{"application": "athena", "environment": "test"}},
	}
	registry := NewRegistry[*stubProvider]()
	for _, name := range []string{"quick", "lush", "batchy"} {
		registry.Register(name, func(spec Spec) (*stubProvider, error) {
			provider := &stubProvider{model: spec.Model}
			if failing && name == "quick" {
				provider.startErr = errors.New("approved model unavailable")
			}
			return provider, nil
		})
	}
	router, err := New(Options[*stubProvider]{Modality: LLM, Config: config, Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *RoutingSuite) TestCustomerPolicyFiltersAliasesAndProviderPriorities() {
	router := s.policyRouter([]string{"quick/multi"}, false)
	for _, request := range []Request{
		{Target: "en-low-latency"},
		{Providers: []string{"lush/multi", "quick"}},
		{Target: "quick/multi"},
	} {
		request.CustomerID = "athena-app"
		provider, selected, err := router.Select(s.ctx, request)
		s.Require().NoError(err)
		s.Equal("quick/multi", selected.Name())
		s.NoError(provider.Close())
	}
	for _, request := range []Request{
		{Target: "quick/en"},
		{Providers: []string{"lush", "quick/en"}},
	} {
		request.CustomerID = "athena-app"
		_, _, err := router.Select(s.ctx, request)
		s.ErrorContains(err, "no requested model is permitted")
	}
	provider, selected, err := router.Select(s.ctx, Request{CustomerID: "another-app", Target: "quick/en"})
	s.Require().NoError(err)
	s.Equal("quick/en", selected.Name())
	s.NoError(provider.Close())
}

func (s *RoutingSuite) TestEmptyCustomerAllowlistDisablesAllModels() {
	router := s.policyRouter(nil, false)
	_, _, err := router.Select(s.ctx, Request{CustomerID: "athena-app", Target: "en-low-latency"})
	s.ErrorContains(err, "no requested model is permitted")
}

func (s *RoutingSuite) TestCustomerPolicyDoesNotFailOverToAnUnapprovedModel() {
	router := s.policyRouter([]string{"quick/multi"}, true)
	_, _, err := router.Select(s.ctx, Request{CustomerID: "athena-app", Providers: []string{"quick/multi", "lush/multi"}})
	s.ErrorContains(err, "approved model unavailable")
}

func (s *RoutingSuite) TestCustomerPoliciesRejectUndeclaredModelsAndInvalidTags() {
	config := s.config()
	for _, policy := range []CustomerPolicy{
		{AllowedModels: []string{"en-low-latency"}},
		{AllowedModels: []string{"unknown/model"}},
		{AllowedModels: []string{"quick/en"}, Tags: Tags{"invalid key": "value"}},
	} {
		config.CustomerPolicies = map[string]CustomerPolicy{"athena-app": policy}
		s.Error(config.Validate())
	}
	config.CustomerPolicies = map[string]CustomerPolicy{"": {}}
	s.ErrorContains(config.Validate(), "requires an app ID")
}

func (s *RoutingSuite) TestCustomerPolicyLoadsFromDeploymentYAML() {
	config, err := parseConfig([]byte(`llm:
  providers:
    - provider: meta
      model: muse-spark-1.3
      languages: [en]
  customer_policies:
    "1257545":
      allowed_models: [meta/muse-spark-1.3]
      tags:
        application: athena
        environment: development
`))
	s.Require().NoError(err)
	policy := config[LLM].CustomerPolicies["1257545"]
	s.Equal([]string{"meta/muse-spark-1.3"}, policy.AllowedModels)
	s.Equal(Tags{"application": "athena", "environment": "development"}, policy.Tags)
}

func (s *RoutingSuite) TestMergedAttributionMustFitTheTagBudgetBeforeRouting() {
	router := s.policyRouter([]string{"quick/multi"}, false)
	tags := Tags{}
	for i := range tagLimit {
		tags[fmt.Sprintf("tag-%d", i)] = "value"
	}
	_, _, err := router.Select(s.ctx, Request{CustomerID: "athena-app", Target: "quick/multi", Tags: tags})
	s.ErrorContains(err, "at most 16 tags")
}

func (s *RoutingSuite) TestRecordedAttributionOverridesSpoofedLabelsAndCopiesTags() {
	// Inspect the exact row queued for storage, without starting a database writer.
	recorder := &Recorder{modality: LLM, store: &store.Store{}, queue: make(chan store.Request, 2),
		customerPolicies: map[string]CustomerPolicy{"athena-app": {Tags: Tags{"application": "athena", "environment": "test"}}},
	}
	tags := Tags{"application": "other", "environment": "production", "feature": "chat"}
	for _, success := range []bool{true, false} {
		recorder.Record(ProviderConfig{Provider: "meta", Model: "muse-spark-1.3"}, Stat{Owner: Owner{CustomerID: "athena-app", Tags: tags}, Success: success})
	}
	tags["feature"] = "changed-after-enqueue"
	for range 2 {
		row := <-recorder.queue
		s.Equal(map[string]string{"application": "athena", "environment": "test", "feature": "chat"}, row.Tags)
	}
	s.Equal("other", tags["application"], "recording must not mutate caller-owned input")
}
