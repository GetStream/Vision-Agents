package llmrouter

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// The keys the vendor of these tests is asked with: the deployment's own, read from the
// variable its instances name, and a customer's.
const (
	deploymentKey    = "sk-ant-deployment-0000000000000"
	euDeploymentKey  = "sk-ant-deployment-eu-000000000"
	customerKey      = "sk-ant-customer-11111111111111"
	deploymentKeyEnv = "BYOK_TEST_DEPLOYMENT_KEY"
	euKeyEnv         = "BYOK_TEST_EU_DEPLOYMENT_KEY"
	haiku            = "claude-haiku-5-5"
)

// keptKeys are the keys customers keep, by customer, provider and region.
type keptKeys map[string]string

func (k keptKeys) CustomerKey(_ context.Context, customerID, provider, region string) (string, bool, error) {
	key, ok := k[customerID+"/"+provider+"/"+region]
	return key, ok, nil
}

// vendor is an Anthropic Messages API that answers every key but the ones it refuses, and
// remembers the key each request carried.
type vendor struct {
	*httptest.Server
	mu   sync.Mutex
	keys []string
}

func startVendor(t *testing.T, refused ...string) *vendor {
	v := &vendor{}
	v.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Header.Get("X-Api-Key")
		v.mu.Lock()
		v.keys = append(v.keys, key)
		v.mu.Unlock()
		if slices.Contains(refused, key) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, claudeAnswer("hello", 5, 2))
	}))
	t.Cleanup(v.Close)
	return v
}

// asked is every key the vendor was asked with, in order.
func (v *vendor) asked() []string {
	v.mu.Lock()
	defer v.mu.Unlock()
	return slices.Clone(v.keys)
}

// keyedConfig declares Claude Haiku in the us at the vendor, on the deployment's key, and the
// instances of extra after it.
func keyedConfig(t *testing.T, at *vendor, extra ...routing.ProviderConfig) routing.ModalityConfig {
	t.Setenv(deploymentKeyEnv, deploymentKey)
	providers := []routing.ProviderConfig{{Provider: anthropic.ProviderName, Model: haiku, Languages: []string{"en"},
		Endpoint: at.URL, APIKeyEnv: deploymentKeyEnv}}
	return routing.ModalityConfig{Providers: append(providers, extra...)}
}

// answeredFor asks the router for one response of the config's first model for customerID.
func answeredFor(ctx context.Context, t *testing.T, router *Router, customerID string) llm.Response {
	return answeredIn(ctx, t, router, customerID, "")
}

// answeredIn is answeredFor for work done in region.
func answeredIn(ctx context.Context, t *testing.T, router *Router, customerID, region string) llm.Response {
	session, err := router.Start(ctx, Request{CustomerID: customerID, Region: region, Target: router.Config().Providers[0].Name()})
	require.NoError(t, err)
	defer session.Close()
	stream, err := session.Create(ctx, llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hello"}}})
	require.NoError(t, err)
	response, err := llm.Collect(stream)
	require.NoError(t, err)
	return response
}

// keyedRouter routes config over the real vendor factories, the customer's keys being keys,
// held to gate.
func (s *LLMRouterSuite) keyedRouter(config routing.ModalityConfig, keys routing.CustomerKeys, gate routing.Gate) *Router {
	router, err := New(Options{Registry: DefaultRegistry(), Config: config, Keys: keys, Gate: gate})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// answered asks the router for one response to Haiku for acme, and returns it.
func (s *LLMRouterSuite) answered(router *Router) llm.Response {
	return answeredFor(s.ctx, s.T(), router, "acme")
}

func (s *LLMRouterSuite) TestAModelIsAskedOnTheCustomersKeyImmediatelyBeforeTheDeploymentsInstance() {
	at := startVendor(s.T())
	router := s.keyedRouter(keyedConfig(s.T(), at), keptKeys{"acme/anthropic/us": customerKey}, nil)

	candidates, err := router.Candidates(s.ctx, routing.Request{CustomerID: "acme", Target: "anthropic/" + haiku})
	s.Require().NoError(err)
	response := s.answered(router)

	s.Require().Len(candidates, 2)
	s.True(candidates[0].Config.BYOK, "the customer's instance first")
	s.False(candidates[1].Config.BYOK, "the deployment's right after it")
	s.Equal(at.URL, candidates[0].Config.Endpoint, "at the deployment's endpoint: a customer gives a key, never an address")
	s.Equal("hello", response.OutputText)
	s.Equal([]string{customerKey}, at.asked())
}

func (s *LLMRouterSuite) TestTheCustomersPriorityBetweenVendorsIsKeptWithTheirKeyFirstForEach() {
	at := startVendor(s.T())
	s.T().Setenv("BYOK_TEST_OPENAI_KEY", "sk-deployment-openai")
	config := keyedConfig(s.T(), at, routing.ProviderConfig{Provider: "openai", Model: "gpt-5.6-luna", Languages: []string{"en"},
		Endpoint: at.URL, APIKeyEnv: "BYOK_TEST_OPENAI_KEY"})
	router := s.keyedRouter(config, keptKeys{"acme/anthropic/us": customerKey, "acme/openai/us": "sk-customer-openai"}, nil)

	candidates, err := router.Candidates(s.ctx, routing.Request{CustomerID: "acme", Providers: []string{"openai/gpt-5.6-luna", "anthropic/" + haiku}})

	s.Require().NoError(err)
	var walked []string
	for _, candidate := range candidates {
		walked = append(walked, fmt.Sprintf("%s byok=%t", candidate.Config.Name(), candidate.Config.BYOK))
	}
	s.Equal([]string{"openai/gpt-5.6-luna byok=true", "openai/gpt-5.6-luna byok=false",
		"anthropic/claude-haiku-5-5 byok=true", "anthropic/claude-haiku-5-5 byok=false"}, walked)
}

func (s *LLMRouterSuite) TestARefusedCustomerKeyFallsBackToTheDeploymentsInstance() {
	at := startVendor(s.T(), customerKey)
	router := s.keyedRouter(keyedConfig(s.T(), at), keptKeys{"acme/anthropic/us": customerKey}, nil)

	response := s.answered(router)

	s.Equal("hello", response.OutputText)
	s.Equal([]string{customerKey, deploymentKey}, at.asked(), "the customer's instance, then the deployment's")
}

func (s *LLMRouterSuite) TestACustomersKeyForTheEUIsNotUsedForAUSInstance() {
	at := startVendor(s.T())
	router := s.keyedRouter(keyedConfig(s.T(), at), keptKeys{"acme/anthropic/eu": customerKey}, nil)

	s.answered(router)

	s.Equal([]string{deploymentKey}, at.asked())
}

func (s *LLMRouterSuite) TestACustomersKeyForTheUSIsNotUsedForAnEUInstance() {
	at := startVendor(s.T())
	s.T().Setenv(euKeyEnv, euDeploymentKey)
	config := keyedConfig(s.T(), at, routing.ProviderConfig{Provider: anthropic.ProviderName, Model: haiku, Languages: []string{"en"},
		Region: pricing.RegionEU, Endpoint: at.URL, APIKeyEnv: euKeyEnv})
	router := s.keyedRouter(config, keptKeys{"acme/anthropic/us": customerKey}, nil)

	answeredIn(s.ctx, s.T(), router, "acme", pricing.RegionEU)

	s.Equal([]string{euDeploymentKey}, at.asked(), "only the eu instance serves, on the deployment's eu key")
}

func (s *LLMRouterSuite) TestACustomersKeyForTheEUIsUsedForTheEUInstance() {
	at := startVendor(s.T())
	s.T().Setenv(euKeyEnv, euDeploymentKey)
	config := keyedConfig(s.T(), at, routing.ProviderConfig{Provider: anthropic.ProviderName, Model: haiku, Languages: []string{"en"},
		Region: pricing.RegionEU, Endpoint: at.URL, APIKeyEnv: euKeyEnv})
	router := s.keyedRouter(config, keptKeys{"acme/anthropic/eu": customerKey}, nil)

	answeredIn(s.ctx, s.T(), router, "acme", pricing.RegionEU)

	s.Equal([]string{customerKey}, at.asked())
}

func (s *LLMRouterSuite) TestACustomersKeyIsLookedUpForTheWorksRegionAlone() {
	at := startVendor(s.T())
	s.T().Setenv(euKeyEnv, euDeploymentKey)
	config := keyedConfig(s.T(), at, routing.ProviderConfig{Provider: anthropic.ProviderName, Model: haiku, Languages: []string{"en"},
		Region: pricing.RegionEU, Endpoint: at.URL, APIKeyEnv: euKeyEnv})
	keys := &lookedUp{kept: keptKeys{"acme/anthropic/us": customerKey, "acme/anthropic/eu": customerKey}}
	router := s.keyedRouter(config, keys, nil)

	candidates, err := router.Candidates(s.ctx, routing.Request{CustomerID: "acme", Region: pricing.RegionEU, Target: "anthropic/" + haiku})

	s.Require().NoError(err)
	s.Equal([]string{"anthropic/eu"}, keys.asked(), "an instance in the us never serves eu work, so its key is never read")
	var own []string
	for _, candidate := range candidates {
		if candidate.Config.BYOK {
			own = append(own, candidate.Config.Region)
		}
	}
	s.Equal([]string{pricing.RegionEU}, own)
}

// lookedUp is kept keys that remember each provider and region they were asked for.
type lookedUp struct {
	kept keptKeys
	mu   sync.Mutex
	seen []string
}

func (k *lookedUp) CustomerKey(ctx context.Context, customerID, provider, region string) (string, bool, error) {
	k.mu.Lock()
	k.seen = append(k.seen, provider+"/"+region)
	k.mu.Unlock()
	return k.kept.CustomerKey(ctx, customerID, provider, region)
}

func (k *lookedUp) asked() []string {
	k.mu.Lock()
	defer k.mu.Unlock()
	return slices.Clone(k.seen)
}

func (s *LLMRouterSuite) TestAVendorWhoseFactoryTakesNoKeyIsNeverAskedOnACustomersKey() {
	registry := NewRegistry()
	registry.Register("stub", func(routing.Spec) (Provider, error) { return Started(newStubLLM(), nil) })
	router, err := New(Options{Registry: registry, Keys: keptKeys{"acme/stub/us": customerKey}, Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "stub", Model: "stub-model", Languages: []string{"en"}}},
	}})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	candidates, err := router.Candidates(s.ctx, routing.Request{CustomerID: "acme", Target: "stub/stub-model"})

	s.Require().NoError(err)
	s.Require().Len(candidates, 1)
	s.False(candidates[0].Config.BYOK)
	s.False(router.TakesCustomerKey("stub"))
	s.True(DefaultRegistry().Honors(anthropic.ProviderName, routing.OverrideAPIKey))
}

func (s *LLMRouterSuite) TestAnotherCustomersKeyIsNeverUsed() {
	at := startVendor(s.T())
	router := s.keyedRouter(keyedConfig(s.T(), at), keptKeys{"globex/anthropic/us": customerKey}, nil)

	s.answered(router)

	s.Equal([]string{deploymentKey}, at.asked())
}
