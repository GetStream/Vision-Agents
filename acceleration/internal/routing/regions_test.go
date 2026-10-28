package routing

import (
	"log/slog"
	"os"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// euEndpoint is where the eu instances of the regional config are reached.
const euEndpoint = "https://eu.quick.example/v1"

// regionalConfig declares quick/en and steady/en in the us and in the eu, the eu ones at
// endpoints of their own, and lush/multi in the us alone. quick/en is the shortcut's pin, and
// is declared last in the eu, so only the pin puts it first there.
func (s *RoutingSuite) regionalConfig() ModalityConfig {
	return ModalityConfig{
		Providers: []ProviderConfig{
			{Provider: "lush", Model: "multi", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionUS},
			{Provider: "quick", Model: "en", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionUS},
			{Provider: "steady", Model: "en", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionUS},
			{Provider: "steady", Model: "en", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionEU, Endpoint: "https://eu.steady.example/v1"},
			{Provider: "quick", Model: "en", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionEU, Endpoint: euEndpoint},
		},
		Aliases: map[string]Alias{"en": {Languages: []string{"en"}, Prefer: "quick/en"}},
	}
}

// euQuick is where regionalConfig declares quick/en in the eu.
const euQuick = 4

// built is what each build was given, newest last.
type built struct {
	mu    sync.Mutex
	specs []Spec
}

func (b *built) factory(spec Spec) (*stubProvider, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.specs = append(b.specs, spec)
	return &stubProvider{model: spec.Model}, nil
}

func (b *built) last() Spec {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.specs[len(b.specs)-1]
}

// regionalRouter routes the regional config, held to admission, every provider's factory
// honoring honors and recording its builds in into.
func (s *RoutingSuite) regionalRouter(config ModalityConfig, admission Admission, into *built, honors ...Override) (*Router[*stubProvider], error) {
	registry := NewRegistry[*stubProvider]()
	for _, provider := range config.Providers {
		registry.Register(provider.Provider, into.factory, honors...)
	}
	return New(Options[*stubProvider]{
		Modality: LLM,
		Config:   config,
		Registry: registry,
		Gate:     admitting(admission),
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
}

func (s *RoutingSuite) startedRegional(admission Admission, into *built) *Router[*stubProvider] {
	router, err := s.regionalRouter(s.regionalConfig(), admission, into, OverrideBaseURL, OverrideAPIKey)
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *RoutingSuite) TestAModelMayBeDeclaredOncePerRegion() {
	s.NoError(s.regionalConfig().Validate())
}

func (s *RoutingSuite) TestTheSameModelTwiceInOneRegionIsRefused() {
	_, err := parseConfig([]byte(`
llm:
  providers:
    - {provider: quick, model: en, languages: [en], region: eu, endpoint: "https://eu.quick.example/v1"}
    - {provider: quick, model: en, languages: [en], region: eu, endpoint: "https://eu2.quick.example/v1"}
`))

	s.ErrorContains(err, "quick/en in eu is declared twice")
}

func (s *RoutingSuite) TestAModelLeftWithoutARegionTwiceIsDeclaredTwiceInTheDefaultRegion() {
	_, err := parseConfig([]byte(`
llm:
  providers:
    - {provider: quick, model: en, languages: [en]}
    - {provider: quick, model: en, languages: [en], region: us}
`))

	s.ErrorContains(err, "quick/en in us is declared twice")
}

func (s *RoutingSuite) TestAnEndpointThatIsNoURLIsRefused() {
	config := ModalityConfig{Providers: []ProviderConfig{{Provider: "quick", Model: "en", Languages: []string{"en"}, Endpoint: "eu.quick.example"}}}

	s.ErrorContains(config.Validate(), `declares the endpoint "eu.quick.example"`)
}

func (s *RoutingSuite) TestAKeyVariableThatCannotBeOneIsRefused() {
	config := ModalityConfig{Providers: []ProviderConfig{{Provider: "quick", Model: "en", Languages: []string{"en"}, APIKeyEnv: "QUICK KEY"}}}

	s.ErrorContains(config.Validate(), `declares api_key_env "QUICK KEY"`)
}

func (s *RoutingSuite) TestAConcreteTargetResolvesToEveryRegionsInstance() {
	router := s.startedRegional(Admission{}, &built{})

	candidates, err := router.Resolve(s.ctx, "quick/en", nil)

	s.Require().NoError(err)
	s.Require().Len(candidates, 2)
	s.Equal(pricing.RegionUS, candidates[0].Config.Region)
	s.Equal(pricing.RegionEU, candidates[1].Config.Region)
}

func (s *RoutingSuite) TestARequestAskingForTheEUIsServedByTheEUInstanceAtItsEndpoint() {
	builds := &built{}
	router := s.startedRegional(Admission{}, builds)

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en",
		DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})

	s.Require().NoError(err)
	s.Equal(pricing.RegionEU, config.Region)
	s.Equal(euEndpoint, builds.last().BaseURL)
	s.Equal(pricing.RegionEU, builds.last().Region)
}

func (s *RoutingSuite) TestARequestAskingNothingIsServedInTheDefaultRegion() {
	builds := &built{}
	router := s.startedRegional(Admission{}, builds)

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})

	s.Require().NoError(err)
	s.Equal(pricing.RegionUS, config.Region)
	s.Empty(builds.last().BaseURL, "the us instance is the vendor's default host")
	s.Equal(pricing.RegionUS, builds.last().Region)
}

func (s *RoutingSuite) TestARegionNoInstanceRunsInIsRefusedNamingIt() {
	router := s.startedRegional(Admission{}, &built{})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "lush/multi",
		DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})

	s.Require().ErrorIs(err, ErrRequest, "the caller's to change, and no model's health pays for it")
	s.ErrorContains(err, "no candidate in region eu")
}

func (s *RoutingSuite) TestAPriorityListIsNeverServedInAnotherRegion() {
	router := s.startedRegional(Admission{}, &built{})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"lush", "quick/en"},
		DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})

	s.Require().NoError(err)
	s.Equal("quick/en", config.Name(), "lush is first in the list and runs in the us alone")
	s.Equal(pricing.RegionEU, config.Region)
}

func (s *RoutingSuite) TestAShortcutsPinHoldsInEveryRegion() {
	router := s.startedRegional(Admission{}, &built{})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "en",
		DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})
	s.Require().NoError(err)
	s.Equal("quick/en", config.Name())
	s.Equal(pricing.RegionEU, config.Region)

	_, config, err = router.Select(s.ctx, Request{CustomerID: "acme", Target: "en"})
	s.Require().NoError(err)
	s.Equal("quick/en", config.Name(), "the pin goes ahead of lush in the us too")
	s.Equal(pricing.RegionUS, config.Region)
}

func (s *RoutingSuite) TestTheWorksRegionHoldsARequestItsDataPolicyAsksNoneOf() {
	builds := &built{}
	router := s.startedRegional(Admission{}, builds)

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en", Region: pricing.RegionEU})

	s.Require().NoError(err)
	s.Equal(pricing.RegionEU, config.Region)
	s.Equal(euEndpoint, builds.last().BaseURL)
}

func (s *RoutingSuite) TestAPoliciesFloorNamingARegionMovesNoWork() {
	router := s.startedRegional(Admission{DataPolicy: options.DataPolicy{Region: pricing.RegionEU}}, &built{})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})

	s.Require().NoError(err)
	s.Equal(pricing.RegionUS, config.Region, "where work is done is never an admission's to say")
}

func (s *RoutingSuite) TestTheWorksRegionAndAnotherAskedByItsDataPolicyAreRefusedNamingBoth() {
	router := s.startedRegional(Admission{}, &built{})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en", Region: pricing.RegionEU,
		DataPolicy: options.DataPolicy{Region: pricing.RegionUS}})

	s.Require().ErrorIs(err, options.ErrRegionConflict)
	s.ErrorIs(err, ErrRequest)
	s.ErrorContains(err, "the work is done in eu and its data policy asks for us")
}

func (s *RoutingSuite) TestTheWorksRegionAndTheSameAskedByItsDataPolicyAreServedThere() {
	router := s.startedRegional(Admission{}, &built{})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en", Region: pricing.RegionEU,
		DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})

	s.Require().NoError(err)
	s.Equal(pricing.RegionEU, config.Region)
}

func (s *RoutingSuite) TestARequestsOwnerCarriesTheRegionItsWorkIsDoneIn() {
	s.Equal(pricing.RegionEU, Request{Region: pricing.RegionEU}.Owner().Region)
	s.Equal(pricing.RegionEU, Request{DataPolicy: options.DataPolicy{Region: pricing.RegionEU}}.Owner().Region,
		"a direct call's own data policy")
	s.Empty(Request{}.Owner().Region, "nothing named is left for the default region")
}

func (s *RoutingSuite) TestATargetIsInARegionWhenOneOfItsInstancesIs() {
	router := s.startedRegional(Admission{}, &built{})

	for _, target := range []string{"quick/en", "en", "steady"} {
		in, err := router.InRegion(s.ctx, target, pricing.RegionEU)
		s.Require().NoError(err, target)
		s.True(in, "%s has an instance in the eu", target)
	}
	in, err := router.InRegion(s.ctx, "lush/multi", pricing.RegionEU)
	s.Require().NoError(err)
	s.False(in, "lush/multi runs in the us alone")
	in, err = router.InRegion(s.ctx, "lush/multi", "")
	s.Require().NoError(err)
	s.True(in, "no region named is the default one")
	in, err = router.InRegion(s.ctx, "custom/prod", pricing.RegionEU)
	s.Require().NoError(err)
	s.False(in, "a customer's own model runs in the us alone")
	_, err = router.InRegion(s.ctx, "nobody/nothing", pricing.RegionEU)
	s.Error(err, "a target that names nothing is the resolution's error")
}

func (s *RoutingSuite) TestAnInstanceOutsideTheDefaultRegionWithNoEndpointIsRefusedAtStart() {
	config := s.regionalConfig()
	config.Providers[euQuick].Endpoint = ""

	_, err := s.regionalRouter(config, Admission{}, &built{}, OverrideBaseURL)

	s.ErrorContains(err, "quick/en in eu sets no endpoint")
}

func (s *RoutingSuite) TestAFactoryThatPicksTheRegionsHostNeedsNoEndpoint() {
	config := s.regionalConfig()
	config.Providers[euQuick].Endpoint = ""

	_, err := s.regionalRouter(config, Admission{}, &built{}, OverrideBaseURL, OverrideRegion)

	s.NoError(err)
}

func (s *RoutingSuite) TestAnEndpointTheFactoryDoesNotTakeIsRefusedAtStart() {
	_, err := s.regionalRouter(s.regionalConfig(), Admission{}, &built{}, OverrideAPIKey)

	s.ErrorContains(err, "steady/en in eu sets an endpoint, which the steady factory does not take")
}

func (s *RoutingSuite) TestAKeyVariableTheFactoryDoesNotTakeIsRefusedAtStart() {
	config := s.regionalConfig()
	config.Providers[euQuick].APIKeyEnv = "QUICK_EU_KEY"
	s.T().Setenv("QUICK_EU_KEY", "eu-key")

	_, err := s.regionalRouter(config, Admission{}, &built{}, OverrideBaseURL)

	s.ErrorContains(err, "quick/en in eu sets api_key_env, which the quick factory does not take")
}

func (s *RoutingSuite) TestAKeyVariableThatIsUnsetIsRefusedAtStartWithoutAValue() {
	config := s.regionalConfig()
	config.Providers[euQuick].APIKeyEnv = "QUICK_EU_KEY_UNSET"
	s.T().Setenv("QUICK_EU_KEY_UNSET", "")

	_, err := s.regionalRouter(config, Admission{}, &built{}, OverrideBaseURL, OverrideAPIKey)

	s.ErrorContains(err, "quick/en in eu names its key in QUICK_EU_KEY_UNSET, which is not set")
}

func (s *RoutingSuite) TestAnInstancesKeyReachesItsFactoryAndNoOtherInstances() {
	config := s.regionalConfig()
	config.Providers[euQuick].APIKeyEnv = "QUICK_EU_KEY"
	s.T().Setenv("QUICK_EU_KEY", "eu-key")
	builds := &built{}
	router, err := s.regionalRouter(config, Admission{}, builds, OverrideBaseURL, OverrideAPIKey)
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	_, _, err = router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en", DataPolicy: options.DataPolicy{Region: pricing.RegionEU}})
	s.Require().NoError(err)
	s.Equal("eu-key", builds.last().APIKey)

	_, _, err = router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})
	s.Require().NoError(err)
	s.Empty(builds.last().APIKey, "the us instance takes the vendor's own variable")
}

func (s *RoutingSuite) TestAPolicyAskingForARegionNothingRunsInIsNotMet() {
	s.False(s.regionalConfig().Meets(options.DataPolicy{Region: "ap"}))
	s.True(s.regionalConfig().Meets(options.DataPolicy{Region: pricing.RegionEU}))
}
