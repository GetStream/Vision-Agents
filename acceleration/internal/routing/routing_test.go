package routing

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

// stubProvider stands in for a real provider so routing decisions can be tested without
// credentials. startErr makes it refuse to start, which is what failover reacts to.
type stubProvider struct {
	provider string
	model    string
	startErr error
	closed   bool
}

func (s *stubProvider) Start(context.Context) error { return s.startErr }
func (s *stubProvider) Close() error                { s.closed = true; return nil }
func (s *stubProvider) Provider() string            { return s.provider }
func (s *stubProvider) Model() string               { return s.model }

type RoutingSuite struct {
	suite.Suite
	ctx context.Context
}

func TestRoutingSuite(t *testing.T) {
	suite.Run(t, new(RoutingSuite))
}

func (s *RoutingSuite) SetupTest() {
	s.ctx = context.Background()
}

// config is a small capability set that keeps the routing assertions readable.
func (s *RoutingSuite) config() ModalityConfig {
	return ModalityConfig{
		Providers: []ProviderConfig{
			{Provider: "quick", Model: "en", Languages: []string{"en"}, Realtime: true, Tier: LowLatency},
			{Provider: "quick", Model: "multi", Languages: []string{"en", "es"}, Realtime: true, Tier: LowLatency},
			{Provider: "lush", Model: "multi", Languages: []string{"en", "es", "de"}, Realtime: true, Tier: HighQuality},
			{Provider: "batchy", Model: "offline-en", Languages: []string{"en"}},
		},
		Aliases: map[string]Alias{
			"en-low-latency":           {Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency},
			"multilingual-low-latency": {Multilingual: true, RequireRealtime: true, Tier: LowLatency},
			"en-high-accuracy":         {Languages: []string{"en"}},
			"multilingual-quality":     {Multilingual: true, Tier: HighQuality},
		},
	}
}

// newRouter returns a router with no stats backends, so routing decisions are the only
// thing under test. Every configured provider builds and starts.
func (s *RoutingSuite) newRouter() *Router[*stubProvider] {
	return s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model}, nil
	})
}

func (s *RoutingSuite) newRouterWith(factory Factory[*stubProvider]) *Router[*stubProvider] {
	registry := NewRegistry[*stubProvider]()
	for _, provider := range []string{"quick", "lush", "batchy"} {
		registry.Register(provider, factory)
	}

	router, err := New(Options[*stubProvider]{
		Modality: STT,
		Config:   s.config(),
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// names reduces candidates to their registry names for easy assertions.
func names(candidates []Candidate) []string {
	result := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		result = append(result, candidate.Config.Name())
	}
	return result
}

func (s *RoutingSuite) TestNewRequiresAModality() {
	_, err := New(Options[*stubProvider]{Config: s.config(), Registry: NewRegistry[*stubProvider]()})
	s.ErrorContains(err, "modality is required")
}

func (s *RoutingSuite) TestNewRequiresARegistry() {
	_, err := New(Options[*stubProvider]{Modality: STT, Config: s.config()})
	s.ErrorContains(err, "registry is required")
}

func (s *RoutingSuite) TestNewRejectsAnInvalidConfig() {
	_, err := New(Options[*stubProvider]{
		Modality: STT,
		Config:   ModalityConfig{},
		Registry: NewRegistry[*stubProvider](),
	})
	s.ErrorContains(err, "at least one provider")
}

func (s *RoutingSuite) TestLoadConfigFallsBackToTheBuiltInDefault() {
	fromEmptyPath, err := LoadConfig("")
	s.Require().NoError(err)

	builtIn, err := DefaultConfig()
	s.Require().NoError(err)
	s.Equal(builtIn, fromEmptyPath)
}

func (s *RoutingSuite) TestLoadConfigReportsAMissingFile() {
	_, err := LoadConfig(filepath.Join(s.T().TempDir(), "absent.yaml"))
	s.ErrorContains(err, "read config")
}

func (s *RoutingSuite) TestDefaultConfigCoversEveryModality() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	s.Contains(config, STT)
	s.Contains(config, TTS)
}

func (s *RoutingSuite) TestDefaultConfigDeclaresTheSameShortcutsForEveryModality() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for modality := range config {
		for _, alias := range []string{
			"en-low-latency",
			"multilingual-low-latency",
			"en-high-accuracy",
			"multilingual-high-accuracy",
		} {
			s.Containsf(config[modality].Aliases, alias, "%s should offer %s", modality, alias)
		}
	}
}

func (s *RoutingSuite) TestDefaultConfigPricesEveryProvider() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for modality, section := range config {
		for _, provider := range section.Providers {
			s.NotZerof(provider.Price, "%s/%s has no price, so its cost would report as zero",
				modality, provider.Name())
		}
	}
}

func (s *RoutingSuite) TestConfigRejectsDuplicateProviders() {
	config := ModalityConfig{Providers: []ProviderConfig{
		{Provider: "quick", Model: "en", Languages: []string{"en"}},
		{Provider: "quick", Model: "en", Languages: []string{"en"}},
	}}

	s.ErrorContains(config.Validate(), "is declared twice")
}

func (s *RoutingSuite) TestConfigRejectsProvidersWithoutLanguages() {
	config := ModalityConfig{Providers: []ProviderConfig{{Provider: "quick", Model: "en"}}}

	s.ErrorContains(config.Validate(), "declares no languages")
}

func (s *RoutingSuite) TestConfigRejectsAnUnknownTier() {
	config := ModalityConfig{Providers: []ProviderConfig{
		{Provider: "quick", Model: "en", Languages: []string{"en"}, Tier: "cheapest"},
	}}

	s.ErrorContains(config.Validate(), `unknown tier "cheapest"`)
}

func (s *RoutingSuite) TestConfigRejectsAnAliasNoProviderCanServe() {
	config := s.config()
	config.Aliases["klingon-best"] = Alias{Languages: []string{"tlh"}}

	s.ErrorContains(config.Validate(), "alias klingon-best matches no provider")
}

func (s *RoutingSuite) TestConfigRejectsAModalityWithNoProviders() {
	config := Config{TTS: ModalityConfig{}}

	s.ErrorContains(config.Validate(), "tts: config must declare at least one provider")
}

func (s *RoutingSuite) TestConfigRejectsBeingEmpty() {
	s.ErrorContains(Config{}.Validate(), "at least one modality")
}

func (s *RoutingSuite) TestConcreteTargetResolvesToItself() {
	candidates, err := s.newRouter().Resolve(s.ctx, "quick/en", nil)
	s.Require().NoError(err)

	s.Equal([]string{"quick/en"}, names(candidates))
}

func (s *RoutingSuite) TestUnknownTargetIsRejected() {
	_, err := s.newRouter().Resolve(s.ctx, "quick/does-not-exist", nil)
	s.ErrorContains(err, `unknown target "quick/does-not-exist"`)
}

func (s *RoutingSuite) TestEmptyTargetIsRejected() {
	_, err := s.newRouter().Resolve(s.ctx, "", nil)
	s.ErrorContains(err, "target is required")
}

func (s *RoutingSuite) TestRealtimeAliasExcludesOfflineModels() {
	candidates, err := s.newRouter().Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)

	s.Equal([]string{"quick/en", "quick/multi"}, names(candidates),
		"an offline model cannot serve live audio and a quality model is the wrong tier")
}

func (s *RoutingSuite) TestAnAliasWithoutATierAcceptsEveryModel() {
	candidates, err := s.newRouter().Resolve(s.ctx, "en-high-accuracy", nil)
	s.Require().NoError(err)

	s.Contains(names(candidates), "batchy/offline-en", "accuracy is worth waiting for")
	s.Contains(names(candidates), "lush/multi")
	s.Len(candidates, 4)
}

func (s *RoutingSuite) TestTierNarrowsTheCandidates() {
	candidates, err := s.newRouter().Resolve(s.ctx, "multilingual-quality", nil)
	s.Require().NoError(err)

	s.Equal([]string{"lush/multi"}, names(candidates), "only one model is high-quality")
}

func (s *RoutingSuite) TestMultilingualAliasExcludesSingleLanguageModels() {
	candidates, err := s.newRouter().Resolve(s.ctx, "multilingual-low-latency", nil)
	s.Require().NoError(err)

	s.Equal([]string{"quick/multi"}, names(candidates), "an English-only model is not multilingual")
}

func (s *RoutingSuite) TestLanguageHintsNarrowTheCandidates() {
	candidates, err := s.newRouter().Resolve(s.ctx, "en-high-accuracy", []string{"de"})
	s.Require().NoError(err)

	s.Equal([]string{"lush/multi"}, names(candidates), "only one model speaks German")
}

func (s *RoutingSuite) TestLanguageHintsAreCaseInsensitive() {
	candidates, err := s.newRouter().Resolve(s.ctx, "en-high-accuracy", []string{"DE"})
	s.Require().NoError(err)

	s.Equal([]string{"lush/multi"}, names(candidates))
}

func (s *RoutingSuite) TestUnservableLanguageIsRejected() {
	_, err := s.newRouter().Resolve(s.ctx, "en-low-latency", []string{"ja"})
	s.ErrorContains(err, `no provider satisfies "en-low-latency" for languages ja`)
}

func (s *RoutingSuite) TestSelectRequiresACustomer() {
	_, _, err := s.newRouter().Select(s.ctx, Request{Target: "en-low-latency"})
	s.ErrorContains(err, "customer id is required")
}

func (s *RoutingSuite) TestSelectRejectsTagsTheRollupsCannotCarry() {
	_, _, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Tags:       Tags{"not a key": "moderation"},
	})
	s.ErrorContains(err, "tag key")
}

func (s *RoutingSuite) TestTagsAcceptWhateverKeysTheCustomerChooses() {
	s.NoError(Tags{"customer_id": "123", "project": "moderation", "environment": "dev"}.Validate())
}

func (s *RoutingSuite) TestTagsRejectTooManyLabels() {
	tags := Tags{}
	for i := range tagLimit + 1 {
		tags[fmt.Sprintf("key%d", i)] = "value"
	}
	s.ErrorContains(tags.Validate(), "at most 16 tags")
}

func (s *RoutingSuite) TestTagsRejectAnOversizedValue() {
	s.ErrorContains(Tags{"project": strings.Repeat("x", tagValueLimit+1)}.Validate(),
		`tag "project" is longer than 256 characters`)
}

func (s *RoutingSuite) TestSelectReturnsTheBestCandidate() {
	provider, config, err := s.newRouter().Select(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	s.Equal("quick/en", config.Name())
	s.Equal("en", provider.Model(), "the model comes from the candidate's config")
}

func (s *RoutingSuite) TestSelectPassesTheRequestThroughToTheFactory() {
	var got Spec
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		got = spec
		return &stubProvider{model: spec.Model}, nil
	})

	_, _, err := router.Select(s.ctx, Request{
		CustomerID:    "acme",
		Target:        "multilingual-low-latency",
		LanguageHints: []string{"es"},
		Voice:         "cherry",
	})
	s.Require().NoError(err)

	s.Equal("multi", got.Model)
	s.Equal([]string{"es"}, got.LanguageHints)
	s.Equal("cherry", got.Voice)
}

func (s *RoutingSuite) TestSelectFailsOverToTheNextCandidate() {
	// The first candidate in config order refuses to start, so the second must serve.
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
		}
		return &stubProvider{model: spec.Model}, nil
	})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	s.Equal("quick/multi", config.Name())
}

func (s *RoutingSuite) TestSelectReportsEveryFailureWhenNoCandidateStarts() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
	})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.ErrorContains(err, "every candidate")
	s.ErrorContains(err, "quick/en: upstream is down")
	s.ErrorContains(err, "quick/multi: upstream is down")
}

func (s *RoutingSuite) TestSelectFailsWhenNoCandidateCanBeBuilt() {
	// "batchy" has capabilities declared but no factory, so it can never serve a request.
	router, err := New(Options[*stubProvider]{
		Modality: STT,
		Config: ModalityConfig{
			Providers: []ProviderConfig{
				{Provider: "batchy", Model: "offline-en", Languages: []string{"en"}},
			},
			Aliases: map[string]Alias{"en-high-accuracy": {Languages: []string{"en"}}},
		},
		Registry: NewRegistry[*stubProvider](),
	})
	s.Require().NoError(err)
	defer router.Close()

	_, _, err = router.Select(s.ctx, Request{CustomerID: "acme", Target: "en-high-accuracy"})
	s.ErrorContains(err, "every candidate")
	s.ErrorContains(err, "no factory registered")
}

func (s *RoutingSuite) TestSelectClosesAProviderThatFailedToStart() {
	var built *stubProvider
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		built = &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}
		return built, nil
	})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})
	s.Require().Error(err)
	s.True(built.closed, "a provider that failed to start must not leak its connection")
}

func (s *RoutingSuite) TestProvidersListsEveryConfiguredModelInOrder() {
	providers := s.newRouter().Providers(s.ctx)

	s.Equal([]string{"quick/en", "quick/multi", "lush/multi", "batchy/offline-en"}, names(providers))
	s.True(providers[0].Health.Available, "an unmeasured provider is available")
}

func (s *RoutingSuite) TestRankPrefersAvailableProviders() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "a", Model: "m"}, Health: live.Health{Requests: 10, Errors: 9, Available: false}},
		{Config: ProviderConfig{Provider: "b", Model: "m"}, Health: live.Health{Requests: 10, Errors: 1, Available: true}},
	}

	rank(candidates)

	s.Equal([]string{"b/m", "a/m"}, names(candidates))
}

func (s *RoutingSuite) TestRankPrefersFewerErrorsBeforeLatency() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "slow-but-reliable", Model: "m"}, Health: live.Health{Requests: 10, Errors: 0, LatencyMsAvg: 500, Available: true}},
		{Config: ProviderConfig{Provider: "fast-but-flaky", Model: "m"}, Health: live.Health{Requests: 10, Errors: 4, LatencyMsAvg: 50, Available: true}},
	}

	rank(candidates)

	s.Equal([]string{"slow-but-reliable/m", "fast-but-flaky/m"}, names(candidates))
}

func (s *RoutingSuite) TestRankBreaksTiesOnLatency() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "slower", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 300, Available: true}},
		{Config: ProviderConfig{Provider: "faster", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 100, Available: true}},
	}

	rank(candidates)

	s.Equal([]string{"faster/m", "slower/m"}, names(candidates))
}

func (s *RoutingSuite) TestRankKeepsConfigOrderForUnmeasuredProviders() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "first", Model: "m"}, Health: live.Health{Available: true}},
		{Config: ProviderConfig{Provider: "second", Model: "m"}, Health: live.Health{Available: true}},
	}

	rank(candidates)

	s.Equal([]string{"first/m", "second/m"}, names(candidates),
		"a provider with no history should not win on an unmeasured zero latency")
}

func (s *RoutingSuite) TestRankPrefersAMeasuredProviderOverAnUnmeasuredOne() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "unmeasured", Model: "m"}, Health: live.Health{Available: true}},
		{Config: ProviderConfig{Provider: "known-good", Model: "m"}, Health: live.Health{Requests: 100, LatencyMsAvg: 80, Available: true}},
	}

	rank(candidates)

	s.Equal([]string{"known-good/m", "unmeasured/m"}, names(candidates))
}

func (s *RoutingSuite) TestRegistryBuildsRegisteredProvidersOnly() {
	registry := NewRegistry[*stubProvider]()
	registry.Register("quick", func(spec Spec) (*stubProvider, error) {
		return &stubProvider{provider: "quick", model: spec.Model}, nil
	})

	s.True(registry.Has("quick"))
	s.False(registry.Has("batchy"))

	built, err := registry.Build("quick", Spec{Model: "en"})
	s.Require().NoError(err)
	s.Equal("en", built.Model())

	_, err = registry.Build("batchy", Spec{Model: "offline-en"})
	s.ErrorContains(err, `no factory registered for provider "batchy"`)
}

func (s *RoutingSuite) TestRegistryRejectsAFactoryThatBuildsNothing() {
	registry := NewRegistry[Provider]()
	registry.Register("empty", func(Spec) (Provider, error) { return nil, nil })

	_, err := registry.Build("empty", Spec{Model: "m"})
	s.ErrorContains(err, "factory returned no provider")
}

func (s *RoutingSuite) TestCostIsZeroWithoutAPrice() {
	s.Zero(Price{}.CostMicros(Usage{Characters: 1000, AudioMs: 60_000, InputTokens: 500, OutputTokens: 500}))
}

func (s *RoutingSuite) TestCostPricesCharacters() {
	// A million characters at $50 per million is $50, which is 50 million micros.
	price := Price{PerMillionChars: 50}

	s.EqualValues(50_000_000, price.CostMicros(Usage{Characters: 1_000_000}))
	s.EqualValues(50_000, price.CostMicros(Usage{Characters: 1_000}), "a thousand characters is five cents")
}

func (s *RoutingSuite) TestCostPricesAudioByTheHour() {
	// An hour of audio at $0.36 per hour is 360000 micros; a minute is a sixtieth of that.
	price := Price{PerAudioHour: 0.36}

	s.EqualValues(360_000, price.CostMicros(Usage{AudioMs: 3_600_000}))
	s.EqualValues(6_000, price.CostMicros(Usage{AudioMs: 60_000}))
}

func (s *RoutingSuite) TestCostPricesTokensByDirection() {
	// DeepSeek Flash: $0.13 per million in, $0.26 per million out.
	price := Price{PerMillionInputTokens: 0.13, PerMillionOutputTokens: 0.26}

	s.EqualValues(130_000, price.CostMicros(Usage{InputTokens: 1_000_000}))
	s.EqualValues(260_000, price.CostMicros(Usage{OutputTokens: 1_000_000}))
	s.EqualValues(390_000, price.CostMicros(Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}),
		"output is dearer than input, so the direction has to be tracked separately")
}

func (s *RoutingSuite) TestCostBillsCachedPromptTokensOnlyOnce() {
	// $1 per million fresh, a tenth of that cached. Half of a million-token prompt was
	// cached, so the bill is half a dollar plus five cents rather than the full dollar.
	price := Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.1}

	s.EqualValues(550_000, price.CostMicros(Usage{InputTokens: 1_000_000, CachedInputTokens: 500_000}))
	s.EqualValues(100_000, price.CostMicros(Usage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000}),
		"a prompt served entirely from cache costs only the cached rate")
}

func (s *RoutingSuite) TestCostIgnoresMoreCachedTokensThanPromptTokens() {
	// A provider that reported nonsense should not produce a negative bill.
	price := Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.1}

	s.EqualValues(100_000, price.CostMicros(Usage{InputTokens: 100, CachedInputTokens: 1_000_000}))
}

func (s *RoutingSuite) TestCostAddsEveryUnitTheModelBillsFor() {
	price := Price{PerMillionChars: 50, PerAudioHour: 0.36}

	s.EqualValues(410_000, price.CostMicros(Usage{Characters: 1_000, AudioMs: 3_600_000}),
		"an hour of audio plus a thousand characters")
}

func (s *RoutingSuite) TestTierDefaultsToLowLatency() {
	s.Equal(LowLatency, ProviderConfig{}.tier())
	s.Equal(HighQuality, ProviderConfig{Tier: HighQuality}.tier())
}
