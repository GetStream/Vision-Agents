package routing

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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

// keeps, trains and silent are the data policies the four test providers declare, chosen
// so a policy assertion can tell one from another.
var (
	keeps   = options.DataHandling{TrainsOnData: options.ClaimNo, Retention: "30d"}
	discard = options.DataHandling{TrainsOnData: options.ClaimNo, Retention: options.RetentionNone}
	trains  = options.DataHandling{TrainsOnData: options.ClaimYes, Retention: options.RetentionUnspecified}
	silent  = options.DataHandling{TrainsOnData: options.ClaimUnknown, Retention: options.RetentionUnknown}
)

// config is a small capability set that keeps the routing assertions readable.
func (s *RoutingSuite) config() ModalityConfig {
	return ModalityConfig{
		Providers: []ProviderConfig{
			{Provider: "quick", Model: "en", Languages: []string{"en"}, Realtime: true, Tier: LowLatency, DataPolicy: discard},
			{Provider: "quick", Model: "multi", Languages: []string{"en", "es"}, Realtime: true, Tier: LowLatency, DataPolicy: keeps},
			{Provider: "lush", Model: "multi", Languages: []string{"en", "es", "de"}, Realtime: true, Tier: HighQuality, DataPolicy: trains},
			{Provider: "batchy", Model: "offline-en", Languages: []string{"en"}, DataPolicy: silent},
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
	return s.newRouterWithConfig(s.config(), factory)
}

func (s *RoutingSuite) newRouterWithConfig(config ModalityConfig, factory Factory[*stubProvider]) *Router[*stubProvider] {
	registry := NewRegistry[*stubProvider]()
	for _, provider := range config.Providers {
		registry.Register(provider.Provider, factory)
	}

	router, err := New(Options[*stubProvider]{
		Modality: STT,
		Config:   config,
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// prepared is the provider ids one customer's own voice "founder" has been given. Any
// other name is somebody's library voice and comes back untouched.
type prepared map[string]string

func (p prepared) ResolveVoice(_ context.Context, _, provider, voice string) (string, error) {
	if voice != "founder" {
		return voice, nil
	}
	external, ok := p[provider]
	if !ok {
		return "", ErrVoiceNotPrepared
	}
	return external, nil
}

// newRouterResolving returns a router that knows about a customer's own voices.
func (s *RoutingSuite) newRouterResolving(
	voices VoiceResolver, factory Factory[*stubProvider],
) *Router[*stubProvider] {
	registry := NewRegistry[*stubProvider]()
	for _, provider := range []string{"quick", "lush", "batchy"} {
		registry.Register(provider, factory)
	}

	router, err := New(Options[*stubProvider]{
		Modality: TTS,
		Config:   s.config(),
		Registry: registry,
		Voices:   voices,
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

func (s *RoutingSuite) TestARecordingToALiveShortcutGoesToTheRecordedOneForItsLanguages() {
	config := s.config()
	config.Aliases["en-recorded"] = Alias{Languages: []string{"en"}, RequireRecorded: true}
	config.Aliases["multilingual-recorded"] = Alias{Multilingual: true, RequireRecorded: true}

	english, err := config.Recorded("en-low-latency", nil)
	s.Require().NoError(err)
	s.Equal("en-recorded", english)
	multilingual, err := config.Recorded("multilingual-low-latency", nil)
	s.Require().NoError(err)
	s.Equal("multilingual-recorded", multilingual)
}

func (s *RoutingSuite) TestARecordingToAShortcutThatIsNotLiveOnlyStaysWhereItWasSent() {
	// en-high-accuracy takes either half here, so the batch model is already among its own.
	target, err := s.config().Recorded("en-high-accuracy", nil)
	s.Require().NoError(err)
	s.Equal("en-high-accuracy", target)
}

func (s *RoutingSuite) TestARecordingToALiveShortcutInAConfigWithNoRecordedOneAsksForABatchModel() {
	_, err := s.config().Recorded("en-low-latency", nil)
	s.EqualError(err, "routing: en-low-latency is a shortcut for live calls; name a batch model for a recording")
}

func (s *RoutingSuite) TestALiveCallPinnedToABatchModelIsRefusedAndToALiveOneOrAShortcutIsNot() {
	config := s.config()

	err := config.Live("batchy/offline-en")
	s.Require().ErrorIs(err, ErrRequest, "the caller's to change")
	s.EqualError(err, "routing: batchy/offline-en is a batch model for recordings, not live calls; for a live call name a live model")
	s.NoError(config.Live("quick/en"))
	s.NoError(config.Live("en-high-accuracy"), "a shortcut keeps to its own half")
	s.NoError(config.Live("nobody/unknown"), "routing refuses it in its own words")
}

func (s *RoutingSuite) TestALiveCallPinnedToABatchModelIsToldTheVendorsLiveModelsAndTheLiveShortcuts() {
	config := s.config()
	config.Providers = append(config.Providers, ProviderConfig{Provider: "quick", Model: "offline", Languages: []string{"en"}})
	config.Aliases["en-low-latency"] = Alias{Title: "Fast", Languages: []string{"en"}, RequireRealtime: true}
	config.Aliases["en-recorded"] = Alias{Title: "Recorded", Languages: []string{"en"}, RequireRecorded: true}

	s.EqualError(config.Live("quick/offline"),
		"routing: quick/offline is a batch model for recordings, not live calls; for a live call name quick/en or quick/multi or en-low-latency")
}

func (s *RoutingSuite) TestALiveCallOnAShortcutWithOnlyBatchCandidatesIsRefusedNamingTheLiveShortcuts() {
	// On d1654d42 Live passed every shortcut, so a live session on en-recorded dialled its
	// batch models at the vendor's streaming endpoint.
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{Title: "Fast", Languages: []string{"en"}, RequireRealtime: true}
	config.Aliases["en-recorded"] = Alias{Title: "Recorded", Languages: []string{"en"}, RequireRecorded: true}
	config.Aliases["offline"] = Alias{Only: []string{"batchy/offline-en"}}

	err := config.Live("en-recorded")
	s.Require().ErrorIs(err, ErrRequest, "the caller's to change")
	s.EqualError(err, "routing: en-recorded is a shortcut for recordings, not live calls; for a live call name en-low-latency")
	s.ErrorIs(config.Live("offline"), ErrRequest, "every candidate is a batch model, whatever the shortcut requires")
}

func (s *RoutingSuite) TestALiveCallOnAShortcutWithALiveCandidateOrNoneAtAllIsLeftToRouting() {
	config := s.config()
	config.Aliases["french"] = Alias{Languages: []string{"fr"}}

	s.NoError(config.Live("en-low-latency"))
	s.NoError(config.Live("en-high-accuracy"), "batchy/offline-en is a candidate, and so is quick/en")
	s.NoError(config.Live("french"), "nothing serves it, which routing says in its own words")
}

func (s *RoutingSuite) TestARecordingPinnedToALiveOnlyModelIsRefusedAndToABatchOneIsNot() {
	config := s.config()
	config.Aliases["en-recorded"] = Alias{Languages: []string{"en"}, RequireRecorded: true}

	_, err := config.Recorded("quick/en", nil)
	s.EqualError(err, "routing: quick/en is a model for live calls; send a recording to en-recorded, or name a batch model")
	target, err := config.Recorded("batchy/offline-en", nil)
	s.Require().NoError(err)
	s.Equal("batchy/offline-en", target)
}

func (s *RoutingSuite) TestOfferedShortcutsAreTheTitledOnesInTheOrderWritten() {
	config, err := parseConfig([]byte(`
llm:
  providers:
    - provider: a
      model: fast
      languages: [en]
  aliases:
    zeta:
      title: Zeta
    plumbing: {}
    alpha:
      title: Alpha
`))
	s.Require().NoError(err)

	s.Equal([]string{"zeta", "alpha"}, config[LLM].Offered())
}

func (s *RoutingSuite) TestTheShippedConversationShortcutComesFirst() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	s.Equal([]string{"llm-conversational", "llm-fast", "llm-smart"}, config[LLM].Offered())
	s.Equal("sts-fast", config[STS].Offered()[0])
	s.Equal([]string{"stt-fast", "stt-accurate"}, config[STT].Offered())
	s.Equal([]string{"tts-fast", "tts-quality"}, config[TTS].Offered())
	s.Equal("search-fast", config[Search].Offered()[0])
	s.NotContains(config[LLM].Offered(), "llm-flow", "the flow controller's shortcut is not a choice")
	s.NotContains(config[LLM].Offered(), "llm-judge", "a simulation's default is not a choice for a conversation")
	s.NotContains(config[LLM].Offered(), "llm-scenario-runner", "a simulation's default is not a choice for a conversation")
}

// shipped is a router over the configuration the binary ships, for the shortcuts whose
// membership is the thing being tested rather than the mechanism behind it. Every provider
// builds, since what a base group resolves to is a question about the config and not about
// whether the vendor's own factory is happy.
func (s *RoutingSuite) shipped(modality Modality) *Router[*stubProvider] {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	registry := NewRegistry[*stubProvider]()
	for _, provider := range config[modality].Providers {
		registry.Register(provider.Provider, func(spec Spec) (*stubProvider, error) {
			return &stubProvider{model: spec.Model}, nil
		})
	}

	router, err := New(Options[*stubProvider]{
		Modality: modality,
		Config:   config[modality],
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *RoutingSuite) TestABaseGroupResolvesToTheModelsVisionAgentsPicked() {
	candidates, err := s.shipped(STT).resolveChain(s.ctx, Request{
		Providers: []string{"stt-fast"},
	})
	s.Require().NoError(err)

	s.Equal("deepgram/flux-general-en", names(candidates)[0],
		"the pinned model heads the group while it is up")
	s.ElementsMatch([]string{
		"deepgram/flux-general-en",
		"deepgram/flux-general-multi",
		"cartesia/ink-2",
		"inworld/inworld-stt-1",
		"together-nemotron/nvidia/nemotron-3-asr-streaming-0.6b",
	}, names(candidates),
		"a group is the list it names and nothing else, so a model joining the router does not join it")
}

func (s *RoutingSuite) TestTheConversationalGroupLeadsWithGemmaAndKeepsLunaToFailOverTo() {
	candidates, err := s.shipped(LLM).resolveChain(s.ctx, Request{
		Providers: []string{"llm-conversational"},
	})
	s.Require().NoError(err)

	s.Equal("gemma/gemma-4-26B-A4B-it", names(candidates)[0],
		"the cheapest model that holds a short turn heads the group while it is up")
	s.ElementsMatch([]string{
		"gemma/gemma-4-26B-A4B-it",
		"cerebras/gemma-4-31b",
		"openai/gpt-5.6-luna",
	}, names(candidates),
		"Luna stays in the group, so a turn still has a model when both Gemmas are down")
}

func (s *RoutingSuite) TestTheFlowGroupLeadsWithGemmaAndLunaCatchesItBeingDown() {
	candidates, err := s.shipped(LLM).resolveChain(s.ctx, Request{
		Providers: []string{"llm-flow"},
	})
	s.Require().NoError(err)

	s.Equal("gemma/gemma-4-26B-A4B-it", names(candidates)[0],
		"a ruling is pinned to the model that starts writing at once while it is up")
	s.Contains(names(candidates), "openai/gpt-5.6-luna",
		"the rest of the fast tier, Luna with it, still catches Gemma being down")
}

func (s *RoutingSuite) TestTheAccurateBaseGroupLeadsWithTheModelThatNamesTheVoice() {
	candidates, err := s.shipped(STT).resolveChain(s.ctx, Request{
		Providers: []string{"stt-accurate"},
	})
	s.Require().NoError(err)

	s.Equal("muse/muse-voice-transcribe-1.0", names(candidates)[0],
		"a config that goes on to ask for diarize should narrow to the model this group already starts with")
	s.ElementsMatch([]string{
		"muse/muse-voice-transcribe-1.0",
		"elevenlabs/scribe_v2_realtime",
		"cartesia/ink-2",
		"together-nemotron/nvidia/nemotron-3-asr-streaming-0.6b",
	}, names(candidates))
}

func (s *RoutingSuite) TestABaseGroupIsALiveGroupOnly() {
	batch := false
	_, err := s.shipped(STT).resolveChain(s.ctx, Request{
		Providers: []string{"stt-accurate"},
		Realtime:  &batch,
	})

	s.ErrorContains(err, "nothing in the priority list stt-accurate",
		"a recording belongs at en-recorded, where the batch models are")
}

func (s *RoutingSuite) TestABaseGroupNarrowsToWhatCanServeTheTermsAsked() {
	// The whole of what these groups do not carry: naming one asks for its models, not for
	// its options. Of the accurate group only Muse declares diarize, so a config that names
	// the group and then asks to be told who spoke gets that one rather than all four.
	candidates, err := s.shipped(STT).resolveChain(s.ctx, Request{
		Providers: []string{"stt-accurate"},
	})
	s.Require().NoError(err)

	serving, err := serving(candidates, []options.Term{options.Diarize}, "")
	s.Require().NoError(err)

	s.Equal([]string{"muse/muse-voice-transcribe-1.0"}, names(serving))
}

func (s *RoutingSuite) TestAShortcutLeftWithOneModelStillRefusesAsNoProvider() {
	// The caller named a shortcut, not the model, so the refusal says what it said on the
	// commit before this PR rather than naming a model the caller never chose.
	only := []Candidate{{Config: ProviderConfig{Provider: "microsoft", Model: "MAI-Transcribe-2-Streaming"}}}

	_, err := serving(only, []options.Term{options.Keyterms}, "multilingual-low-latency")

	s.EqualError(err, "routing: no provider can express keyterms")
}

func (s *RoutingSuite) TestDefaultConfigPricesEveryProvider() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for modality, section := range config {
		for _, provider := range section.Providers {
			if provider.Provider == "mock" {
				// The load benchmark's stand-ins reach no vendor, so nothing is what they
				// cost and nothing is what they should report.
				continue
			}
			if modality == Detection && provider.Provider == "ultralytics" {
				// Ultralytics' shared API bills nothing; a dedicated endpoint is billed by
				// the hour, which no per-request rate describes.
				continue
			}
			s.NotZerof(provider.Price, "%s/%s has no price, so its cost would report as zero",
				modality, provider.Name())
		}
	}
}

func (s *RoutingSuite) TestDefaultConfigDeclaresAContextWindowForEveryTextModel() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for _, provider := range config[LLM].Providers {
		s.Positivef(provider.ContextWindow, "%s has no context_window, so a long call is never compacted before it overflows",
			provider.Name())
	}
}

func (s *RoutingSuite) TestDefaultConfigMeasuresTextModelsOnIntelligenceAndSpeed() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	measured := 0
	for _, provider := range config[LLM].Providers {
		b := provider.Benchmark
		s.Equalf(Benchmark{IntelligenceIndex: b.IntelligenceIndex, OutputTokensPerSecond: b.OutputTokensPerSecond}, b,
			"%s carries a benchmark that is not a text model's", provider.Name())
		if b.IntelligenceIndex > 0 && b.OutputTokensPerSecond > 0 {
			measured++
		}
	}
	s.Positive(measured)
}

func (s *RoutingSuite) TestDefaultConfigDeclaresADataPolicyForEverySpeechModel() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for _, provider := range config[STT].Providers {
		s.Truef(provider.DataPolicy.Declared(),
			"%s declares no data policy, so it would serve a request that asked about one",
			provider.Name())
		s.Truef(provider.DataPolicy.Valid(),
			"%s declares a data policy nothing can be compared against", provider.Name())
	}
}

func (s *RoutingSuite) TestDefaultConfigDeclaresADataPolicyForEveryVoice() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for _, provider := range config[TTS].Providers {
		s.Truef(provider.DataPolicy.Declared(),
			"%s declares no data policy, so it would serve a request that asked about one",
			provider.Name())
		s.Truef(provider.DataPolicy.Valid(),
			"%s declares a data policy nothing can be compared against", provider.Name())
	}
}

func (s *RoutingSuite) TestConfigRejectsASpeechModelWithoutADataPolicy() {
	config := Config{STT: ModalityConfig{Providers: []ProviderConfig{
		{Provider: "quick", Model: "en", Languages: []string{"en"}},
	}}}

	s.ErrorContains(config.Validate(), "quick/en declares no data_policy")
}

func (s *RoutingSuite) TestConfigRejectsAVoiceWithoutADataPolicy() {
	config := Config{TTS: ModalityConfig{Providers: []ProviderConfig{
		{Provider: "lush", Model: "en", Languages: []string{"en"}},
	}}}

	s.ErrorContains(config.Validate(), "lush/en declares no data_policy")
}

func (s *RoutingSuite) TestConfigRejectsADataPolicyItCannotCompare() {
	config := Config{STT: ModalityConfig{Providers: []ProviderConfig{{
		Provider:   "quick",
		Model:      "en",
		Languages:  []string{"en"},
		DataPolicy: options.DataHandling{TrainsOnData: options.ClaimNo, Retention: "a while"},
	}}}}

	s.ErrorContains(config.Validate(), "which a request cannot be compared against")
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

func (s *RoutingSuite) TestAliasRequiresEveryTermFromAnyConfiguredProvider() {
	config := ModalityConfig{
		Providers: []ProviderConfig{
			{Provider: "alpha", Model: "one", Languages: []string{"en"}, Terms: []options.Term{options.FullDuplex, options.TextInput}},
			{Provider: "beta", Model: "two", Languages: []string{"en"}, Terms: []options.Term{options.FullDuplex}},
			{Provider: "gamma", Model: "three", Languages: []string{"en"}, Terms: []options.Term{options.FullDuplex, options.TextInput}},
		},
		Aliases: map[string]Alias{
			"duplex": {RequireTerms: []options.Term{options.FullDuplex, options.TextInput}},
		},
	}
	router := s.newRouterWithConfig(config, func(spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model}, nil
	})

	candidates, err := router.Resolve(s.ctx, "duplex", nil)
	s.Require().NoError(err)
	s.Equal([]string{"alpha/one", "gamma/three"}, names(candidates))
}

func (s *RoutingSuite) TestConfigRejectsAnAliasNoProviderCanExpressEveryTerm() {
	config := ModalityConfig{
		Providers: []ProviderConfig{
			{Provider: "alpha", Model: "one", Languages: []string{"en"}, Terms: []options.Term{options.FullDuplex}},
			{Provider: "beta", Model: "two", Languages: []string{"en"}, Terms: []options.Term{options.TextInput}},
		},
		Aliases: map[string]Alias{
			"duplex": {RequireTerms: []options.Term{options.FullDuplex, options.TextInput}},
		},
	}

	s.ErrorContains(config.Validate(), "alias duplex matches no provider")
}

func (s *RoutingSuite) TestConfigRejectsAPreferredAliasModelMissingARequiredTerm() {
	config := s.config()
	config.Providers[0].Terms = []options.Term{options.FullDuplex, options.TextInput}
	config.Providers[1].Terms = []options.Term{options.FullDuplex}
	config.Aliases["duplex"] = Alias{
		RequireTerms: []options.Term{options.FullDuplex, options.TextInput},
		Prefer:       "quick/multi",
	}

	s.ErrorContains(config.Validate(), "alias duplex prefers quick/multi, which does not meet its own requirements")
}

func (s *RoutingSuite) TestAliasFailsOverOnlyToCandidatesThatMeetEveryTerm() {
	config := s.config()
	config.Providers[0].Terms = []options.Term{options.FullDuplex}
	config.Providers[1].Terms = []options.Term{options.FullDuplex}
	config.Providers[2].Model = "quality"
	config.Aliases["duplex"] = Alias{RequireTerms: []options.Term{options.FullDuplex}}
	factory := func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
		}
		return &stubProvider{model: spec.Model}, nil
	}

	_, selected, err := s.newRouterWithConfig(config, factory).Select(s.ctx, Request{CustomerID: "acme", Target: "duplex"})
	s.Require().NoError(err)
	s.Equal("quick/multi", selected.Name())

	_, _, err = s.newRouterWithConfig(config, func(spec Spec) (*stubProvider, error) {
		if spec.Model != "en" && spec.Model != "multi" {
			return &stubProvider{model: spec.Model}, nil
		}
		return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
	}).Select(s.ctx, Request{CustomerID: "acme", Target: "duplex"})
	s.Require().Error(err)
	s.ErrorContains(err, "quick/en: upstream is down")
	s.ErrorContains(err, "quick/multi: upstream is down")
	s.NotContains(err.Error(), "lush/quality", "an ineligible model must not receive a downgraded request")
}

func (s *RoutingSuite) TestConfigRejectsAPreferredModelThatIsNotDeclared() {
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{
		Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency,
		Prefer: "quick/does-not-exist",
	}

	s.ErrorContains(config.Validate(),
		"alias en-low-latency prefers quick/does-not-exist, which is not declared")
}

func (s *RoutingSuite) TestConfigRejectsAPreferredModelTheAliasWouldNotPick() {
	// lush/multi is high-quality, so pinning it to a low-latency shortcut would be a pin
	// that never applies rather than one that changes where requests go.
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{
		Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency,
		Prefer: "lush/multi",
	}

	s.ErrorContains(config.Validate(), "does not meet its own requirements")
}

func (s *RoutingSuite) TestNamingCandidatesLeavesOutTheOnesNotNamed() {
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{
		Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency,
		Only: []string{"quick/en"},
	}
	s.Require().NoError(config.Validate())

	var matched []string
	for _, provider := range config.Providers {
		if config.Aliases["en-low-latency"].matches(provider) {
			matched = append(matched, provider.Name())
		}
	}

	s.Equal([]string{"quick/en"}, matched, "quick/multi meets every requirement but was not named")
}

func (s *RoutingSuite) TestConfigRejectsANamedModelThatIsNotDeclared() {
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{
		Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency,
		Only: []string{"quick/en", "quick/does-not-exist"},
	}

	s.ErrorContains(config.Validate(),
		"alias en-low-latency names quick/does-not-exist, which is not declared")
}

func (s *RoutingSuite) TestConfigRejectsANamedModelTheAliasWouldNotPick() {
	// lush/multi is high-quality, so naming it in a low-latency shortcut asks for a model
	// that would be filtered straight back out, which is a config that says one thing and
	// does another.
	config := s.config()
	config.Aliases["en-low-latency"] = Alias{
		Languages: []string{"en"}, RequireRealtime: true, Tier: LowLatency,
		Only: []string{"quick/en", "lush/multi"},
	}

	s.ErrorContains(config.Validate(),
		"alias en-low-latency names lush/multi, which does not meet its own requirements")
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

func (s *RoutingSuite) TestAPriorityListIsTriedInTheOrderItWasWritten() {
	_, config, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush", "quick"},
	})
	s.Require().NoError(err)

	s.Equal("lush/multi", config.Name(),
		"the vendor the caller put first serves, whatever the ranking would have said")
}

func (s *RoutingSuite) TestACandidateTheCallerSawFailIsTriedLast() {
	_, config, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush", "quick"},
		Failed:     []string{"lush/multi"},
	})
	s.Require().NoError(err)

	s.Equal("quick/en", config.Name(),
		"a voice lost on a call came straight back rather than giving way to the next one")
}

func (s *RoutingSuite) TestACandidateTheCallerSawFailStillServesWhenNothingElseCan() {
	_, config, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush"},
		Failed:     []string{"lush/multi"},
	})
	s.Require().NoError(err)

	s.Equal("lush/multi", config.Name(), "a voice lost on a call is tried again when it is all there is")
}

func (s *RoutingSuite) TestAnExcludedCandidateIsNotTried() {
	_, config, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush", "quick"},
		Exclude:    []string{"lush/multi"},
	})
	s.Require().NoError(err)

	s.Equal("quick/en", config.Name())
}

func (s *RoutingSuite) TestARequestThatExcludesEveryCandidateIsNotServed() {
	_, _, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush"},
		Exclude:    []string{"lush/multi"},
	})

	s.Error(err, "a transcription every model refused was handed back to one of them")
}

func (s *RoutingSuite) TestAPriorityListWinsOverATarget() {
	_, config, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Target:     "en-low-latency",
		Providers:  []string{"lush"},
	})
	s.Require().NoError(err)

	s.Equal("lush/multi", config.Name())
}

func (s *RoutingSuite) TestAPriorityListNamingAVendorMeansEveryModelTheyHave() {
	candidates, err := s.newRouter().resolveChain(s.ctx, Request{Providers: []string{"quick"}})
	s.Require().NoError(err)

	s.Equal([]string{"quick/en", "quick/multi"}, names(candidates))
}

func (s *RoutingSuite) TestAPriorityListKeepsAModelWhereItWasFirstNamed() {
	candidates, err := s.newRouter().resolveChain(s.ctx, Request{
		Providers: []string{"lush", "en-low-latency"},
	})
	s.Require().NoError(err)

	s.Equal([]string{"lush/multi", "quick/en", "quick/multi"}, names(candidates),
		"naming a vendor and then a shortcut they are in keeps them where the caller put them")
}

func (s *RoutingSuite) TestAPriorityListFallsOverToTheNextEntry() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "multi" {
			return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
		}
		return &stubProvider{model: spec.Model}, nil
	})

	_, config, err := router.Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush", "quick/en"},
	})
	s.Require().NoError(err)

	s.Equal("quick/en", config.Name())
}

func (s *RoutingSuite) TestAPriorityListReportsWhatItCouldNotResolve() {
	_, err := s.newRouter().resolveChain(s.ctx, Request{Providers: []string{"nobody", "nothing"}})

	s.ErrorContains(err, "nothing in the priority list nobody, nothing")
	s.ErrorContains(err, `unknown target "nobody"`)
	s.ErrorContains(err, `unknown target "nothing"`)
}

func (s *RoutingSuite) TestAPriorityListKeepsABatchModelOffTheLivePath() {
	// "batchy" is the only vendor here that is not realtime, so a socket asking for them
	// by name has to be told rather than handed a model that cannot stream.
	live := true
	_, err := s.newRouter().resolveChain(s.ctx, Request{
		Providers: []string{"batchy"},
		Realtime:  &live,
	})

	s.ErrorContains(err, "nothing in the priority list batchy")
	s.NotContains(err.Error(), "%!w", "no refusal is wrapped when every entry resolved")
	s.NotContains(err.Error(), "can serve this request:", "nothing follows the sentence")
}

func (s *RoutingSuite) TestAPriorityListKeepsALiveModelOffTheRecordedPath() {
	batch := false
	candidates, err := s.newRouter().resolveChain(s.ctx, Request{
		Providers: []string{"quick", "batchy"},
		Realtime:  &batch,
	})
	s.Require().NoError(err)

	s.Equal([]string{"batchy/offline-en"}, names(candidates))
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

func (s *RoutingSuite) TestACustomVoiceBecomesTheIdTheChosenProviderKnowsItBy() {
	var spoke Spec
	router := s.newRouterResolving(prepared{"lush": "lush-88"}, func(spec Spec) (*stubProvider, error) {
		spoke = spec
		return &stubProvider{model: spec.Model}, nil
	})

	_, config, err := router.Select(s.ctx,
		Request{CustomerID: "acme", Target: "en-high-accuracy", Voice: "founder"})
	s.Require().NoError(err)

	s.Equal("lush", config.Provider,
		"the providers that were never given this voice cannot speak in it")
	s.Equal("lush-88", spoke.Voice, "the provider is asked for the voice by its own id")
}

func (s *RoutingSuite) TestAProvidersOwnLibraryVoicePassesStraightThrough() {
	var spoke Spec
	router := s.newRouterResolving(prepared{}, func(spec Spec) (*stubProvider, error) {
		spoke = spec
		return &stubProvider{model: spec.Model}, nil
	})

	_, config, err := router.Select(s.ctx,
		Request{CustomerID: "acme", Target: "en-high-accuracy", Voice: "nova"})
	s.Require().NoError(err)

	s.Equal("quick", config.Provider, "a library voice rules nobody out")
	s.Equal("nova", spoke.Voice)
}

func (s *RoutingSuite) TestTheirOwnVoiceIsRefusedWhereNoVoicesAreResolvedAtAll() {
	_, _, err := s.newRouter().Select(s.ctx,
		Request{CustomerID: "acme", Target: "quick/en", Voice: options.OwnVoicePrefix + "founder"})

	s.ErrorContains(err, "no voices of its own",
		"handing the prefix to a provider would ask its library for a voice called custom:founder")
}

func (s *RoutingSuite) TestEachRouterReadsTheOverwritesForItsOwnModality() {
	var spoke Spec
	router := s.newRouterResolving(prepared{}, func(spec Spec) (*stubProvider, error) {
		spoke = spec
		return &stubProvider{model: spec.Model}, nil
	})

	_, _, err := router.Select(s.ctx, Request{
		CustomerID: "acme",
		Target:     "quick/en",
		STT:        options.STT{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"heard":true}`)}},
		TTS:        options.TTS{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"spoke":true}`)}},
	})
	s.Require().NoError(err)

	s.JSONEq(`{"spoke":true}`, string(spoke.Overwrites),
		"a vendor that both transcribes and speaks must not be handed the other half's settings")
}

// newRouterReadingOverwrites is a router whose providers read a "mode" overwrite through
// Settings, as every real factory that takes overwrites does, and record what was built.
func (s *RoutingSuite) newRouterReadingOverwrites(built *[]string) *Router[*stubProvider] {
	return s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		*built = append(*built, spec.Model)
		var settings struct {
			Mode string `json:"mode"`
		}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}
		return &stubProvider{model: spec.Model}, nil
	})
}

func (s *RoutingSuite) TestAnOverwriteTheCandidateHasNoFieldForIsRefusedRatherThanRoutedAround() {
	var built []string
	router := s.newRouterReadingOverwrites(&built)

	_, _, err := router.Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"quick/en", "lush/multi"},
		STT:        options.STT{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"moode":"fast"}`)}},
	})

	s.Require().ErrorIs(err, ErrOverwrite)
	s.ErrorContains(err, "quick/en", "the caller is told whose block it was")
	s.ErrorContains(err, "moode", "and which field")
	s.Equal([]string{"en"}, built, "lush is never handed a request whose overwrite was dropped")
}

func (s *RoutingSuite) TestAnOverwriteOfTheWrongTypeIsRefusedRatherThanRoutedAround() {
	var built []string
	router := s.newRouterReadingOverwrites(&built)

	_, _, err := router.Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"quick/en", "lush/multi"},
		STT:        options.STT{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"mode":5}`)}},
	})

	s.Require().ErrorIs(err, ErrOverwrite)
	s.ErrorContains(err, "mode")
	s.Equal([]string{"en"}, built)
}

func (s *RoutingSuite) TestABuildFailureThatIsNotAnOverwriteStillFailsOver() {
	// A provider with no key is the provider's problem, not the caller's, so the next
	// candidate serves, as it always has.
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return nil, errors.New("quick: api key is required")
		}
		return &stubProvider{model: spec.Model}, nil
	})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().NoError(err)
	s.Equal("lush/multi", config.Name())
}

// capturingStats swaps the router's recorder for one that keeps every stat in its queue,
// so a test can see what would have reached the shared health counters without writing
// to them.
func (s *RoutingSuite) capturingStats(router *Router[*stubProvider]) *Recorder {
	router.recorder.Close()
	done := make(chan struct{})
	close(done)
	recorder := &Recorder{live: &live.Client{}, queue: make(chan queued, 16), done: done}
	router.recorder = recorder
	return recorder
}

func (s *RoutingSuite) TestASettingTheModelRefusesLeavesItsHealthAlone() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return nil, Refused(errors.New("quick: en takes no similarity"))
		}
		return &stubProvider{model: spec.Model}, nil
	})
	stats := s.capturingStats(router)

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().NoError(err)
	s.Equal("lush/multi", config.Name(), "a setting one model refuses another may take")
	s.Empty(stats.queue, "health is every customer's, so one caller's bad value is not counted against the model")
}

func (s *RoutingSuite) TestAProviderThatFailsToBuildIsStillCountedAgainstItsHealth() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return nil, errors.New("quick: api key is required")
		}
		return &stubProvider{model: spec.Model}, nil
	})
	stats := s.capturingStats(router)

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().NoError(err)
	s.Require().Len(stats.queue, 1)
	recorded := (<-stats.queue).request
	s.Equal("build_failed", recorded.ErrorCode)
	s.Equal("en", recorded.Model)
}

func (s *RoutingSuite) TestARefusalIsWhatTheCallerIsToldWhenNothingServes() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" {
			return nil, Refused(errors.New("quick: stability 1.5 is outside 0 to 1"))
		}
		return nil, errors.New("lush: api key is required")
	})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().ErrorIs(err, ErrRequest)
	s.ErrorContains(err, "quick/en: quick: stability 1.5 is outside 0 to 1")
	s.NotContains(err.Error(), "every candidate", "the caller's mistake is not an outage")
}

func (s *RoutingSuite) TestOneVendorsModelsRefusingInTheSameWordsAreToldOnce() {
	router := s.newRouterWith(func(spec Spec) (*stubProvider, error) {
		if spec.Model == "en" || spec.Model == "multi" {
			return nil, Refused(errors.New("quick: no voice x"))
		}
		return &stubProvider{model: spec.Model}, nil
	})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "quick/multi"}})

	s.Require().ErrorIs(err, ErrRequest)
	s.Equal(1, strings.Count(err.Error(), "quick: no voice x"), "said once, not once per model: %s", err)
}

func (s *RoutingSuite) TestAnOverwriteOneModelCannotReadIsTriedOnTheSameVendorsOtherModels() {
	// OpenAI's realtime and Live models read different structs out of one "openai" block,
	// which is the shape this copies: quick/en has a threshold, quick/multi a backend.
	var built []string
	registry := NewRegistry[*stubProvider]()
	for _, vendor := range []string{"quick", "lush"} {
		registry.Register(vendor, func(spec Spec) (*stubProvider, error) {
			built = append(built, vendor+"/"+spec.Model)
			var err error
			if vendor == "quick" && spec.Model == "en" {
				err = spec.Settings(&struct {
					Threshold float64 `json:"threshold"`
				}{})
			} else {
				err = spec.Settings(&struct {
					Backend string `json:"backend"`
				}{})
			}
			if err != nil {
				return nil, err
			}
			return &stubProvider{model: spec.Model}, nil
		})
	}
	router, err := New(Options[*stubProvider]{Modality: STT, Config: s.config(), Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	_, config, err := router.Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"quick/en", "lush/multi", "quick/multi"},
		STT:        options.STT{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"backend":"b2"}`)}},
	})

	s.Require().NoError(err)
	s.Equal("quick/multi", config.Name())
	s.Equal([]string{"quick/en", "quick/multi"}, built,
		"lush is handed its own block, never quick's, so it is skipped rather than served")
}

func (s *RoutingSuite) TestEachElevenLabsModelDeclaresTheVoiceSettingsItsSocketSends() {
	// The text-to-speech socket sends stability and similarity_boost on every context
	// (tts/elevenlabs/elevenlabs.go openContext); the dialogue socket sends a stability and
	// takes no speed or similarity (dialogue.go NewDialogue); the batch endpoint sends
	// stability and similarity. It declares no speed: live on 2026-10-16 eleven_v3 answered
	// 200 to a speed of 0.7, 1.0, 1.2 and 1.5 alike, in times that did not follow it, where
	// Flash refused 1.5 with a 400.
	want := map[string][]options.Term{
		"eleven_flash_v2_5":        {options.Speed, options.Stability, options.Similarity},
		"eleven_multilingual_v2":   {options.Speed, options.Stability, options.Similarity},
		"eleven_v4_turbo":          {options.Stability},
		"eleven_v3_conversational": {options.Stability},
		"eleven_v4":                {options.Stability},
		"eleven_v3":                {options.Stability, options.Similarity, options.Format},
	}
	config, err := DefaultConfig()
	s.Require().NoError(err)

	got := map[string][]options.Term{}
	for _, provider := range config[TTS].Providers {
		if provider.Provider == "elevenlabs" {
			got[provider.Model] = provider.Terms
		}
	}
	s.Equal(want, got)
}

func (s *RoutingSuite) TestAVoiceNoProviderWasGivenLeavesNobodyToSayIt() {
	router := s.newRouterResolving(prepared{"nobody": "unused"}, func(spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model}, nil
	})

	_, _, err := router.Select(s.ctx,
		Request{CustomerID: "acme", Target: "en-high-accuracy", Voice: "founder"})
	s.ErrorContains(err, "every candidate")
}

// vetting is a resolver where lush looks voices up and has none, and the providers in
// unvouched may not be handed a voice once they are asked about. It keeps the providers it
// was asked about.
type vetting struct {
	unvouched []string
	asked     *[]string
}

func (v vetting) ResolveVoice(_ context.Context, _, provider, voice string) (string, error) {
	if provider == "lush" {
		return "", Refused(fmt.Errorf("lush has no voice %q", voice))
	}
	return voice, nil
}

func (v vetting) Unvouched(_ context.Context, _, _ string, providers []string) []string {
	*v.asked = providers
	var named []string
	for _, provider := range v.unvouched {
		if slices.Contains(providers, provider) {
			named = append(named, provider)
		}
	}
	return named
}

func (s *RoutingSuite) TestAProviderNotVouchedForIsNeverHandedTheVoiceWhateverItsRank() {
	var built, asked []string
	router := s.newRouterResolving(vetting{unvouched: []string{"quick"}, asked: &asked},
		func(spec Spec) (*stubProvider, error) {
			built = append(built, spec.Model)
			return &stubProvider{model: spec.Model}, nil
		})

	_, _, err := router.Select(s.ctx, Request{
		CustomerID: "acme", Providers: []string{"quick/en", "lush/multi", "quick/multi"}, Voice: "nova",
	})

	s.Require().ErrorIs(err, ErrRequest)
	s.EqualError(err, `lush/multi: lush has no voice "nova"`, "the refusal, not every candidate failing")
	s.Empty(built)
	s.Equal([]string{"quick", "lush"}, asked, "each provider is asked about once")
}

func (s *RoutingSuite) TestAProviderWhoseOverwriteNamesAVoiceIsNotAskedAboutTheRequestsVoice() {
	var asked []string
	router := s.newRouterResolving(vetting{unvouched: []string{"quick"}, asked: &asked},
		func(spec Spec) (*stubProvider, error) { return &stubProvider{model: spec.Model}, nil })

	_, config, err := router.Select(s.ctx, Request{
		CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}, Voice: "nova",
		TTS: options.TTS{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"voice_id":"own"}`)}},
	})

	s.Require().NoError(err, "quick speaks the voice its overwrite names")
	s.Equal("quick/en", config.Name())
	s.Equal([]string{"lush"}, asked)
}

func (s *RoutingSuite) TestARequestWithNoVoiceIsNotVetted() {
	var asked []string
	router := s.newRouterResolving(vetting{unvouched: []string{"quick"}, asked: &asked},
		func(spec Spec) (*stubProvider, error) { return &stubProvider{model: spec.Model}, nil })

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().NoError(err, "a provider's default voice is its own")
	s.Equal("quick/en", config.Name())
	s.Nil(asked, "nothing to ask about")
}

func (s *RoutingSuite) TestProvidersListsEveryConfiguredModelInOrder() {
	providers := s.newRouter().Providers(s.ctx)

	s.Equal([]string{"quick/en", "quick/multi", "lush/multi", "batchy/offline-en"}, names(providers))
	s.True(providers[0].Health.Available, "an unmeasured provider is available")
}

func (s *RoutingSuite) TestDemoteMovesUnavailableProvidersToTheBack() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "down", Model: "m"}, Health: live.Health{Requests: 10, Errors: 10, Available: false}},
		{Config: ProviderConfig{Provider: "up", Model: "m"}, Health: live.Health{Requests: 10, Available: true}},
	}

	demote(candidates)

	s.Equal([]string{"up/m", "down/m"}, names(candidates))
}

func (s *RoutingSuite) TestDemoteLeavesASlowerProviderWhereTheCallerPutIt() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "chosen", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 500, Available: true}},
		{Config: ProviderConfig{Provider: "faster", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 20, Available: true}},
	}

	demote(candidates)

	s.Equal([]string{"chosen/m", "faster/m"}, names(candidates),
		"a priority list means the order is the point; only being down costs a place")
}

func (s *RoutingSuite) TestADataPolicyNarrowsToTheProvidersThatMeetIt() {
	no := false
	candidates, err := permitted(s.newRouter().Providers(s.ctx), options.DataPolicy{AllowTraining: &no})
	s.Require().NoError(err)

	s.Equal([]string{"quick/en", "quick/multi"}, names(candidates),
		"the vendor that trains and the one that has said nothing are both out")
}

func (s *RoutingSuite) TestARetentionCeilingKeepsProvidersWithinIt() {
	candidates, err := permitted(s.newRouter().Providers(s.ctx), options.DataPolicy{Retention: "30d"})
	s.Require().NoError(err)

	s.Equal([]string{"quick/en", "quick/multi"}, names(candidates))
}

func (s *RoutingSuite) TestAskingForNoRetentionExcludesAProviderThatKeepsAnything() {
	candidates, err := permitted(s.newRouter().Providers(s.ctx), options.DataPolicy{
		Retention: options.RetentionNone,
	})
	s.Require().NoError(err)

	s.Equal([]string{"quick/en"}, names(candidates))
}

func (s *RoutingSuite) TestADataPolicyNothingMeetsNamesWhatWentUnmet() {
	no := false
	_, err := permitted(
		[]Candidate{{Config: ProviderConfig{Provider: "lush", Model: "multi", DataPolicy: trains}}},
		options.DataPolicy{AllowTraining: &no, Retention: "1h"},
	)

	s.ErrorContains(err, "no provider meets your data policy")
	s.ErrorContains(err, "not training on what it is sent")
	s.ErrorContains(err, "a retention of 1h or less")
}

func (s *RoutingSuite) TestADataPolicyReportsOnlyTheHalfNothingOffered() {
	no := false
	_, err := permitted(
		[]Candidate{{Config: ProviderConfig{Provider: "quick", Model: "multi", DataPolicy: keeps}}},
		options.DataPolicy{AllowTraining: &no, Retention: options.RetentionNone},
	)

	s.ErrorContains(err, "keeping nothing at all")
	s.NotContains(err.Error(), "not training",
		"this provider does not train, so saying nothing offers that would be wrong")
}

func (s *RoutingSuite) TestSelectRefusesADataPolicyNoProviderMeets() {
	no := false
	_, _, err := s.newRouter().Select(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"lush"},
		DataPolicy: options.DataPolicy{AllowTraining: &no},
	})

	s.ErrorContains(err, "no provider meets your data policy")
}

func (s *RoutingSuite) TestADataPolicyAskingNothingRulesNothingOut() {
	candidates, err := permitted(s.newRouter().Providers(s.ctx), options.DataPolicy{})
	s.Require().NoError(err)

	s.Len(candidates, 4)
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

func (s *RoutingSuite) TestPreferredModelGoesAheadOfABetterMeasuredOne() {
	// The point of a pin: the deployment has chosen, and a rival's numbers do not reopen
	// the question.
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "faster", Model: "m"}, Health: live.Health{Requests: 100, LatencyMsAvg: 40, Available: true}},
		{Config: ProviderConfig{Provider: "chosen", Model: "m"}, Health: live.Health{Requests: 100, LatencyMsAvg: 300, Available: true}},
	}

	rank(candidates)
	prefer(candidates, "chosen/m")

	s.Equal([]string{"chosen/m", "faster/m"}, names(candidates))
}

func (s *RoutingSuite) TestAnUnavailablePreferredModelLetsTheRankingDecide() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "standby", Model: "m"}, Health: live.Health{Requests: 100, LatencyMsAvg: 300, Available: true}},
		{Config: ProviderConfig{Provider: "chosen", Model: "m"}, Health: live.Health{Requests: 100, Errors: 100, Available: false}},
	}

	rank(candidates)
	prefer(candidates, "chosen/m")

	s.Equal([]string{"standby/m", "chosen/m"}, names(candidates),
		"a pin says which model is wanted, not that the request should fail with it")
}

func (s *RoutingSuite) TestPreferringKeepsTheRestOfTheRankingIntact() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "fast", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 50, Available: true}},
		{Config: ProviderConfig{Provider: "middling", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 100, Available: true}},
		{Config: ProviderConfig{Provider: "chosen", Model: "m"}, Health: live.Health{Requests: 10, LatencyMsAvg: 500, Available: true}},
	}

	rank(candidates)
	prefer(candidates, "chosen/m")

	s.Equal([]string{"chosen/m", "fast/m", "middling/m"}, names(candidates))
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

func (s *RoutingSuite) TestCostBillsCacheWritesAtTheWriteRate() {
	// Opus 5.5: $4 per million fresh, $0.20 read from cache, $5 written to it. Of a
	// million-token prompt, 900k were read and 90k written, leaving 10k fresh.
	price := Price{PerMillionInputTokens: 4, PerMillionCachedInputTokens: 0.2, PerMillionCacheWriteInputTokens: 5}

	s.EqualValues(40_000+180_000+450_000, price.CostMicros(Usage{InputTokens: 1_000_000, CachedInputTokens: 900_000, CacheWriteInputTokens: 90_000}))
}

func (s *RoutingSuite) TestCostBillsCacheWritesAsFreshInputWithoutAWriteRate() {
	// A provider that reports writes but does not charge extra for them.
	price := Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.1}

	s.EqualValues(550_000, price.CostMicros(Usage{InputTokens: 1_000_000, CachedInputTokens: 500_000, CacheWriteInputTokens: 400_000}))
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

func (s *RoutingSuite) TestCostPricesImagesByThePicture() {
	// Qwen Image 3 on FAL: $0.04 a picture, whatever its size.
	price := Price{PerImage: 0.04}

	s.EqualValues(40_000, price.CostMicros(Usage{Images: 1, Pixels: 1024 * 1024}))
	s.EqualValues(120_000, price.CostMicros(Usage{Images: 3, Pixels: 3 * 2048 * 2048}),
		"three pictures cost three times one, however large they are")
}

func (s *RoutingSuite) TestCostPricesImagesByThePixelsThatCameBack() {
	// $0.02 a megapixel, a million pixels to the megapixel.
	price := Price{PerMegapixel: 0.02}

	s.EqualValues(20_000, price.CostMicros(Usage{Images: 1, Pixels: 1_000_000}))
	s.EqualValues(83_886, price.CostMicros(Usage{Images: 1, Pixels: 2048 * 2048}),
		"a larger picture costs more even though it is still one picture")
	s.Zero(price.CostMicros(Usage{Images: 0, Pixels: 0}), "nothing drawn is nothing billed")
}

func (s *RoutingSuite) TestTierDefaultsToLowLatency() {
	s.Equal(LowLatency, ProviderConfig{}.tier())
	s.Equal(HighQuality, ProviderConfig{Tier: HighQuality}.tier())
}

func (s *RoutingSuite) TestSeeingDropsModelsThatCannotTakeAnImage() {
	candidates := []Candidate{
		{Config: ProviderConfig{Provider: "quick", Model: "text", Languages: []string{"en"}}},
		{Config: ProviderConfig{Provider: "lush", Model: "vision", Languages: []string{"en"}, InputModalities: []string{"image"}}},
	}

	kept, err := seeing(candidates, []string{"image"})
	s.Require().NoError(err)
	s.Equal([]string{"lush/vision"}, names(kept))

	_, err = seeing(candidates[:1], []string{"image"})
	s.ErrorContains(err, "no provider accepts image input")
}

func (s *RoutingSuite) TestDefaultConfigDeclaresVisionOnOpenAIGeminiAnthropicMetaXAIMoondreamAndNemotronVL() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for _, provider := range config[LLM].Providers {
		if provider.Provider == "openai" || provider.Provider == "gemini" || provider.Provider == "anthropic" || provider.Provider == "meta" || provider.Provider == "xai" || provider.Provider == "moondream" || provider.Model == "nvidia/NVIDIA-Nemotron-3.5-Super-VL" {
			s.Containsf(provider.InputModalities, "image", "%s should accept images", provider.Name())
			continue
		}
		s.Emptyf(provider.InputModalities, "%s is text-only", provider.Name())
	}
}

// failingUse is a Walk's use that fails for the models in failing, with their error, and
// keeps every candidate it was handed, in order.
func failingUse(asked *[]string, failing map[string]error) func(*stubProvider, ProviderConfig) error {
	return func(provider *stubProvider, config ProviderConfig) error {
		*asked = append(*asked, config.Name())
		provider.Close()
		return failing[config.Name()]
	}
}

func (s *RoutingSuite) TestAWalkHandsTheNextCandidateOnWhenTheFirstFailsInUse() {
	var logged bytes.Buffer
	router := s.newRouter()
	router.logger = slog.New(slog.NewTextHandler(&logged, nil))
	var asked []string

	config, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}},
		failingUse(&asked, map[string]error{"quick/en": errors.New("quick: 500")}))

	s.Require().NoError(err)
	s.Equal("lush/multi", config.Name())
	s.Equal([]string{"quick/en", "lush/multi"}, asked)
	s.NotContains(logged.String(), "failed to start", "quick/en started; its vendor failed it after")
}

func (s *RoutingSuite) TestAWalkWhoseOnlyCandidateFailsInUseReturnsItsErrorAndConfigAsTheyAre() {
	router := s.newRouter()
	var asked []string

	config, err := router.Walk(s.ctx, Request{CustomerID: "acme", Target: "quick/en"},
		failingUse(&asked, map[string]error{"quick/en": errors.New("quick: 500")}))

	s.EqualError(err, "quick: 500", "as a caller of Select reported it")
	s.Equal("quick/en", config.Name())
}

func (s *RoutingSuite) TestAWalkEveryCandidateFailsInUseNamesEachInTheOrderMet() {
	router := s.newRouter()
	var asked []string

	config, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}},
		failingUse(&asked, map[string]error{"quick/en": errors.New("quick: 500"), "lush/multi": errors.New("lush: 429")}))

	s.EqualError(err, "quick/en: quick: 500\nlush/multi: lush: 429")
	s.NotErrorIs(err, ErrRequest)
	s.Equal(ProviderConfig{}, config, "nobody served it")
}

func (s *RoutingSuite) TestAWalkWithARefusalAmongItsFailuresIsTheCallersToChangeAndSaysBoth() {
	router := s.newRouter()
	var asked []string

	_, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}},
		failingUse(&asked, map[string]error{"quick/en": Refused(errors.New("quick: too long")), "lush/multi": errors.New("lush: 429")}))

	s.Require().ErrorIs(err, ErrRequest)
	s.EqualError(err, "quick/en: quick: too long\nlush/multi: lush: 429")
}

func (s *RoutingSuite) TestAWalkWhoseCallerHasGoneAsksNoFurtherCandidate() {
	router := s.newRouter()
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	var asked []string

	_, err := router.Walk(ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}},
		func(provider *stubProvider, config ProviderConfig) error {
			asked = append(asked, config.Name())
			cancel()
			return context.Canceled
		})

	s.Require().ErrorIs(err, context.Canceled)
	s.Equal([]string{"quick/en"}, asked)
}

func (s *RoutingSuite) TestASelectWhoseContextHasEndedStillStartsTheFirstCandidateAsBefore() {
	ctx, cancel := context.WithCancel(s.ctx)
	cancel()

	_, config, err := s.newRouter().Select(ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.Require().NoError(err, "a stub starts whatever the context; Select leaves that to the provider")
	s.Equal("quick/en", config.Name())
}

func (s *RoutingSuite) TestAWalkNeverServesAnOverwriteOnAnotherVendorAfterItsOwnModelFails() {
	// quick/en has a threshold and quick/multi a backend, as in
	// TestAnOverwriteOneModelCannotReadIsTriedOnTheSameVendorsOtherModels.
	var built []string
	registry := NewRegistry[*stubProvider]()
	for _, vendor := range []string{"quick", "lush"} {
		registry.Register(vendor, func(spec Spec) (*stubProvider, error) {
			built = append(built, vendor+"/"+spec.Model)
			var err error
			if vendor == "quick" && spec.Model == "en" {
				err = spec.Settings(&struct {
					Threshold float64 `json:"threshold"`
				}{})
			} else {
				err = spec.Settings(&struct {
					Backend string `json:"backend"`
				}{})
			}
			if err != nil {
				return nil, err
			}
			return &stubProvider{model: spec.Model}, nil
		})
	}
	router, err := New(Options[*stubProvider]{Modality: STT, Config: s.config(), Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	var asked []string

	config, err := router.Walk(s.ctx, Request{
		CustomerID: "acme",
		Providers:  []string{"quick/en", "lush/multi", "quick/multi"},
		STT:        options.STT{Overwrites: map[string]json.RawMessage{"quick": json.RawMessage(`{"backend":"b2"}`)}},
	}, failingUse(&asked, map[string]error{"quick/multi": errors.New("quick: 500")}))

	s.Require().ErrorIs(err, ErrOverwrite)
	s.ErrorContains(err, "quick/multi: quick: 500")
	s.Equal(ProviderConfig{}, config, "nobody served it")
	s.Equal([]string{"quick/multi"}, asked)
	s.Equal([]string{"quick/en", "quick/multi"}, built, "lush is handed its own block, so it would drop quick's")
}
