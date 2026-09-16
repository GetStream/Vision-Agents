package sttrouter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/gemini"
)

type STTRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestSTTRouterSuite(t *testing.T) {
	suite.Run(t, new(STTRouterSuite))
}

func (s *STTRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// newRouter routes over the built-in speech-to-text config, which is what a deployment
// gets when it sets no config file.
func (s *STTRouterSuite) newRouter() *Router {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := New(Options{Config: config[routing.STT], Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *STTRouterSuite) TestRouterServesTheSpeechToTextModality() {
	s.Equal(routing.STT, s.newRouter().Modality())
}

func (s *STTRouterSuite) TestEveryShortcutResolvesToAProvider() {
	router := s.newRouter()

	for alias := range router.Config().Aliases {
		candidates, err := router.Resolve(s.ctx, alias, nil)
		s.Require().NoErrorf(err, "alias %s", alias)
		s.NotEmptyf(candidates, "alias %s", alias)
	}
}

func (s *STTRouterSuite) TestLowLatencyShortcutPrefersTheEnglishFluxModel() {
	candidates, err := s.newRouter().Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)

	s.Require().NotEmpty(candidates)
	s.Equal("deepgram/flux-general-en", candidates[0].Config.Name())
}

func (s *STTRouterSuite) TestAnEnglishCallGoesToTheFourTrustedModelsAndNowhereElse() {
	// Every live model is realtime, low-latency and speaks English, so this shortcut is
	// the ones the deployment trusts an English call to rather than everything that could
	// serve one. The rest stay reachable by name and through the other shortcuts.
	candidates, err := s.newRouter().Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)

	var names []string
	for _, candidate := range candidates {
		names = append(names, candidate.Config.Name())
	}
	s.ElementsMatch([]string{
		"deepgram/flux-general-en",
		"deepgram/flux-general-multi",
		"grok/grok-stt",
		"muse/muse-voice-transcribe-1.0",
	}, names)
}

func (s *STTRouterSuite) TestTheModelsLeftOutOfTheEnglishShortcutAreStillReachable() {
	router := s.newRouter()

	for _, name := range []string{
		"gemini/gemini-3.5-transcribe-live",
		"parakeet/parakeet-tdt-0.6b-v3",
		"cartesia/ink-2",
		"inworld/inworld-stt-1",
		"elevenlabs/scribe_v2_realtime",
	} {
		candidates, err := router.Resolve(s.ctx, name, nil)
		s.Require().NoErrorf(err, "target %s", name)
		s.Require().Lenf(candidates, 1, "target %s", name)
		s.Equal(name, candidates[0].Config.Name())
	}
}

func (s *STTRouterSuite) TestAnEnglishCallCanStillAskForTheVoiceToBeNamed() {
	// Two of the four can name it, and asking narrows to them rather than being served
	// without the label by one of the two that cannot.
	diarize := true
	asked := options.STT{Diarize: &diarize}.Terms()
	candidates, err := s.newRouter().Resolve(s.ctx, "en-low-latency", nil)
	s.Require().NoError(err)

	var names []string
	for _, candidate := range candidates {
		if candidate.Config.Supports(asked) {
			names = append(names, candidate.Config.Name())
		}
	}
	s.ElementsMatch([]string{"grok/grok-stt", "muse/muse-voice-transcribe-1.0"}, names)
}

func (s *STTRouterSuite) TestGermanNarrowsToTheModelThatSpeaksIt() {
	candidates, err := s.newRouter().Resolve(s.ctx, "multilingual-low-latency", []string{"de"})
	s.Require().NoError(err)

	for _, candidate := range candidates {
		s.NotEqual("flux-general-en", candidate.Config.Model, "the English model cannot serve German")
	}
}

func (s *STTRouterSuite) TestRegistryKnowsEveryConfiguredProvider() {
	router := s.newRouter()
	registry := DefaultRegistry()

	for _, provider := range router.Config().Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is configured but has no factory, so it can never serve a request", provider.Provider)
	}
}

func (s *STTRouterSuite) TestRegistryPassesLanguageHintsOnlyToTheMultilingualModel() {
	registry := DefaultRegistry()
	s.T().Setenv("DEEPGRAM_API_KEY", "test-key")

	// The English model rejects language hints, so passing them through would break it.
	_, err := registry.Build("deepgram", routing.Spec{Model: "flux-general-en", LanguageHints: []string{"es"}})
	s.NoError(err)

	_, err = registry.Build("deepgram", routing.Spec{Model: "flux-general-multi", LanguageHints: []string{"es"}})
	s.NoError(err)
}

func (s *STTRouterSuite) TestRegistryPassesKeytermsAndHintsToGemini() {
	registry := DefaultRegistry()
	s.T().Setenv("GOOGLE_API_KEY", "test-key")

	// Gemini detects the language itself and has no vocabulary field, so both are only
	// ever a request. Building must still accept them rather than refuse the session.
	_, err := registry.Build("gemini", routing.Spec{
		Model:         "gemini-3.5-transcribe-live",
		Keyterms:      []string{"Vision Agents"},
		LanguageHints: []string{"es"},
	})
	s.NoError(err)
}

func (s *STTRouterSuite) TestRegistryNarrowsAListOfHintsForGrok() {
	registry := DefaultRegistry()
	s.T().Setenv("XAI_API_KEY", "test-key")

	// xAI takes one language code. A multilingual request arrives with several, and
	// refusing to build over that would lose the session rather than the formatting.
	built, err := registry.Build("grok", routing.Spec{
		Model:         "grok-stt",
		LanguageHints: []string{"es", "fr"},
	})
	s.Require().NoError(err)
	s.Equal("grok-stt", built.Model())
}

func (s *STTRouterSuite) TestRegistryBuildsTheTogetherHostedParakeet() {
	registry := DefaultRegistry()
	s.T().Setenv("TOGETHER_API_KEY", "test-key")

	built, err := registry.Build("together-parakeet", routing.Spec{
		Model: "nvidia/parakeet-tdt-0.6b-v3-realtime",
	})
	s.Require().NoError(err)
	s.Equal("nvidia/parakeet-tdt-0.6b-v3-realtime", built.Model())
	s.Equal("together-parakeet", built.Provider(),
		"the self-hosted deployment of the same weights is a different provider")
}

// TestRegistryBuildsTheThreeRealtimeModelsThatNameTheirVendorTwice covers the models whose
// wire name and routing name differ. Together's are namespaced by the lab that trained
// them and Inworld's by Inworld itself, and a provider that reported the wire name would
// not match the row in router.yaml that chose it.
func (s *STTRouterSuite) TestRegistryBuildsTheThreeRealtimeModelsThatNameTheirVendorTwice() {
	registry := DefaultRegistry()
	s.T().Setenv("CARTESIA_API_KEY", "test-key")
	s.T().Setenv("INWORLD_API_KEY", "test-key")
	s.T().Setenv("ELEVENLABS_API_KEY", "test-key")

	for provider, model := range map[string]string{
		"cartesia":   "ink-2",
		"inworld":    "inworld-stt-1",
		"elevenlabs": "scribe_v2_realtime",
	} {
		built, err := registry.Build(provider, routing.Spec{Model: model})
		s.Require().NoError(err)
		s.Equal(provider, built.Provider())
		s.Equal(model, built.Model(),
			"the model is reported as router.yaml names it, whatever the vendor calls it on the wire")
	}
}

// TestMuseTakesTheTurnBoundaryAsAModeAndRefusesOneItHasNot covers the vendor whose only
// vocabulary for the boundary is a mode. A caller who wrote it out meant it, so it wins
// over the mode the shared diarize term worked out.
func (s *STTRouterSuite) TestMuseTakesTheTurnBoundaryAsAModeAndRefusesOneItHasNot() {
	registry := DefaultRegistry()
	s.T().Setenv("META_API_KEY", "test-key")
	diarize := true

	built, err := registry.Build("muse", routing.Spec{
		Model:      "muse-voice-transcribe-1.0",
		STT:        options.STT{Diarize: &diarize},
		Overwrites: json.RawMessage(`{"mode":"PUSH_TO_TALK"}`),
	})
	s.Require().NoError(err)
	s.Equal("muse-voice-transcribe-1.0", built.Model())

	_, err = registry.Build("muse", routing.Spec{
		Model:      "muse-voice-transcribe-1.0",
		Overwrites: json.RawMessage(`{"mode":"WHENEVER"}`),
	})
	s.ErrorContains(err, "WHENEVER",
		"a mode this model does not have has to be reported rather than sent")
}

// TestNemotronsTurnGraceIsTheRoutersOwnWait is the overwrite that is not the vendor's knob
// at all: nothing on that protocol says where a turn ended, so the wait is ours.
func (s *STTRouterSuite) TestNemotronsTurnGraceIsTheRoutersOwnWait() {
	var settings togetherNemotronSettings
	s.Require().NoError(routing.Spec{
		Overwrites: json.RawMessage(`{"turn_grace_ms":400}`),
	}.Settings(&settings))
	s.Equal(400, settings.TurnGraceMs)

	registry := DefaultRegistry()
	s.T().Setenv("TOGETHER_API_KEY", "test-key")

	_, err := registry.Build("together-nemotron", routing.Spec{
		Model:      "nvidia/nemotron-3-asr-streaming-0.6b",
		Overwrites: json.RawMessage(`{"silence_ms":400}`),
	})
	s.ErrorContains(err, "silence_ms",
		"this provider has no server-side endpointer, so it cannot be asked for one")
}

func (s *STTRouterSuite) TestRegistryReadsInk2sTurnThresholdsFromOverwrites() {
	var settings cartesiaSettings
	spec := routing.Spec{
		Model:      "ink-2",
		Overwrites: json.RawMessage(`{"turn_end_threshold":0.7,"turn_end_timeout_ms":600}`),
	}

	s.Require().NoError(spec.Settings(&settings))

	s.InDelta(0.7, settings.TurnEndThreshold, 0.001)
	s.Equal(600, settings.TurnEndTimeoutMs)
	s.Zero(settings.TurnStartThreshold, "what was not named keeps Ink 2's own default")
}

// TestRegistryTellsInworldsVadThresholdOffFromAnAbsentOne is why that one field is a
// pointer. Zero turns the server's turn detection off, which on a live call means nothing
// settles until the call ends, so it cannot also be how an unset field reads.
func (s *STTRouterSuite) TestRegistryTellsInworldsVadThresholdOffFromAnAbsentOne() {
	var off inworldSettings
	s.Require().NoError(routing.Spec{
		Overwrites: json.RawMessage(`{"vad_threshold":0}`),
	}.Settings(&off))
	s.Require().NotNil(off.VadThreshold)
	s.Zero(*off.VadThreshold)

	var unset inworldSettings
	s.Require().NoError(routing.Spec{Overwrites: json.RawMessage(`{}`)}.Settings(&unset))
	s.Nil(unset.VadThreshold, "a request that said nothing leaves the detector on")
}

func (s *STTRouterSuite) TestRegistryReadsScribesDetectorFromOverwrites() {
	var settings elevenlabsSettings
	spec := routing.Spec{
		Model:      "scribe_v2_realtime",
		Overwrites: json.RawMessage(`{"vad_silence_threshold_secs":0.4,"min_silence_duration_ms":200}`),
	}

	s.Require().NoError(spec.Settings(&settings))

	s.InDelta(0.4, settings.VadSilenceThresholdSecs, 0.001)
	s.Equal(200, settings.MinSilenceDurationMs)
	s.Zero(settings.MinSpeechDurationMs, "what was not named keeps Scribe's own default")
}

func (s *STTRouterSuite) TestRegistryReadsTheFluxTurnThresholdsFromOverwrites() {
	var settings deepgramSettings
	spec := routing.Spec{
		Model:      "flux-general-en",
		Overwrites: json.RawMessage(`{"eot_threshold":0.6,"eot_timeout_ms":800}`),
	}

	s.Require().NoError(spec.Settings(&settings))

	s.InDelta(0.6, settings.EotThreshold, 0.001)
	s.Equal(800, settings.EotTimeoutMs)
	s.Zero(settings.EagerEotThreshold, "what was not named keeps Flux's own default")
}

func (s *STTRouterSuite) TestRegistryRefusesAnOverwriteTheProviderHasNoFieldFor() {
	registry := DefaultRegistry()
	s.T().Setenv("DEEPGRAM_API_KEY", "test-key")

	_, err := registry.Build("deepgram", routing.Spec{
		Model:      "flux-general-en",
		Overwrites: json.RawMessage(`{"eot_treshold":0.6}`),
	})

	s.ErrorContains(err, "eot_treshold",
		"a misspelt setting has to be reported, since the alternative is silently not sending it")
}

func (s *STTRouterSuite) TestRegistryBuildsWithoutOverwrites() {
	registry := DefaultRegistry()
	s.T().Setenv("DEEPGRAM_API_KEY", "test-key")

	built, err := registry.Build("deepgram", routing.Spec{Model: "flux-general-en"})
	s.Require().NoError(err)

	s.Equal("flux-general-en", built.Model())
}

func (s *STTRouterSuite) TestDeepgramOptsOutOfTrainingOnlyWhenAsked() {
	no, yes := false, true

	s.True(trainingRefused(routing.Spec{
		STT: options.STT{DataPolicy: options.DataPolicy{AllowTraining: &no}},
	}))
	s.False(trainingRefused(routing.Spec{
		STT: options.STT{DataPolicy: options.DataPolicy{AllowTraining: &yes}},
	}))
	s.False(trainingRefused(routing.Spec{}), "a request that asked nothing is not a request to opt out")
}

func (s *STTRouterSuite) TestGeminiIsAskedForTheModeTheRequestNamed() {
	s.Equal(gemini.ModeVerbatim, transcriptionMode(options.ModeVerbatim))
	s.Equal(gemini.ModeSmart, transcriptionMode(options.ModeSmart))
	s.Empty(transcriptionMode(""), "saying nothing leaves the server its own default")
}

func (s *STTRouterSuite) TestStartRequiresACustomer() {
	_, err := s.newRouter().Start(s.ctx, Request{Target: "en-low-latency"})
	s.ErrorContains(err, "customer id is required")
}

func (s *STTRouterSuite) TestKeytermsReachTheProviderTheRouterBuilt() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	var built routing.Spec
	registry := NewRegistry()
	registry.Register("deepgram", func(spec routing.Spec) (stt.STT, error) {
		built = spec
		return &quietSTT{emitter: stt.NewEmitter(1)}, nil
	})

	router, err := New(Options{Config: config[routing.STT], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, Request{
		CustomerID: "acme",
		Target:     "deepgram/flux-general-en",
		Keyterms:   []string{"Vision Agents", "Stream"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal([]string{"Vision Agents", "Stream"}, built.Keyterms)
}

// quietSTT transcribes nothing, so a test can watch what the router built without
// reaching a provider.
type quietSTT struct {
	emitter *stt.Emitter
}

func (q *quietSTT) Start(context.Context) error                     { return nil }
func (q *quietSTT) ProcessAudio(stt.PcmData, stt.Participant) error { return nil }
func (q *quietSTT) Events() <-chan stt.Event                        { return q.emitter.Events() }
func (q *quietSTT) Close() error                                    { q.emitter.Close(); return nil }
func (q *quietSTT) Provider() string                                { return "deepgram" }
func (q *quietSTT) Model() string                                   { return "flux-general-en" }
