package sttrouter

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
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
