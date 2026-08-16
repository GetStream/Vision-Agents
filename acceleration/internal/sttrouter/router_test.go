package sttrouter

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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

func (s *STTRouterSuite) TestStartRequiresACustomer() {
	_, err := s.newRouter().Start(s.ctx, Request{Target: "en-low-latency"})
	s.ErrorContains(err, "customer id is required")
}
