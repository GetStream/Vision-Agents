package llmrouter

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/cerebras"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/deepseek"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/xai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// loaded is a router.yaml as a router's start reads it.
func (s *LLMRouterSuite) loaded(yaml string) (routing.Config, error) {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte(yaml), 0o600))
	return routing.LoadConfig(path)
}

// regionalConfig is router.yaml declaring Claude Haiku in the us, and the instance eu writes
// after it.
func (s *LLMRouterSuite) regionalConfig(eu string) routing.ModalityConfig {
	config, err := s.loaded(`
llm:
  providers:
    - provider: anthropic
      model: claude-haiku-5-5
      languages: [en]
` + eu)
	s.Require().NoError(err)
	return config[routing.LLM]
}

func (s *LLMRouterSuite) TestTheVendorsWhoseClientsTakeAHostAndAKeyHonorBoth() {
	registry := DefaultRegistry()
	for _, vendor := range []string{openai.ProviderName, anthropic.ProviderName, gemini.ProviderName, xai.ProviderName, deepseek.ProviderName} {
		s.True(registry.Honors(vendor, routing.OverrideBaseURL), vendor)
		s.True(registry.Honors(vendor, routing.OverrideAPIKey), vendor)
		s.False(registry.Honors(vendor, routing.OverrideRegion), "%s's default host is one region's", vendor)
	}
	s.False(DefaultRegistry().Honors(cerebras.ProviderName, routing.OverrideBaseURL))
}

func (s *LLMRouterSuite) TestAnEUInstanceIsServedAtItsEndpointWithItsOwnKey() {
	keys := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		keys <- r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, claudeAnswer("bonjour", 5, 2))
	}))
	s.T().Cleanup(server.Close)
	s.T().Setenv("ANTHROPIC_EU_TEST_KEY", "eu-key")
	router, err := New(Options{Registry: DefaultRegistry(), Config: s.regionalConfig(`
    - provider: anthropic
      model: claude-haiku-5-5
      languages: [en]
      region: eu
      endpoint: ` + server.URL + `
      api_key_env: ANTHROPIC_EU_TEST_KEY
`)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Region: pricing.RegionEU, Target: "anthropic/claude-haiku-5-5"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { session.Close() })
	stream, err := session.Create(s.ctx, llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hello"}}})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)

	s.Equal("bonjour", response.OutputText)
	s.Equal("eu-key", <-keys)
}

func (s *LLMRouterSuite) TestAnEUInstanceWithNoEndpointIsRefusedAtStart() {
	_, err := New(Options{Registry: DefaultRegistry(), Config: s.regionalConfig(`
    - provider: anthropic
      model: claude-haiku-5-5
      languages: [en]
      region: eu
`)})

	s.ErrorContains(err, "anthropic/claude-haiku-5-5 in eu sets no endpoint")
}

func (s *LLMRouterSuite) TestAnEndpointOnAFactoryThatDoesNotTakeOneIsRefusedAtStart() {
	_, err := New(Options{Registry: DefaultRegistry(), Config: s.regionalConfig(`
    - provider: cerebras
      model: gpt-oss-120b
      languages: [en]
      region: eu
      endpoint: https://eu.cerebras.example/v1
`)})

	s.ErrorContains(err, "cerebras/gpt-oss-120b in eu sets an endpoint, which the cerebras factory does not take")
}

func (s *LLMRouterSuite) TestAKeyVariableThatIsUnsetIsRefusedAtStart() {
	_, err := New(Options{Registry: DefaultRegistry(), Config: s.regionalConfig(`
    - provider: anthropic
      model: claude-haiku-5-5
      languages: [en]
      region: eu
      endpoint: https://eu.anthropic.example
      api_key_env: ANTHROPIC_EU_KEY_NOBODY_SET
`)})

	s.ErrorContains(err, "names its key in ANTHROPIC_EU_KEY_NOBODY_SET, which is not set")
}

func (s *LLMRouterSuite) TestTheSameModelTwiceInOneRegionIsRefused() {
	_, err := s.loaded(`
llm:
  providers:
    - {provider: anthropic, model: claude-haiku-5-5, languages: [en], region: eu, endpoint: "https://eu.anthropic.example"}
    - {provider: anthropic, model: claude-haiku-5-5, languages: [en], region: eu, endpoint: "https://eu2.anthropic.example"}
`)

	s.ErrorContains(err, "anthropic/claude-haiku-5-5 in eu is declared twice")
}
