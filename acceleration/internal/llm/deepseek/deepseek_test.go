package deepseek

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type DeepSeekSuite struct {
	suite.Suite
}

func TestDeepSeekSuite(t *testing.T) {
	suite.Run(t, new(DeepSeekSuite))
}

func (s *DeepSeekSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.T().Setenv(baseURLEnvVar, "")
}

func (s *DeepSeekSuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *DeepSeekSuite) TestStatsUseTheRoutingNameNotTheQualifiedModelID() {
	provider, err := New(Options{APIKey: "k", Model: "DeepSeek-V4-Flash-0731"})
	s.Require().NoError(err)

	s.Equal("DeepSeek-V4-Flash-0731", provider.Model(),
		"the config names the model, so stats and health key on that")
}

func (s *DeepSeekSuite) TestModelIsQualifiedWithItsOwnerForBaseten() {
	s.Equal("deepseek-ai/DeepSeek-V4-Flash-0731", upstreamModel("DeepSeek-V4-Flash-0731"))
	s.Equal("deepseek-ai/DeepSeek-V4-Pro-0813", upstreamModel("deepseek-ai/DeepSeek-V4-Pro-0813"),
		"an already-qualified id is left alone")
}

func (s *DeepSeekSuite) TestThinkingIsOffByDefault() {
	// Reasoning spends the token budget and the latency before the first word of the
	// answer, which is the wrong trade on the live path.
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.False(provider.Reasoning())
}

func (s *DeepSeekSuite) TestThinkingCanBeTurnedOn() {
	provider, err := New(Options{APIKey: "k", Thinking: true, ReasoningEffort: "high"})
	s.Require().NoError(err)

	s.True(provider.Reasoning())
}

func (s *DeepSeekSuite) TestBaseURLCanBeOverriddenForADedicatedDeployment() {
	s.T().Setenv(baseURLEnvVar, "https://model-abc.api.baseten.co/environments/production/sync/v1")

	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)
	s.NotNil(provider.Client())
}
