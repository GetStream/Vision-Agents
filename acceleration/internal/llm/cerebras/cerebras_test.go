package cerebras

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type CerebrasSuite struct {
	suite.Suite
}

func TestCerebrasSuite(t *testing.T) {
	suite.Run(t, new(CerebrasSuite))
}

func (s *CerebrasSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.T().Setenv(baseURLEnvVar, "")
}

func (s *CerebrasSuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *CerebrasSuite) TestTheModelIsTheIdCerebrasServes() {
	provider, err := New(Options{APIKey: "k", Model: "gemma-4-31b"})
	s.Require().NoError(err)

	s.Equal("gemma-4-31b", provider.Model())
}

func (s *CerebrasSuite) TestTheDefaultIsGemma4() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.Equal(defaultModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.Empty(provider.Capabilities().ReasoningEfforts,
		"reasoning is the wrong trade on the live path")
}
