package openai

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type OpenAISuite struct {
	suite.Suite
}

func TestOpenAISuite(t *testing.T) {
	suite.Run(t, new(OpenAISuite))
}

func (s *OpenAISuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
}

func (s *OpenAISuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *OpenAISuite) TestModelIsSentUnqualified() {
	// OpenAI's own model ids carry no owner prefix, unlike the open-weight providers.
	provider, err := New(Options{APIKey: "k", Model: "gpt-5.6-terra"})
	s.Require().NoError(err)

	s.Equal("gpt-5.6-terra", provider.Model())
}

func (s *OpenAISuite) TestADefaultModelIsUsedWhenNoneIsNamed() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.Equal(defaultModel, provider.Model())
}

func (s *OpenAISuite) TestReasoningIsNotClaimedForAChatModel() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.False(provider.Reasoning())
}
