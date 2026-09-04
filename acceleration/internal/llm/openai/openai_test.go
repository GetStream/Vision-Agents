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

func (s *OpenAISuite) TestAModelDeclaresTheReasoningEffortsItAccepts() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	model := provider.Capabilities()
	s.Contains(model.ReasoningEfforts, "max", "the 5.6 family is the first to take it")
	s.Equal("none", model.DefaultEffort,
		"a conversation cannot afford to wait while the model thinks")
	s.True(model.Store, "the responses endpoint keeps what it generates")
}

func (s *OpenAISuite) TestAnEffortTheModelDoesNotAcceptIsRefused() {
	_, err := New(Options{APIKey: "k", Model: "gpt-5-mini", ReasoningEffort: "max"})

	s.Require().Error(err)
	s.ErrorContains(err, "minimal, low, medium, high")
}

func (s *OpenAISuite) TestAModelIsRecognisedByItsFamily() {
	// A dated snapshot is the same model, so it accepts the same efforts.
	s.Equal(modelCapabilities["gpt-5.6"].ReasoningEfforts,
		capabilitiesFor("gpt-5.6-sol-2026-02-11").ReasoningEfforts)
	s.Equal(fallbackCapabilities, capabilitiesFor("some-future-model"))
}
