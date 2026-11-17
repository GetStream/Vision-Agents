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
	provider, err := New(Options{Model: "qwen-3.8-27b"})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *CerebrasSuite) TestTheModelIsTheIdCerebrasServes() {
	provider, err := New(Options{APIKey: "k", Model: "qwen-3.8-27b"})
	s.Require().NoError(err)

	s.Equal("qwen-3.8-27b", provider.Model())
}

func (s *CerebrasSuite) TestAModelHasToBeNamed() {
	_, err := New(Options{APIKey: "k"})

	s.ErrorContains(err, "model is required", "the model it used to default to is archived")
}
