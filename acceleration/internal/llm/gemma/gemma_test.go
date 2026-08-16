package gemma

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type GemmaSuite struct {
	suite.Suite
}

func TestGemmaSuite(t *testing.T) {
	suite.Run(t, new(GemmaSuite))
}

func (s *GemmaSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.T().Setenv(baseURLEnvVar, "")
}

func (s *GemmaSuite) TestAnUndeployedGemmaFailsToBuild() {
	// Failing here is what makes routing move to the next candidate, so a capability
	// shortcut still resolves while nobody has pushed the deployment.
	s.T().Setenv(apiKeyEnvVar, "k")

	_, err := New(Options{})
	s.ErrorContains(err, baseURLEnvVar+" is required")
	s.ErrorContains(err, "deploy/gemma-4", "the error says where the recipe is")
}

func (s *GemmaSuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	s.T().Setenv(baseURLEnvVar, "https://model-abc.api.baseten.co/environments/production/sync/v1")

	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *GemmaSuite) TestStatsUseTheRoutingNameNotTheQualifiedModelID() {
	provider, err := New(Options{APIKey: "k", BaseURL: "http://x/v1", Model: "gemma-4-E2B-it"})
	s.Require().NoError(err)

	s.Equal("gemma-4-E2B-it", provider.Model())
}

func (s *GemmaSuite) TestModelIsQualifiedWithItsOwnerForVLLM() {
	s.Equal("google/gemma-4-E2B-it", upstreamModel("gemma-4-E2B-it"))
	s.Equal("google/gemma-4-31B-it", upstreamModel("google/gemma-4-31B-it"),
		"an already-qualified id is left alone")
}
