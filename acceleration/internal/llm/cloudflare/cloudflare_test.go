package cloudflare

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type CloudflareSuite struct {
	hosttest.Unit
}

func TestCloudflareSuite(t *testing.T) {
	suite.Run(t, &CloudflareSuite{hosttest.Unit{
		Host:          Host,
		New:           New,
		Model:         "@cf/google/gemma-4-26b-a4b-it",
		ThinkingModel: "@cf/zai-org/glm-5.3",
	}})
}

// SetupTest gives the shared suite an account to build an endpoint out of, which every
// other host has baked into its base URL.
func (s *CloudflareSuite) SetupTest() {
	s.Unit.SetupTest()
	s.T().Setenv(accountEnvVar, "an-account")
}

func (s *CloudflareSuite) TestTheAccountIsPartOfTheEndpointRatherThanTheToken() {
	s.T().Setenv(accountEnvVar, "")
	_, err := New(openweights.Options{APIKey: "k", Model: s.Model})
	s.ErrorContains(err, accountEnvVar+" is required")
}
