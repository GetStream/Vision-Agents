package minimax

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/stretchr/testify/suite"
)

type MiniMaxSuite struct {
	suite.Suite
}

func TestMiniMaxSuite(t *testing.T) {
	suite.Run(t, new(MiniMaxSuite))
}

func (s *MiniMaxSuite) SetupTest() {
	for _, name := range apiKeyEnvVars {
		s.T().Setenv(name, "")
	}
	s.T().Setenv(baseURLEnvVar, "")
}

func (s *MiniMaxSuite) TestCredentialsComeFromEitherVariable() {
	_, err := New(Options{})
	s.ErrorContains(err, "MINIMAX_API_KEY is required")

	s.T().Setenv("MINIMAX", "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
	s.Equal(defaultModel, provider.Model())
}

func (s *MiniMaxSuite) TestItAlwaysThinksAtLowUnlessTold() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	model := provider.Capabilities()
	s.True(model.StreamsReasoning)
	s.Equal("low", model.DefaultEffort)
	s.Equal(map[string]any{"reasoning_effort": "low", "temperature": temperature},
		requestFields("low")(llm.ResponseParams{}, ""))
}

func (s *MiniMaxSuite) TestARequestedEffortAndTemperatureWin() {
	zero := 0.0
	s.Equal(map[string]any{"reasoning_effort": "medium"},
		requestFields("low")(llm.ResponseParams{Temperature: &zero}, "medium"))
}
