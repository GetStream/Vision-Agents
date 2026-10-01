// Package hosttest holds the suites every host of open-weight models is held to.
//
// Twenty hosts serve the same weights over the same protocol, so they are owed the same
// tests rather than twenty transcriptions of them. A host package embeds Unit in its own
// suite, and the llmsuite.Suite that Live builds behind the integration tag.
package hosttest

import (
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// Unit is what every host package is checked for without reaching the network.
type Unit struct {
	suite.Suite
	// Host is the host under test.
	Host openweights.Host
	// New builds a provider, which is the host package's own constructor rather than
	// Host.New, so the wiring in between is what is being tested.
	New func(openweights.Options) (*openaicompat.LLM, error)
	// Model is an id this host serves. It is only ever sent upstream, never reached.
	Model string
	// ThinkingModel is an id this host serves whose weights reason. Empty when the host
	// serves none.
	ThinkingModel string
}

func (s *Unit) SetupTest() {
	s.T().Setenv(s.Host.APIKeyEnvVar, "")
	s.T().Setenv(s.Host.BaseURLEnvVar, "")
}

func (s *Unit) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := s.New(openweights.Options{Model: s.Model})
	s.ErrorContains(err, s.Host.APIKeyEnvVar+" is required")

	s.T().Setenv(s.Host.APIKeyEnvVar, "from-env")
	provider, err := s.New(openweights.Options{Model: s.Model})
	s.Require().NoError(err)
	s.Equal(s.Host.Provider, provider.Provider())
}

func (s *Unit) TestAModelIsRequiredBecauseEveryHostSpellsThemDifferently() {
	_, err := s.New(openweights.Options{APIKey: "k"})
	s.ErrorContains(err, "model is required")
}

func (s *Unit) TestTheModelIdTravelsAsThisHostSpellsIt() {
	provider, err := s.New(openweights.Options{APIKey: "k", Model: s.Model})
	s.Require().NoError(err)

	s.Equal(s.Model, provider.Model())
}

func (s *Unit) TestReasoningIsOffUnlessTheModelEntryAsksForIt() {
	provider, err := s.New(openweights.Options{APIKey: "k", Model: s.Model})
	s.Require().NoError(err)

	s.False(provider.Capabilities().StreamsReasoning,
		"thinking is the wrong trade on the live path")
	s.Empty(provider.Capabilities().ReasoningEfforts)
}

func (s *Unit) TestAThinkingModelReportsThatItStreamsItsReasoning() {
	if s.ThinkingModel == "" {
		s.T().Skip("this host serves no reasoning weights")
	}

	provider, err := s.New(openweights.Options{APIKey: "k", Model: s.ThinkingModel, Thinking: true})
	s.Require().NoError(err)

	s.True(provider.Capabilities().StreamsReasoning)
}
