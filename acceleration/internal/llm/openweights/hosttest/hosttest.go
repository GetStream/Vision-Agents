// Package hosttest holds the suites every host of open-weight models is held to.
//
// Twenty hosts serve the same weights over the same protocol, so they are owed the same
// tests rather than twenty transcriptions of them. A host package embeds Unit in its own
// suite, and Live behind the integration tag.
package hosttest

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// liveTimeout bounds one live response. A model that has not answered by now has failed
// as far as a conversation is concerned.
const liveTimeout = 120 * time.Second

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

// Live is what every host package is checked for against the host itself.
type Live struct {
	suite.Suite
	// Host is the host under test.
	Host openweights.Host
	// New builds a provider.
	New func(openweights.Options) (*openaicompat.LLM, error)
	// Model is the id this host is asked for.
	Model string
}

func (s *Live) SetupSuite() {
	if os.Getenv(s.Host.APIKeyEnvVar) == "" {
		s.T().Skip(s.Host.APIKeyEnvVar + " not set")
	}
}

// start builds a provider and closes it when the test ends.
func (s *Live) start() *openaicompat.LLM {
	provider, err := s.New(openweights.Options{Model: s.Model})
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected
// request reports what went wrong rather than timing out.
func (s *Live) ask(provider *openaicompat.LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	defer cancel()

	stream, err := provider.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	s.Require().NoError(stream.Err())
	return stream.Response(), events
}

func (s *Live) TestAnswersAndReportsWhatItCost() {
	provider := s.start()

	complete, events := s.ask(provider, llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxOutputTokens: 512,
	})

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal("c1", complete.ID)
	s.Positive(complete.Usage.InputTokens)
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *Live) TestConversationHistoryIsHonoured() {
	provider := s.start()

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Answer with a single number and nothing else.",
		Input: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxOutputTokens: 512,
	})

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}
