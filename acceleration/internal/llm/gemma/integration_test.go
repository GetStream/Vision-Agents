//go:build integration

package gemma

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type GemmaIntegrationSuite struct {
	suite.Suite
}

func TestGemmaIntegrationSuite(t *testing.T) {
	suite.Run(t, new(GemmaIntegrationSuite))
}

func (s *GemmaIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " not set")
	}
	if os.Getenv(baseURLEnvVar) == "" {
		s.T().Skip(baseURLEnvVar + " not set: see deploy/gemma-4")
	}
}

func (s *GemmaIntegrationSuite) start(options Options) *openaicompat.LLM {
	provider, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected request reports what went wrong rather than timing out.
func (s *GemmaIntegrationSuite) ask(provider *openaicompat.LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
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

func (s *GemmaIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	complete, events := s.ask(provider, llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxOutputTokens: 32,
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

func (s *GemmaIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Answer with a single number and nothing else.",
		Input: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxOutputTokens: 32,
	})

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}
