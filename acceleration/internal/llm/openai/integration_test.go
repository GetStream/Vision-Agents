//go:build integration

package openai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type OpenAIIntegrationSuite struct {
	suite.Suite
}

func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIIntegrationSuite))
}

func (s *OpenAIIntegrationSuite) SetupSuite() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		s.T().Skip("OPENAI_API_KEY not set")
	}
}

func (s *OpenAIIntegrationSuite) start(options Options) *LLM {
	provider, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected request reports what went wrong rather than timing out.
func (s *OpenAIIntegrationSuite) ask(provider *LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
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

func (s *OpenAIIntegrationSuite) TestAnswersAndReportsWhatItCost() {
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
	s.Equal(llm.StatusCompleted, complete.Status)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *OpenAIIntegrationSuite) TestConversationHistoryIsHonoured() {
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

func (s *OpenAIIntegrationSuite) TestATruncatedAnswerSaysWhyItStopped() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Write a long essay about the sea."}},
		MaxOutputTokens: 16,
	})

	s.Equal(llm.StatusIncomplete, complete.Status)
	s.Equal(llm.ReasonMaxOutputTokens, complete.IncompleteReason)
	s.NotEmpty(complete.OutputText)
}
