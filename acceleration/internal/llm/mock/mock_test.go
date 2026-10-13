package mock

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestMockLLMSuite(t *testing.T) { suite.Run(t, new(MockLLMSuite)) }

type MockLLMSuite struct {
	suite.Suite

	provider *LLM
}

func (s *MockLLMSuite) SetupTest() {
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.provider = provider
	s.T().Cleanup(func() { s.Require().NoError(provider.Close()) })
}

func (s *MockLLMSuite) ask() *llm.Stream {
	stream, err := s.provider.Create(context.Background(), llm.ResponseParams{
		ID:    "r1",
		Input: []llm.Message{{Role: llm.User, Content: "when do you open?"}},
	})
	s.Require().NoError(err)
	return stream
}

func (s *MockLLMSuite) TestTheAnswerArrivesAWordAtATime() {
	stream := s.ask()
	defer stream.Close()

	var deltas int
	for stream.Next() {
		if _, text := stream.Current().(llm.OutputTextDelta); text {
			deltas++
		}
	}

	s.Require().NoError(stream.Err())
	s.Greater(deltas, 1, "a reply that arrived whole would hide what streaming costs")
	s.Equal(DefaultAnswer, stream.Response().OutputText)
	s.Equal(llm.StatusCompleted, stream.Response().Status)
}

func (s *MockLLMSuite) TestNothingIsBilled() {
	stream := s.ask()
	defer stream.Close()

	for stream.Next() {
	}

	s.Zero(stream.Response().Usage, "a mocked turn must not show up in anybody's usage")
}

func (s *MockLLMSuite) TestTheFirstWordWaitsAndTheRestFollowAtTheGivenPace() {
	provider, err := New(Options{TimeToFirstTokenMs: 200, TokensPerSecond: 1000})
	s.Require().NoError(err)
	defer func() { s.Require().NoError(provider.Close()) }()

	started := time.Now()
	stream, err := provider.Create(context.Background(), llm.ResponseParams{ID: "r1"})
	s.Require().NoError(err)
	defer stream.Close()
	s.True(stream.Await())

	s.GreaterOrEqual(time.Since(started), 200*time.Millisecond)

	for stream.Next() {
	}
	s.Positive(stream.Response().TimeToFirstTokenMs)
	s.Less(stream.Response().DurationMs, 1000.0, "a thousand words a second is a fast model")
}

func (s *MockLLMSuite) TestClosingTheStreamAbandonsTheAnswer() {
	stream := s.ask()
	s.True(stream.Await())
	s.Require().NoError(stream.Close())

	for stream.Next() {
	}

	s.Equal(llm.StatusCancelled, stream.Response().Status)
	s.Less(len(stream.Response().OutputText), len(DefaultAnswer),
		"barge-in should stop the model mid-sentence")
}

func (s *MockLLMSuite) TestAReasoningEffortIsRefused() {
	_, err := s.provider.Create(context.Background(), llm.ResponseParams{
		ID:        "r1",
		Reasoning: llm.ReasoningParams{Effort: "high"},
	})

	s.ErrorContains(err, "does not reason")
}
