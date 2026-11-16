package mock

import (
	"context"
	"encoding/json"
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

func (s *MockLLMSuite) TestTheFlowControllerIsToldTheWordsWereMeantForTheAgent() {
	// The one question whose answer decides whether a turn happens at all. A sentence here
	// would be unparseable, and an unparseable decision is one the router warns about on
	// every turn of a run.
	answer := s.answerTo(llm.ResponseParams{
		ID: "turn-1", Purpose: "flow", Text: llm.TextParams{Format: llm.FormatJSONSchema},
	})

	var decided struct {
		Disposition string `json:"disposition"`
		Floor       string `json:"floor"`
	}
	s.Require().NoError(json.Unmarshal([]byte(answer), &decided))
	s.Equal("respond", decided.Disposition)
	s.Equal("continue", decided.Floor)
}

func (s *MockLLMSuite) TestAnythingElseAskedForAnObjectGetsOne() {
	answer := s.answerTo(llm.ResponseParams{ID: "s1", Text: llm.TextParams{Format: llm.FormatJSONObject}})

	s.JSONEq("{}", answer, "naming a conversation must not fail on a sentence")
}

// answerTo drains a response and returns what it said.
func (s *MockLLMSuite) answerTo(params llm.ResponseParams) string {
	stream, err := s.provider.Create(context.Background(), params)
	s.Require().NoError(err)
	defer stream.Close()

	for stream.Next() {
	}
	s.Require().NoError(stream.Err())
	return stream.Response().OutputText
}

func (s *MockLLMSuite) TestAReasoningEffortIsRefused() {
	_, err := s.provider.Create(context.Background(), llm.ResponseParams{
		ID:        "r1",
		Reasoning: llm.ReasoningParams{Effort: "high"},
	})

	s.ErrorContains(err, "does not reason")
}
