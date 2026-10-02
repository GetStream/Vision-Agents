//go:build integration

package llmsuite

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// TestAnswersAndReportsWhatItCost is the one that says the provider works at all: the
// answer is right, it streamed, it is filed under the id it was asked under, and there is
// a token count to bill.
func (s *Suite) TestAnswersAndReportsWhatItCost() {
	params := capital
	params.ID = "c1"

	complete, events := s.Ask(params)

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal(llm.StatusCompleted, complete.Status)
	s.Equal("c1", complete.ID)
	s.Equal(s.LLM.Provider(), complete.Provider)
	s.Equal(s.LLM.Model(), complete.Model)
	s.Positive(complete.Usage.InputTokens, "there is nothing to bill without a token count")
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)
	s.LessOrEqual(complete.TimeToFirstTokenMs, complete.DurationMs)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

// TestConversationHistoryIsHonoured is what makes a second turn possible: the model is
// handed the whole conversation, not only the last thing said.
func (s *Suite) TestConversationHistoryIsHonoured() {
	complete, _ := s.Ask(favouriteNumber)

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}

// TestATruncatedAnswerSaysWhyItStopped is what lets a caller tell a model that ran out of
// room from one that had finished.
func (s *Suite) TestATruncatedAnswerSaysWhyItStopped() {
	complete, _ := s.Ask(essay)
	s.T().Logf("stopped after %d output tokens, %d of them thinking: %q",
		complete.Usage.OutputTokens, complete.Usage.OutputTokensDetails.ReasoningTokens, complete.OutputText)

	s.Equal(llm.StatusIncomplete, complete.Status)
	s.Equal(llm.ReasonMaxOutputTokens, complete.IncompleteReason)
	if !s.Reasons() {
		s.NotEmpty(complete.OutputText, "what fitted in the budget is still the answer so far")
	}
}

// TestAToolIsCalledWithItsArguments is how an agent does anything a conversation cannot:
// the model asks for the tool by name, with arguments that parse.
func (s *Suite) TestAToolIsCalledWithItsArguments() {
	called, _ := s.Ask(llm.ResponseParams{
		Instructions: "Use the tool to answer.",
		Input:        []llm.Message{weather},
		Tools:        []llm.Tool{weatherTool},
		ToolChoice:   "required",
	})

	s.Require().Len(called.ToolCalls, 1)
	call := called.ToolCalls[0]
	s.Equal(weatherTool.Name, call.Name)
	s.NotEmpty(call.ID, "the result is matched back to the call by its id")
	var arguments struct{ City string }
	s.Require().NoError(json.Unmarshal([]byte(call.Arguments), &arguments))
	s.Contains(strings.ToLower(arguments.City), "paris")
}

// TestAToolResultIsAnsweredRatherThanRefused is the turn that tells the caller what the
// tool found. A provider that rejects the replayed call, or ignores its result, leaves a
// caller who asked a question and never heard the answer.
func (s *Suite) TestAToolResultIsAnsweredRatherThanRefused() {
	called, _ := s.Ask(llm.ResponseParams{
		Instructions: "Use the tool to answer.",
		Input:        []llm.Message{weather},
		Tools:        []llm.Tool{weatherTool},
		ToolChoice:   "required",
	})
	s.Require().Len(called.ToolCalls, 1)

	answered, _ := s.Ask(llm.ResponseParams{
		Instructions: "Tell the caller what the tool found.",
		Input: []llm.Message{
			weather,
			{Role: llm.Assistant, Content: called.OutputText, ToolCalls: called.ToolCalls},
			{Role: llm.ToolResult, ToolCallID: called.ToolCalls[0].ID, Content: weatherReport},
		},
		Tools: []llm.Tool{weatherTool},
	})

	s.Equal(llm.StatusCompleted, answered.Status)
	s.Contains(answered.OutputText, "20")
}

// TestClosingStopsTheAnswerMidStream is barge-in: the caller talks over the model, and
// the model stops rather than finishing a reply nobody is listening to.
func (s *Suite) TestClosingStopsTheAnswerMidStream() {
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	stream, err := s.LLM.Create(ctx, counting)
	s.Require().NoError(err)

	// Wait for the model to start talking, then cut it off the way barge-in would. The
	// stream is drained to the end afterwards: what it generated before being closed was
	// generated all the same, and it is the last event that reports it.
	for stream.Next() {
		if _, talking := stream.Current().(llm.OutputTextDelta); talking {
			s.Require().NoError(stream.Close())
			break
		}
	}
	for stream.Next() {
	}
	s.Require().NoError(stream.Err())

	complete := stream.Response()
	s.Equal(llm.StatusCancelled, complete.Status)
	s.NotEmpty(complete.OutputText, "what was already said still counts")
	s.NotContains(complete.OutputText, "200", "the answer was cut short")
}

// TestTwoImagesAreUnderstoodInOrder is what a camera turn relies on: every frame is seen,
// and in the order it was sent.
func (s *Suite) TestTwoImagesAreUnderstoodInOrder() {
	if !s.LLM.Capabilities().Accepts(llm.ModalityImage) {
		s.T().Skip("this model takes no images")
	}
	parts := append([]llm.ContentPart{
		{Text: "Name the color of the first image, then the second. Reply only red, blue."},
	}, s.squares...)

	complete, _ := s.Ask(llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Parts: parts}}})

	answer := strings.ToLower(complete.OutputText)
	s.Contains(answer, "red")
	s.Contains(answer, "blue")
	s.Less(strings.Index(answer, "red"), strings.Index(answer, "blue"))
}

// TestReasoningStreamsAheadOfTheAnswer holds a model that says it streams its reasoning
// to it, since a caller on the live path decides what to show while it waits on that
// promise.
func (s *Suite) TestReasoningStreamsAheadOfTheAnswer() {
	if !s.LLM.Capabilities().StreamsReasoning {
		s.T().Skip("this model does not stream its reasoning")
	}

	complete, events := s.Ask(capital)

	var thinking strings.Builder
	var answered bool
	for _, event := range events {
		switch typed := event.(type) {
		case llm.ReasoningTextDelta:
			s.False(answered, "the reasoning comes before the answer, not during it")
			thinking.WriteString(typed.Delta)
		case llm.OutputTextDelta:
			answered = true
		}
	}
	s.NotEmpty(thinking.String(), "a reasoning model should show its working")
	s.Positive(complete.Usage.OutputTokensDetails.ReasoningTokens)
	s.LessOrEqual(complete.Usage.OutputTokensDetails.ReasoningTokens, complete.Usage.OutputTokens,
		"reasoning is part of the output, not an extra charge on top of it")
}
