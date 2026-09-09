package llm

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

var errUnauthorized = errors.New("unauthorized")

// chunk is one thing a scripted provider does when its stream is advanced.
type chunk func(w *ResponseWriter)

// scripted stands in for a provider, playing chunks back one Advance at a time.
type scripted struct {
	chunks []chunk
	err    error
	closed bool
}

func (s *scripted) Advance(w *ResponseWriter) bool {
	if s.closed || len(s.chunks) == 0 {
		return false
	}
	next := s.chunks[0]
	s.chunks = s.chunks[1:]
	next(w)
	return true
}

func (s *scripted) Err() error { return s.err }

func (s *scripted) Close() error {
	s.closed = true
	return nil
}

type LLMSuite struct {
	suite.Suite
}

func TestLLMSuite(t *testing.T) {
	suite.Run(t, new(LLMSuite))
}

// stream is a scripted provider's stream, named as the tests want to talk about it.
func (s *LLMSuite) stream(chunks ...chunk) *Stream {
	return NewStream(
		StreamOptions{ResponseID: "r1", Provider: "openai", Model: "gpt-5.6-luna"},
		&scripted{chunks: chunks},
	)
}

// drain advances a stream to the end and returns everything it produced.
func (s *LLMSuite) drain(stream *Stream) []Event {
	var events []Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	return events
}

func (s *LLMSuite) TestStreamGeneratesAResponseIDWhenTheCallerHasNone() {
	first := NewStream(StreamOptions{}, &scripted{})
	second := NewStream(StreamOptions{}, &scripted{})

	s.Require().True(first.Next())
	s.Require().True(second.Next())

	firstCreated := first.Current().(ResponseCreated)
	secondCreated := second.Current().(ResponseCreated)

	s.NotEmpty(firstCreated.ResponseID)
	s.NotEqual(firstCreated.ResponseID, secondCreated.ResponseID,
		"two responses must not share an id")
}

func (s *LLMSuite) TestStreamOpensWithCreatedAndEndsWithCompleted() {
	events := s.drain(s.stream(func(w *ResponseWriter) { w.OutputText("Hi") }))

	s.Require().Len(events, 3)
	s.IsType(ResponseCreated{}, events[0])
	s.IsType(OutputTextDelta{}, events[1])
	s.IsType(ResponseCompleted{}, events[2])
}

func (s *LLMSuite) TestStreamAssemblesTheAnswerFromItsDeltas() {
	stream := s.stream(
		func(w *ResponseWriter) { w.OutputText("Hello") },
		func(w *ResponseWriter) { w.OutputText(", world") },
	)
	s.drain(stream)

	s.Equal("Hello, world", stream.Response().OutputText)
	s.Equal(StatusCompleted, stream.Response().Status)
}

func (s *LLMSuite) TestStreamKeepsReasoningOutOfTheAnswer() {
	// Thinking must never be spoken as the reply, so it is streamed but not assembled.
	stream := s.stream(
		func(w *ResponseWriter) { w.ReasoningText("The user greeted me, so") },
		func(w *ResponseWriter) { w.OutputText("Hi there") },
	)
	events := s.drain(stream)

	s.Equal("Hi there", stream.Response().OutputText)
	s.IsType(ReasoningTextDelta{}, events[1])
}

func (s *LLMSuite) TestStreamNumbersDeltasInOrderAcrossBothKinds() {
	stream := s.stream(func(w *ResponseWriter) {
		w.ReasoningText("hmm")
		w.OutputText("Hi")
		w.OutputText(" there")
	})
	events := s.drain(stream)

	s.Equal(0, events[1].(ReasoningTextDelta).Index)
	s.Equal(1, events[2].(OutputTextDelta).Index)
	s.Equal(2, events[3].(OutputTextDelta).Index,
		"reasoning and text share one sequence, so order is total")
	s.Equal("r1", events[3].(OutputTextDelta).ResponseID)
}

func (s *LLMSuite) TestStreamMeasuresTimeToFirstToken() {
	stream := s.stream(
		func(w *ResponseWriter) { time.Sleep(20 * time.Millisecond); w.OutputText("Hi") },
		func(w *ResponseWriter) { time.Sleep(20 * time.Millisecond); w.OutputText(" there") },
	)
	s.drain(stream)

	response := stream.Response()
	s.GreaterOrEqual(response.TimeToFirstTokenMs, 15.0, "the wait for the first token")
	s.Less(response.TimeToFirstTokenMs, response.DurationMs,
		"the first token arrives before the last one")
}

func (s *LLMSuite) TestReasoningCountsTowardsTimeToFirstToken() {
	// A reasoning model is working while it thinks, so the caller's wait ends there.
	stream := s.stream(
		func(w *ResponseWriter) { time.Sleep(20 * time.Millisecond); w.ReasoningText("thinking") },
		func(w *ResponseWriter) { time.Sleep(20 * time.Millisecond); w.OutputText("Hi") },
	)
	s.drain(stream)

	s.Less(stream.Response().TimeToFirstTokenMs, 35.0,
		"thinking is the first sign of life, not the answer")
}

func (s *LLMSuite) TestResponseWithNoOutputReportsNoTimeToFirstToken() {
	stream := s.stream()
	s.drain(stream)

	s.Zero(stream.Response().TimeToFirstTokenMs, "nothing came back, so there was no first token")
	s.Empty(stream.Response().OutputText)
}

func (s *LLMSuite) TestStreamKeepsTheLastUsageItWasTold() {
	// Providers repeat a cumulative usage frame on every chunk, so the last one is right.
	stream := s.stream(
		func(w *ResponseWriter) {
			w.Usage(Usage{InputTokens: 15, OutputTokens: 43,
				OutputTokensDetails: OutputTokensDetails{ReasoningTokens: 43}})
		},
		func(w *ResponseWriter) {
			w.Usage(Usage{
				InputTokens:         15,
				InputTokensDetails:  InputTokensDetails{CachedTokens: 8, CacheWriteTokens: 4},
				OutputTokens:        64,
				OutputTokensDetails: OutputTokensDetails{ReasoningTokens: 47},
			})
		},
	)
	s.drain(stream)

	usage := stream.Response().Usage
	s.EqualValues(15, usage.InputTokens)
	s.EqualValues(8, usage.InputTokensDetails.CachedTokens)
	s.EqualValues(4, usage.InputTokensDetails.CacheWriteTokens)
	s.EqualValues(64, usage.OutputTokens)
	s.EqualValues(47, usage.OutputTokensDetails.ReasoningTokens)
	s.EqualValues(79, usage.TotalTokens, "a provider that reports no total gets one")
}

func (s *LLMSuite) TestClosingAStreamAbandonsTheResponseButStillReportsWhatItCost() {
	stream := s.stream(
		func(w *ResponseWriter) {
			w.OutputText("this will be cut ")
			w.Usage(Usage{InputTokens: 10, OutputTokens: 4})
		},
		func(w *ResponseWriter) { w.OutputText("but this never arrives") },
	)

	s.Require().True(stream.Next()) // created
	s.Require().True(stream.Next()) // the delta that did arrive
	s.Require().NoError(stream.Close())
	s.drain(stream)

	response := stream.Response()
	s.Equal(StatusCancelled, response.Status)
	s.Equal("this will be cut ", response.OutputText, "the text that did arrive still counts")
	s.EqualValues(4, response.Usage.OutputTokens, "the tokens already generated are still billed")
}

func (s *LLMSuite) TestStreamReportsWhyTheModelStoppedEarly() {
	stream := s.stream(func(w *ResponseWriter) {
		w.OutputText("truncated")
		w.Incomplete(ReasonMaxOutputTokens)
	})
	s.drain(stream)

	s.Equal(StatusIncomplete, stream.Response().Status)
	s.Equal(ReasonMaxOutputTokens, stream.Response().IncompleteReason)
}

func (s *LLMSuite) TestStreamAssemblesToolCallsInTheOrderTheModelAskedForThem() {
	stream := s.stream(
		func(w *ResponseWriter) { w.FunctionCall(0, "call_1", "transfer", `{"to":`, "") },
		func(w *ResponseWriter) { w.FunctionCall(1, "call_2", "press", `{"digit"`, "sig") },
		func(w *ResponseWriter) { w.FunctionCall(0, "", "", `"sales"}`, "") },
		func(w *ResponseWriter) { w.FunctionCall(1, "", "", `:"1"}`, "") },
	)
	s.drain(stream)

	calls := stream.Response().ToolCalls
	s.Require().Len(calls, 2)
	s.Equal(ToolCall{ID: "call_1", Name: "transfer", Arguments: `{"to":"sales"}`}, calls[0])
	s.Equal(ToolCall{ID: "call_2", Name: "press", Arguments: `{"digit":"1"}`, Signature: "sig"},
		calls[1])
}

func (s *LLMSuite) TestAToolCallTheProviderDidNotIdentifyGetsAnID() {
	stream := s.stream(func(w *ResponseWriter) { w.FunctionCall(0, "", "transfer", "{}", "") })
	s.drain(stream)

	s.Equal("r1-tool-0", stream.Response().ToolCalls[0].ID)
}

func (s *LLMSuite) TestAFailedResponseStillSettles() {
	stream := NewStream(
		StreamOptions{ResponseID: "r1", Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731"},
		&scripted{err: errUnauthorized},
	)
	events := s.drain(stream)

	s.Require().Len(events, 3)
	s.IsType(ResponseFailed{}, events[1])
	s.Equal(StatusFailed, stream.Response().Status,
		"a caller counting responses still sees this one end")
	s.ErrorIs(stream.Err(), errUnauthorized)
}

func (s *LLMSuite) TestFailedResponseUnwrapsToTheProviderFailure() {
	failure := ResponseFailed{Provider: "deepseek", Err: errUnauthorized, Context: "request"}

	s.ErrorIs(failure, errUnauthorized)
	s.Equal("unauthorized", failure.Error())
}

func (s *LLMSuite) TestCollectReturnsTheWholeAnswer() {
	stream := s.stream(
		func(w *ResponseWriter) { w.OutputText("half ") },
		func(w *ResponseWriter) { w.OutputText("an answer") },
	)

	response, err := Collect(stream)

	s.Require().NoError(err)
	s.Equal("half an answer", response.OutputText)
}

func (s *LLMSuite) TestCollectReportsAProviderFailure() {
	stream := NewStream(StreamOptions{}, &scripted{err: errUnauthorized})

	_, err := Collect(stream)

	s.ErrorIs(err, errUnauthorized)
}

func (s *LLMSuite) TestReplayStandsInForAProvider() {
	stream := Replay(
		ResponseCreated{ResponseID: "r1"},
		OutputTextDelta{ResponseID: "r1", Delta: "canned"},
		ResponseCompleted{Response: Response{ID: "r1", OutputText: "canned",
			Status: StatusCompleted}},
	)

	response, err := Collect(stream)

	s.Require().NoError(err)
	s.Equal("canned", response.OutputText)
}

func (s *LLMSuite) TestCapabilitiesRefuseAnEffortTheModelDoesNotAccept() {
	model := Capabilities{ReasoningEfforts: []string{"none", "low", "medium"}}

	err := model.Validate(ResponseParams{Reasoning: ReasoningParams{Effort: "xhigh"}})

	s.Require().Error(err)
	s.Contains(err.Error(), "none, low, medium", "the caller is told what is valid")
	s.NoError(model.Validate(ResponseParams{Reasoning: ReasoningParams{Effort: "low"}}))
}

func (s *LLMSuite) TestCapabilitiesRefuseReasoningFromAModelThatDoesNotReason() {
	err := Capabilities{}.Validate(ResponseParams{Reasoning: ReasoningParams{Effort: "low"}})

	s.Require().Error(err)
	s.Contains(err.Error(), "does not reason")
}

func (s *LLMSuite) TestCapabilitiesRefuseAConversationAndAPreviousResponseTogether() {
	err := Capabilities{}.Validate(ResponseParams{
		Conversation:       "conv_1",
		PreviousResponseID: "resp_1",
	})

	s.Require().Error(err)
	s.Contains(err.Error(), "not both")
}

func (s *LLMSuite) TestEffortFallsBackToTheModelsOwnDefault() {
	model := Capabilities{ReasoningEfforts: []string{"none", "medium"}, DefaultEffort: "none"}

	s.Equal("none", model.Effort(ResponseParams{}))
	s.Equal("medium", model.Effort(ResponseParams{Reasoning: ReasoningParams{Effort: "medium"}}))
	s.Empty(Capabilities{}.Effort(ResponseParams{}), "a model that does not reason sends nothing")
}

func (s *LLMSuite) TestUnfenceReadsJSONAModelWrappedInACodeFence() {
	s.Equal(`{"ok":true}`, Unfence("```json\n{\"ok\":true}\n```"))
	s.Equal(`{"ok":true}`, Unfence(`{"ok":true}`))
}
