package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type OpenAICompatSuite struct {
	suite.Suite

	// server speaks the chat completions protocol so the provider is exercised over real
	// HTTP rather than against a substitute for it.
	server *httptest.Server
	// frames is what the next request streams back, one SSE data payload per entry.
	frames []string
	// mu guards requests, because a test with two completions in flight is served by two
	// goroutines at once.
	mu sync.Mutex
	// requests records the decoded bodies the provider sent.
	requests []map[string]any
	// hold blocks the response until closed, for tests about cancellation.
	hold chan struct{}
	// status replaces 200 when non-zero.
	status int
}

func TestOpenAICompatSuite(t *testing.T) {
	suite.Run(t, new(OpenAICompatSuite))
}

func (s *OpenAICompatSuite) SetupTest() {
	s.frames = nil
	s.requests = nil
	s.hold = nil
	s.status = 0

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		s.Require().NoError(err)

		var body map[string]any
		s.Require().NoError(json.Unmarshal(raw, &body))
		s.mu.Lock()
		s.requests = append(s.requests, body)
		s.mu.Unlock()

		if s.status != 0 {
			w.WriteHeader(s.status)
			fmt.Fprint(w, `{"error":{"message":"nope"}}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		s.Require().True(ok)

		for _, frame := range s.frames {
			fmt.Fprintf(w, "data: %s\n\n", frame)
			flusher.Flush()
		}
		if s.hold != nil {
			select {
			case <-s.hold:
			case <-r.Context().Done():
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	s.T().Cleanup(s.server.Close)
}

// provider returns a started provider pointed at the test server.
func (s *OpenAICompatSuite) provider(options Options) *LLM {
	if options.Provider == "" {
		options.Provider = "test"
	}
	if options.Model == "" {
		options.Model = "test-model"
	}
	options.APIKey = "key"
	options.BaseURL = s.server.URL

	provider, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// frame renders one chunk as a single-line SSE payload, which is what the protocol
// requires: a newline inside the JSON would end the data field early.
func frame(choices []any, usage map[string]any) string {
	chunk := map[string]any{
		"id":      "c",
		"object":  "chat.completion.chunk",
		"created": 1,
		"model":   "m",
		"choices": choices,
	}
	if usage != nil {
		chunk["usage"] = usage
	}

	encoded, err := json.Marshal(chunk)
	if err != nil {
		panic(err)
	}
	return string(encoded)
}

// textFrame is a chunk carrying one piece of the answer.
func textFrame(text string) string {
	return frame([]any{map[string]any{
		"index": 0,
		"delta": map[string]any{"content": text},
	}}, nil)
}

// reasoningFrame is a chunk carrying a piece of the model's thinking.
func reasoningFrame(text string) string {
	return frame([]any{map[string]any{
		"index": 0,
		"delta": map[string]any{"reasoning_content": text},
	}}, nil)
}

// toolFrame is a chunk carrying one fragment of a tool call. The id and the name arrive on
// the first fragment and the arguments dribble in over the ones after it, which is how a
// provider streams a call.
func toolFrame(index int, id, name, arguments string) string {
	call := map[string]any{
		"index":    index,
		"function": map[string]any{},
	}
	if id != "" {
		call["id"] = id
		call["type"] = "function"
	}
	function := call["function"].(map[string]any)
	if name != "" {
		function["name"] = name
	}
	if arguments != "" {
		function["arguments"] = arguments
	}

	return frame([]any{map[string]any{
		"index": 0,
		"delta": map[string]any{"tool_calls": []any{call}},
	}}, nil)
}

// signedToolFrame renders a call from a provider that signs what it asks for, which it
// nests under extra_content beside the function rather than inside it.
func signedToolFrame(index int, id, name, arguments, signature string) string {
	return frame([]any{map[string]any{
		"index": 0,
		"delta": map[string]any{"tool_calls": []any{map[string]any{
			"index": index,
			"id":    id,
			"type":  "function",
			"function": map[string]any{
				"name":      name,
				"arguments": arguments,
			},
			"extra_content": map[string]any{
				"google": map[string]any{"thought_signature": signature},
			},
		}}},
	}}, nil)
}

// usageFrame reports what the completion consumed, optionally settling it with a finish
// reason. Choices are empty when there is no finish reason, the way a terminal usage-only
// chunk arrives.
func usageFrame(prompt, cached, completion, reasoning int64, finishReason string) string {
	choices := []any{}
	if finishReason != "" {
		choices = append(choices, map[string]any{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": finishReason,
		})
	}

	return frame(choices, map[string]any{
		"prompt_tokens":             prompt,
		"completion_tokens":         completion,
		"total_tokens":              prompt + completion,
		"prompt_tokens_details":     map[string]any{"cached_tokens": cached},
		"completion_tokens_details": map[string]any{"reasoning_tokens": reasoning},
	})
}

// ask sends one request and returns the response and every event it produced.
func (s *OpenAICompatSuite) ask(provider *LLM, params llm.ResponseParams) (
	llm.Response, []llm.Event,
) {
	stream, err := provider.Create(context.Background(), params)
	s.Require().NoError(err)

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	return stream.Response(), events
}

// hello is the simplest request a test can make.
func hello() llm.ResponseParams {
	return llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}}
}

func (s *OpenAICompatSuite) TestNewRejectsAConfigItCannotUse() {
	_, err := New(Options{Model: "m", APIKey: "k", BaseURL: "http://x"})
	s.ErrorContains(err, "provider name is required")

	_, err = New(Options{Provider: "p", APIKey: "k", BaseURL: "http://x"})
	s.ErrorContains(err, "model is required")

	_, err = New(Options{Provider: "p", Model: "m", BaseURL: "http://x"})
	s.ErrorContains(err, "api key is required")

	_, err = New(Options{Provider: "p", Model: "m", APIKey: "k"})
	s.ErrorContains(err, "base url is required")
}

func (s *OpenAICompatSuite) TestStreamedDeltasAssembleIntoTheAnswer() {
	s.frames = []string{textFrame("Hello"), textFrame(", world"), usageFrame(11, 0, 4, 0, "stop")}
	provider := s.provider(Options{})

	response, events := s.ask(provider, hello())

	s.Equal("Hello, world", response.OutputText)
	s.Equal(llm.StatusCompleted, response.Status)

	var deltas []string
	for _, event := range events {
		if delta, ok := event.(llm.OutputTextDelta); ok {
			deltas = append(deltas, delta.Delta)
		}
	}
	s.Equal([]string{"Hello", ", world"}, deltas, "the caller can speak the answer as it arrives")
}

func (s *OpenAICompatSuite) TestUsageIsReportedForBilling() {
	s.frames = []string{textFrame("hi"), usageFrame(15, 8, 64, 47, "stop")}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.EqualValues(15, response.Usage.InputTokens)
	s.EqualValues(8, response.Usage.InputTokensDetails.CachedTokens)
	s.EqualValues(64, response.Usage.OutputTokens)
	s.EqualValues(47, response.Usage.OutputTokensDetails.ReasoningTokens)
}

func (s *OpenAICompatSuite) TestCumulativeUsageFramesSettleOnTheLastOne() {
	// Baseten repeats a growing usage object on every chunk instead of sending one at the
	// end, so billing must not add them up.
	s.frames = []string{
		usageFrame(15, 0, 10, 0, ""),
		usageFrame(15, 0, 30, 0, ""),
		usageFrame(15, 0, 43, 0, "stop"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.EqualValues(43, response.Usage.OutputTokens, "the last frame is the total, not one more instalment")
	s.EqualValues(15, response.Usage.InputTokens)
}

func (s *OpenAICompatSuite) TestThinkingIsSeparatedFromTheAnswer() {
	s.frames = []string{
		reasoningFrame("The user said hi, so"),
		textFrame("Hello!"),
		usageFrame(11, 0, 20, 16, "stop"),
	}
	provider := s.provider(Options{Capabilities: llm.Capabilities{StreamsReasoning: true}})

	response, events := s.ask(provider, hello())

	s.Equal("Hello!", response.OutputText, "thinking must never be spoken as the reply")

	var thinking string
	for _, event := range events {
		if delta, ok := event.(llm.ReasoningTextDelta); ok {
			thinking += delta.Delta
		}
	}
	s.Equal("The user said hi, so", thinking, "but it is still available to a caller that wants it")
}

func (s *OpenAICompatSuite) TestInstructionsAreSentAsASystemMessage() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, llm.ResponseParams{
		Instructions: "Be terse.",
		Input:        []llm.Message{{Role: llm.User, Content: "hi"}},
	})

	messages := s.sentMessages(0)
	s.Require().Len(messages, 2)
	s.Equal("system", messages[0]["role"])
	s.Equal("Be terse.", messages[0]["content"])
	s.Equal("user", messages[1]["role"])
}

func (s *OpenAICompatSuite) TestTheWholeConversationIsSentEveryTurn() {
	// Routing may send consecutive turns to different providers, so history has to travel
	// with the request rather than living upstream.
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.User, Content: "first"},
		{Role: llm.Assistant, Content: "answer"},
		{Role: llm.User, Content: "second"},
	}})

	messages := s.sentMessages(0)
	s.Require().Len(messages, 3)
	s.Equal("user", messages[0]["role"])
	s.Equal("assistant", messages[1]["role"])
	s.Equal("second", messages[2]["content"])
}

func (s *OpenAICompatSuite) TestUsageIsAlwaysRequested() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, hello())

	options, ok := s.requests[0]["stream_options"].(map[string]any)
	s.Require().True(ok, "without usage there is nothing to bill")
	s.Equal(true, options["include_usage"])
}

func (s *OpenAICompatSuite) TestAProvidersOwnRequestFieldsReachIt() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{
		RequestFields: func(llm.ResponseParams, string) map[string]any {
			return map[string]any{
				"chat_template_kwargs": map[string]any{"thinking": false},
			}
		},
	})

	s.ask(provider, hello())

	args, ok := s.requests[0]["chat_template_kwargs"].(map[string]any)
	s.Require().True(ok, "a provider's own request fields must survive")
	s.Equal(false, args["thinking"])
}

func (s *OpenAICompatSuite) TestTheReasoningEffortIsResolvedBeforeItIsSent() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	var sent string
	provider := s.provider(Options{
		Capabilities: llm.Capabilities{
			ReasoningEfforts: []string{"low", "high"},
			DefaultEffort:    "low",
		},
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			sent = effort
			return nil
		},
	})

	s.ask(provider, hello())
	s.Equal("low", sent, "a request naming none gets the model's own default")

	s.ask(provider, llm.ResponseParams{Input: hello().Input,
		Reasoning: llm.ReasoningParams{Effort: "high"}})
	s.Equal("high", sent)
}

func (s *OpenAICompatSuite) TestAnEffortTheModelDoesNotAcceptIsRefused() {
	provider := s.provider(Options{
		Capabilities: llm.Capabilities{ReasoningEfforts: []string{"low", "high"}},
	})

	_, err := provider.Create(context.Background(), llm.ResponseParams{
		Input:     hello().Input,
		Reasoning: llm.ReasoningParams{Effort: "max"},
	})

	s.Require().Error(err)
	s.ErrorContains(err, "low, high", "the caller is told what is valid")
}

func (s *OpenAICompatSuite) TestChatCompletionsCannotStoreOrCacheByKey() {
	// The protocol has nowhere to put them, so they are reported as unsupported rather
	// than silently promised.
	provider := s.provider(Options{
		Capabilities: llm.Capabilities{Store: true, PromptCacheKey: true, Conversations: true},
	})

	model := provider.Capabilities()
	s.False(model.Store)
	s.False(model.PromptCacheKey)
	s.False(model.Conversations)
}

func (s *OpenAICompatSuite) TestMaxTokensAndTemperatureAreOnlySentWhenAsked() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, hello())
	s.NotContains(s.requests[0], "max_completion_tokens")
	s.NotContains(s.requests[0], "temperature", "zero is a real temperature, so unset must not send one")

	temperature := 0.0
	s.ask(provider, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "hi"}},
		MaxOutputTokens: 32,
		Temperature:     &temperature,
	})
	s.EqualValues(32, s.requests[1]["max_completion_tokens"])
	s.EqualValues(0.0, s.requests[1]["temperature"])
}

func (s *OpenAICompatSuite) TestJSONIsOnlyAskedForWhenWanted() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, hello())
	s.NotContains(s.requests[0], "response_format", "prose is what a conversation wants")

	s.ask(provider, llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
		Text:  llm.TextParams{Format: llm.FormatJSONObject},
	})
	format, ok := s.requests[1]["response_format"].(map[string]any)
	s.Require().True(ok, "a caller that parses the answer has to be able to ask for JSON")
	s.Equal("json_object", format["type"])
}

func (s *OpenAICompatSuite) TestClosingAStreamSettlesWhatHadAlreadyArrived() {
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("I was saying"), usageFrame(11, 0, 4, 0, "")}
	provider := s.provider(Options{})

	stream, err := provider.Create(context.Background(), llm.ResponseParams{
		ID: "c1", Input: hello().Input,
	})
	s.Require().NoError(err)

	// Read up to the text before cutting the model off.
	s.Require().True(stream.Next())
	s.Require().True(stream.Next())
	s.Require().IsType(llm.OutputTextDelta{}, stream.Current())
	s.Require().NoError(stream.Close())

	response := drain(stream)
	s.Equal(llm.StatusCancelled, response.Status)
	s.Equal("I was saying", response.OutputText, "the words already spoken still happened")
	s.EqualValues(4, response.Usage.OutputTokens, "and are still billed")
	close(s.hold)
}

// drain reads a stream to the end and returns what it settled as.
func drain(stream *llm.Stream) llm.Response {
	for stream.Next() {
	}
	return stream.Response()
}

func (s *OpenAICompatSuite) TestAnAbandonedResponseIsNotReportedAsAnError() {
	// Barge-in is the design working, so it must not count against the provider's health.
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("cut short")}
	provider := s.provider(Options{})

	stream, err := provider.Create(context.Background(), hello())
	s.Require().NoError(err)
	s.Require().True(stream.Next())
	s.Require().True(stream.Next())
	s.Require().NoError(stream.Close())

	s.Equal(llm.StatusCancelled, drain(stream).Status)
	s.NoError(stream.Err(), "the caller stopped it, so the provider did not fail")
	close(s.hold)
}

func (s *OpenAICompatSuite) TestClosingOneStreamLeavesTheOtherAlone() {
	// A caller delegating background work runs several responses at once, and a premise
	// going stale must abandon that work alone rather than everything in flight.
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("working"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	keep, err := provider.Create(context.Background(), llm.ResponseParams{
		ID: "keep", Input: hello().Input,
	})
	s.Require().NoError(err)
	drop, err := provider.Create(context.Background(), llm.ResponseParams{
		ID: "drop", Input: hello().Input,
	})
	s.Require().NoError(err)

	s.Require().NoError(drop.Close())
	dropped := drain(drop)
	s.Equal("drop", dropped.ID)
	s.Equal(llm.StatusCancelled, dropped.Status)

	close(s.hold)
	kept := drain(keep)
	s.Equal("keep", kept.ID)
	s.Equal(llm.StatusCompleted, kept.Status)
}

func (s *OpenAICompatSuite) TestClosingAStreamTwiceIsSafe() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	stream, err := provider.Create(context.Background(), hello())
	s.Require().NoError(err)

	s.NoError(stream.Close())
	s.NoError(stream.Close())
}

func (s *OpenAICompatSuite) TestAFailedRequestStillSettlesTheResponse() {
	// A caller waiting for a response to settle must never be left hanging, or a turn
	// never ends and the conversation stops.
	s.status = http.StatusUnauthorized
	provider := s.provider(Options{})

	response, events := s.ask(provider, hello())

	s.Empty(response.OutputText)
	s.Equal(llm.StatusFailed, response.Status)

	var failures []llm.ResponseFailed
	for _, event := range events {
		if failure, ok := event.(llm.ResponseFailed); ok {
			failures = append(failures, failure)
		}
	}
	s.Require().NotEmpty(failures, "the failure has to be reported so it can be recorded")
	s.Equal("stream", failures[0].Context)
}

func (s *OpenAICompatSuite) TestARequestWithNoInputIsRefused() {
	provider := s.provider(Options{})

	_, err := provider.Create(context.Background(), llm.ResponseParams{})

	s.ErrorContains(err, "at least one input message")
}

func (s *OpenAICompatSuite) TestCloseIsSafeToRepeat() {
	provider := s.provider(Options{})

	s.NoError(provider.Close())
	s.NoError(provider.Close())
}

func (s *OpenAICompatSuite) TestStatsModelDefaultsToTheUpstreamModel() {
	plain := s.provider(Options{Model: "gpt-4o-mini"})
	s.Equal("gpt-4o-mini", plain.Model())

	qualified := s.provider(Options{Model: "deepseek-ai/DeepSeek-V4-Flash-0731", StatsModel: "DeepSeek-V4-Flash-0731"})
	s.Equal("DeepSeek-V4-Flash-0731", qualified.Model(),
		"stats use the routing name, not the provider's qualified id")
	s.Equal("deepseek-ai/DeepSeek-V4-Flash-0731", s.modelSentBy(qualified))
}

func (s *OpenAICompatSuite) TestClientReachesTheUnderlyingSDK() {
	provider := s.provider(Options{})

	s.NotNil(provider.Client(), "anything this package does not standardise is still reachable")
}

func (s *OpenAICompatSuite) TestToolsAreOnlySentWhenOffered() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, hello())
	s.NotContains(s.requests[0], "tools", "a model offered a toolbox eventually opens it")

	s.ask(provider, llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
		Tools: []llm.Tool{{
			Name:        "transfer",
			Description: "hand the caller to a human",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"to": map[string]any{"type": "string"}},
				"required":   []any{"to"},
			},
		}},
	})

	tools, ok := s.requests[1]["tools"].([]any)
	s.Require().True(ok)
	s.Require().Len(tools, 1)

	declared, ok := tools[0].(map[string]any)
	s.Require().True(ok)
	s.Equal("function", declared["type"])

	function, ok := declared["function"].(map[string]any)
	s.Require().True(ok)
	s.Equal("transfer", function["name"])
	s.Equal("hand the caller to a human", function["description"])

	parameters, ok := function["parameters"].(map[string]any)
	s.Require().True(ok, "without the schema the model cannot fill the arguments in")
	s.Equal("object", parameters["type"])
}

func (s *OpenAICompatSuite) TestASignedToolCallIsHandedBackSigned() {
	// Gemini signs the calls it asks for and refuses a conversation that replays one
	// unsigned, so dropping the signature costs the caller the answer to every question
	// a tool was reached for: the result comes back and the turn that would speak it is
	// rejected.
	s.frames = []string{
		signedToolFrame(0, "call-1", "get_weather", `{"location":"Boulder, CO"}`, "sig-abc"),
		usageFrame(20, 0, 9, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())
	s.Require().Len(response.ToolCalls, 1)
	s.Equal("sig-abc", response.ToolCalls[0].Signature)

	s.frames = []string{textFrame("It is sunny."), usageFrame(30, 0, 4, 0, "stop")}
	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.User, Content: "how is the weather?"},
		{Role: llm.Assistant, ToolCalls: response.ToolCalls},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: "20 and sunny"},
	}})

	replayed := s.sentMessages(1)
	s.Require().Len(replayed, 3)

	calls, ok := replayed[1]["tool_calls"].([]any)
	s.Require().True(ok)
	s.Require().Len(calls, 1)

	call, ok := calls[0].(map[string]any)
	s.Require().True(ok)
	content, ok := call["extra_content"].(map[string]any)
	s.Require().True(ok, "the call went back unsigned and the provider would refuse it")

	google, ok := content["google"].(map[string]any)
	s.Require().True(ok)
	s.Equal("sig-abc", google["thought_signature"])
}

func (s *OpenAICompatSuite) TestAnUnsignedToolCallIsReplayedWithoutAnEmptySignature() {
	// Most providers do not sign, and sending them an empty signature would be a field
	// they never asked for on every tool call a conversation replays.
	s.frames = []string{
		toolFrame(0, "call-1", "transfer", `{"to":"+15551234567"}`),
		usageFrame(20, 0, 9, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())
	s.Empty(response.ToolCalls[0].Signature)

	s.frames = []string{textFrame("done"), usageFrame(30, 0, 4, 0, "stop")}
	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.Assistant, ToolCalls: response.ToolCalls},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: "transferred"},
	}})

	calls, ok := s.sentMessages(1)[0]["tool_calls"].([]any)
	s.Require().True(ok)
	call, ok := calls[0].(map[string]any)
	s.Require().True(ok)
	s.NotContains(call, "extra_content")
}

func (s *OpenAICompatSuite) TestStreamedFragmentsAssembleIntoOneToolCall() {
	// Arguments arrive a few characters at a time, so no single fragment is parseable and
	// the caller has to be handed the finished call rather than the pieces.
	s.frames = []string{
		toolFrame(0, "call-1", "transfer", ""),
		toolFrame(0, "", "", `{"to":"+1555`),
		toolFrame(0, "", "", `1234567"}`),
		usageFrame(20, 0, 9, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.Require().Len(response.ToolCalls, 1)
	s.Equal("call-1", response.ToolCalls[0].ID)
	s.Equal("transfer", response.ToolCalls[0].Name)
	s.Equal(`{"to":"+15551234567"}`, response.ToolCalls[0].Arguments)
	s.Equal(llm.StatusCompleted, response.Status)
}

func (s *OpenAICompatSuite) TestSeveralToolCallsKeepTheirOwnArguments() {
	// A model may ask for two things at once, and the provider interleaves their
	// fragments, so each one is assembled under the index it was streamed on.
	s.frames = []string{
		toolFrame(0, "call-1", "press", ""),
		toolFrame(1, "call-2", "transfer", ""),
		toolFrame(0, "", "", `{"digits":"1"}`),
		toolFrame(1, "", "", `{"to":"+15550001111"}`),
		usageFrame(20, 0, 12, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.Require().Len(response.ToolCalls, 2)
	s.Equal("press", response.ToolCalls[0].Name)
	s.Equal(`{"digits":"1"}`, response.ToolCalls[0].Arguments)
	s.Equal("transfer", response.ToolCalls[1].Name)
	s.Equal(`{"to":"+15550001111"}`, response.ToolCalls[1].Arguments)
}

func (s *OpenAICompatSuite) TestSpeechAndAToolCallArriveTogether() {
	// A model told to keep the caller company while it acts says something and calls the
	// tool in the same reply, and both halves have to survive.
	s.frames = []string{
		textFrame("One moment, putting you through."),
		toolFrame(0, "call-1", "transfer", `{"to":"+15550001111"}`),
		usageFrame(20, 0, 14, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.Equal("One moment, putting you through.", response.OutputText)
	s.Require().Len(response.ToolCalls, 1)
	s.Equal("transfer", response.ToolCalls[0].Name)
}

func (s *OpenAICompatSuite) TestAToolCallWithNoIDStillGetsOne() {
	// A self-hosted parser may name a call without identifying it, and the result sent
	// back has to say which call it answers.
	s.frames = []string{
		toolFrame(0, "", "press", `{"digits":"2"}`),
		usageFrame(20, 0, 6, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, llm.ResponseParams{
		ID:    "turn-1",
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
	})

	s.Require().Len(response.ToolCalls, 1)
	s.NotEmpty(response.ToolCalls[0].ID)
	s.Equal("press", response.ToolCalls[0].Name)
}

func (s *OpenAICompatSuite) TestAToolResultIsSentWithTheCallItAnswers() {
	// The provider matches each result against a call it made, so replaying the turn
	// without the calls on it is a conversation it refuses.
	s.frames = []string{textFrame("Done."), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.User, Content: "put me through"},
		{
			Role:      llm.Assistant,
			Content:   "One moment.",
			ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "transfer", Arguments: `{"to":"+1555"}`}},
		},
		{Role: llm.ToolResult, Content: "transferred", ToolCallID: "call-1"},
	}})

	messages := s.sentMessages(0)
	s.Require().Len(messages, 3)

	s.Equal("assistant", messages[1]["role"])
	s.Equal("One moment.", messages[1]["content"])
	calls, ok := messages[1]["tool_calls"].([]any)
	s.Require().True(ok, "the turn has to carry the call the result answers")
	s.Require().Len(calls, 1)
	call, ok := calls[0].(map[string]any)
	s.Require().True(ok)
	s.Equal("call-1", call["id"])

	s.Equal("tool", messages[2]["role"])
	s.Equal("call-1", messages[2]["tool_call_id"])
	s.Equal("transferred", messages[2]["content"])
}

// sentMessages returns the messages from the nth recorded request.
func (s *OpenAICompatSuite) sentMessages(index int) []map[string]any {
	s.Require().Greater(len(s.requests), index)

	raw, ok := s.requests[index]["messages"].([]any)
	s.Require().True(ok)

	messages := make([]map[string]any, 0, len(raw))
	for _, entry := range raw {
		message, ok := entry.(map[string]any)
		s.Require().True(ok)
		messages = append(messages, message)
	}
	return messages
}

// modelSentBy asks the provider for a completion and reports the model id it sent.
func (s *OpenAICompatSuite) modelSentBy(provider *LLM) string {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	s.requests = nil

	s.ask(provider, hello())
	model, ok := s.requests[0]["model"].(string)
	s.Require().True(ok)
	return model
}
