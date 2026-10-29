package openaicompat

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
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
	// requests records the decoded bodies the provider sent, and headers their headers.
	requests []map[string]any
	headers  []http.Header
	// hold blocks the response until closed, for tests about cancellation.
	hold chan struct{}
	// status replaces 200 when non-zero.
	status int
	// errorBody replaces the error a non-zero status answers with.
	errorBody string
	// keys records the API key each request was sent with, and refusedKey is answered
	// with 429.
	keys       []string
	refusedKey string
}

func TestOpenAICompatSuite(t *testing.T) {
	suite.Run(t, new(OpenAICompatSuite))
}

func (s *OpenAICompatSuite) SetupTest() {
	s.frames = nil
	s.requests = nil
	s.headers = nil
	s.hold = nil
	s.status = 0
	s.errorBody = ""
	s.keys = nil
	s.refusedKey = ""

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		s.Require().NoError(err)

		var body map[string]any
		s.Require().NoError(json.Unmarshal(raw, &body))
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		s.mu.Lock()
		s.requests = append(s.requests, body)
		s.headers = append(s.headers, r.Header.Clone())
		s.keys = append(s.keys, key)
		s.mu.Unlock()

		if key == s.refusedKey {
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}

		if s.status != 0 {
			w.WriteHeader(s.status)
			if s.errorBody != "" {
				fmt.Fprint(w, s.errorBody)
				return
			}
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

// failedWith is the error a stream ends with when the provider sends frame after some text.
func (s *OpenAICompatSuite) failedWith(frame string) error {
	s.frames = []string{textFrame("Hel"), frame}
	stream, err := s.provider(Options{}).Create(context.Background(), hello())
	s.Require().NoError(err)
	s.Equal(llm.StatusFailed, drain(stream).Status)
	return stream.Err()
}

func (s *OpenAICompatSuite) TestAnErrorSentMidStreamWithAServerStatusIsTransient() {
	s.True(llm.Transient(s.failedWith(`{"error":{"code":503,"message":"try later"}}`)))
	s.True(llm.Transient(s.failedWith(`{"error":{"type":"server_error","message":"try later"}}`)))
}

func (s *OpenAICompatSuite) TestAnErrorSentMidStreamAboutTheRequestIsNotTransient() {
	for _, frame := range []string{
		`{"error":{"code":400,"message":"not this"}}`,
		`{"error":{"code":1301,"message":"a code of the provider's own"}}`,
		`{"error":{"type":"invalid_request_error","message":"not this"}}`,
	} {
		failure, ok := errors.AsType[*llm.ProviderFailure](s.failedWith(frame))
		s.Require().True(ok, frame)
		s.False(failure.Transient, frame)
	}
}

func (s *OpenAICompatSuite) TestAnErrorSentMidStreamWithNothingNamedIsLeftUnclassified() {
	err := s.failedWith(`{"error":{"message":"something"}}`)
	s.ErrorContains(err, "something")
	_, ok := errors.AsType[*llm.ProviderFailure](err)
	s.False(ok)
	s.False(llm.Transient(err))
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

func (s *OpenAICompatSuite) TestARateLimitedKeyHandsTheTurnToTheNextKeyInThePool() {
	s.frames = []string{textFrame("hello"), usageFrame(3, 0, 1, 0, "stop")}
	refused, spare := "compat-"+rand.Text(), "compat-"+rand.Text()
	s.refusedKey = refused
	provider, err := New(Options{
		Provider: "test",
		Model:    "test-model",
		APIKey:   refused + ", " + spare,
		BaseURL:  s.server.URL,
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })

	first, _ := s.ask(provider, hello())
	second, _ := s.ask(provider, hello())

	s.Equal(llm.StatusCompleted, first.Status)
	s.Equal(llm.StatusCompleted, second.Status)
	s.Equal([]string{refused, spare, spare}, s.keys,
		"the refused key rests rather than being asked again on the next turn")
	s.Equal(s.requests[0], s.requests[1], "the refused request is sent again as it was")
}

func (s *OpenAICompatSuite) TestAnErrorSentAsAListStillSaysWhy() {
	s.status = http.StatusBadRequest
	s.errorBody = `[{"error":{"code":400,"message":"Function call is missing a thought_signature","status":"INVALID_ARGUMENT"}}]`
	provider := s.provider(Options{})

	_, err := provider.Create(s.T().Context(), hello())

	s.Require().Error(err)
	s.Contains(err.Error(), "Function call is missing a thought_signature")
	var apiError *openai.Error
	s.Require().ErrorAs(err, &apiError, "the router still tells a refused request from an outage")
	s.Equal(http.StatusBadRequest, apiError.StatusCode)
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

// TestReasoningOutsideCompletionIsCounted is xAI's "think" frame: its thinking is in the total
// and in reasoning_tokens, not in completion_tokens, and its own cost comes beside the counts.
func (s *OpenAICompatSuite) TestReasoningOutsideCompletionIsCounted() {
	s.frames = []string{textFrame("4"), frame([]any{}, map[string]any{
		"prompt_tokens":             1_301,
		"completion_tokens":         21,
		"total_tokens":              2_747,
		"prompt_tokens_details":     map[string]any{"cached_tokens": 1_152},
		"completion_tokens_details": map[string]any{"reasoning_tokens": 1_425},
		"cost_in_usd_ticks":         95_500_000,
	})}
	provider := s.provider(Options{Provider: "xai", Model: "grok-4.7"})

	response, _ := s.ask(provider, hello())

	s.EqualValues(1_446, response.Usage.OutputTokens, "the answer and the thinking, once")
	s.EqualValues(1_425, response.Usage.OutputTokensDetails.ReasoningTokens)
	s.EqualValues(9_550, response.Usage.VendorCostMicros)
	list, err := pricing.NewList("vendor", prices.Vendor().Entries, nil)
	s.Require().NoError(err)
	quote := list.Quote(pricing.Item{Modality: "llm", Provider: "xai", Model: "grok-4.7", Region: pricing.RegionUS}, response.Used())
	s.EqualValues(9_550, quote.BaseCostMicros, "priced at the vendor's rates, it is what xAI charged")
}

func (s *OpenAICompatSuite) TestThinkingCountedOnlyInTheTotalIsOutput() {
	// Gemini's compatible endpoint reports no reasoning_tokens: its thinking shows only as the
	// total running past the prompt and the completion.
	s.frames = []string{textFrame("Paris"), frame([]any{}, map[string]any{
		"prompt_tokens":     10,
		"completion_tokens": 11,
		"total_tokens":      123,
	})}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.EqualValues(113, response.Usage.OutputTokens)
	s.EqualValues(102, response.Usage.OutputTokensDetails.ReasoningTokens)
	s.EqualValues(123, response.Usage.TotalTokens, "the vendor's own total is kept")
	s.Zero(response.Usage.VendorCostMicros, "a vendor that sends no cost has none")
}

func (s *OpenAICompatSuite) TestRejectedPredictionsAreNotBilled() {
	// Baseten reports the speculative tokens it threw away, and bills completion_tokens alone.
	s.frames = []string{textFrame("hi"), frame([]any{}, map[string]any{
		"prompt_tokens":             20,
		"completion_tokens":         48,
		"total_tokens":              68,
		"completion_tokens_details": map[string]any{"rejected_prediction_tokens": 26},
	})}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.EqualValues(48, response.Usage.OutputTokens)
	s.Zero(response.Usage.OutputTokensDetails.ReasoningTokens)
}

func (s *OpenAICompatSuite) TestACallersDeadlineCutsTheResponseShort() {
	s.frames = []string{textFrame("Hello")}
	s.hold = make(chan struct{})
	provider := s.provider(Options{})
	ctx, cancel := context.WithTimeout(s.T().Context(), 300*time.Millisecond)
	defer cancel()

	stream, err := provider.Create(ctx, hello())
	s.Require().NoError(err)
	for stream.Next() {
	}
	response := stream.Response()

	s.True(response.TimedOut)
	s.Equal(llm.StatusFailed, response.Status, "whoever waited still sees it fail")
	s.Equal("Hello", response.OutputText)
	s.ErrorIs(stream.Err(), context.DeadlineExceeded)
}

func (s *OpenAICompatSuite) TestTheProvidersOwnLimitCutsTheResponseShort() {
	s.frames = []string{textFrame("Hello")}
	s.hold = make(chan struct{})
	provider := s.provider(Options{Timeout: 300 * time.Millisecond})

	response, _ := s.ask(provider, hello())

	s.True(response.TimedOut)
	s.Equal("Hello", response.OutputText)
}

func (s *OpenAICompatSuite) TestAStreamTheVendorBrokeIsNotCutShort() {
	s.frames = []string{textFrame("Hello"), "{not json"}
	provider := s.provider(Options{})

	stream, err := provider.Create(s.T().Context(), hello())
	s.Require().NoError(err)
	for stream.Next() {
	}

	s.False(stream.Response().TimedOut)
	s.Equal(llm.StatusFailed, stream.Response().Status)
	s.Error(stream.Err())
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
	s.Equal("The user said hi, so", response.ReasoningText)
}

func (s *OpenAICompatSuite) TestThinkingStreamedAsReasoningIsThinkingToo() {
	s.frames = []string{
		frame([]any{map[string]any{"index": 0, "delta": map[string]any{"reasoning": "Greet them back."}}}, nil),
		textFrame("Hi! How can I help?"),
		usageFrame(11, 0, 20, 16, "stop"),
	}
	provider := s.provider(Options{Capabilities: llm.Capabilities{StreamsReasoning: true}})

	response, _ := s.ask(provider, hello())

	s.Equal("Hi! How can I help?", response.OutputText)
	s.Equal("Greet them back.", response.ReasoningText)
}

// reasoningOf joins the reasoning a response streamed.
func reasoningOf(events []llm.Event) string {
	var text string
	for _, event := range events {
		if delta, ok := event.(llm.ReasoningTextDelta); ok {
			text += delta.Delta
		}
	}
	return text
}

func (s *OpenAICompatSuite) TestAThoughtChannelIsNeverSpoken() {
	// What Gemma 4 has sent: the channel written after the filler, just before the
	// tool call, with its markers split over several chunks.
	s.frames = []string{
		textFrame("Just a moment, checking. \n\n<|chan"),
		textFrame("nel>thought\nparty of two<chan"),
		textFrame("nel|>"),
		toolFrame(0, "call-1", "find_room", `{"guests":2}`),
		usageFrame(20, 0, 14, 0, "tool_calls"),
	}
	provider := s.provider(Options{ThoughtChannel: true})

	response, events := s.ask(provider, hello())

	s.Equal("Just a moment, checking. \n\n", response.OutputText)
	s.Equal("party of two", reasoningOf(events), "the channel is still there for a caller that wants it")
	s.Require().Len(response.ToolCalls, 1)
}

func (s *OpenAICompatSuite) TestABareChannelNameOpeningAReplyIsNeverSpoken() {
	// vLLM can take the markers and leave the channel's name at the start of the reply.
	s.frames = []string{textFrame("thou"), textFrame("ght\n You're all set!"), usageFrame(20, 0, 8, 0, "stop")}
	provider := s.provider(Options{ThoughtChannel: true})

	response, _ := s.ask(provider, hello())

	s.Equal(" You're all set!", response.OutputText)
}

func (s *OpenAICompatSuite) TestABareChannelNameAfterWhitespaceIsNeverSpoken() {
	// The channel's name can arrive with whitespace or a colon around it rather than first
	// and bare, and it is still the channel's name, not the start of the answer.
	for _, frames := range [][]string{
		{"\nthought\n", " You're all set!"},
		{" thou", "ght \r\n You're all set!"},
		{"thought\r", "\n You're all set!"},
		{"thought:", " You're all set!"},
		{"thought", ":\n You're all set!"},
	} {
		s.Run(frames[0], func() {
			s.frames = []string{textFrame(frames[0]), textFrame(frames[1]), usageFrame(20, 0, 8, 0, "stop")}
			provider := s.provider(Options{ThoughtChannel: true})

			response, _ := s.ask(provider, hello())

			s.Equal(" You're all set!", response.OutputText)
		})
	}
}

func (s *OpenAICompatSuite) TestAReplyThatIsOnlyAChannelNameSaysNothing() {
	s.frames = []string{textFrame("\nthought"), usageFrame(20, 0, 8, 0, "stop")}
	provider := s.provider(Options{ThoughtChannel: true})

	response, _ := s.ask(provider, hello())

	s.Empty(response.OutputText)
}

func (s *OpenAICompatSuite) TestAReplyThatMerelyLooksLikeAChannelIsSpokenWhole() {
	s.frames = []string{
		textFrame("thought"),
		textFrame(" about it: a <|chan"),
		usageFrame(20, 0, 8, 0, "stop"),
	}
	provider := s.provider(Options{ThoughtChannel: true})

	response, _ := s.ask(provider, hello())

	s.Equal("thought about it: a <|chan", response.OutputText, "a marker that never finished is text")
}

func (s *OpenAICompatSuite) TestAThoughtChannelIsLeftAloneForOtherModels() {
	s.frames = []string{textFrame("<|channel>thought\n<channel|>Hi"), usageFrame(20, 0, 8, 0, "stop")}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.Equal("<|channel>thought\n<channel|>Hi", response.OutputText)
}

// withTools is a request that offers the restaurant's tools.
func withTools() llm.ResponseParams {
	params := hello()
	params.Tools = []llm.Tool{{Name: "find_room"}, {Name: "book_room"}}
	return params
}

func (s *OpenAICompatSuite) TestAToolCallWrittenAsTextIsMadeNotSpoken() {
	// What Gemma 4 has sent once the opening marker went missing, split mid-marker.
	s.frames = []string{
		textFrame("call:find_r"),
		textFrame("oom{guests:4,breakfast:true,time:<|\"|"),
		textFrame(">18:45<|\"|>}<tool_"),
		textFrame("call|>"),
		usageFrame(20, 0, 30, 0, "stop"),
	}
	provider := s.provider(Options{ToolCallText: true})

	response, _ := s.ask(provider, withTools())

	s.Empty(response.OutputText)
	s.Require().Len(response.ToolCalls, 1)
	s.Equal("find_room", response.ToolCalls[0].Name)
	s.JSONEq(`{"guests":4,"breakfast":true,"time":"18:45"}`, response.ToolCalls[0].Arguments)
	s.NotEmpty(response.ToolCalls[0].ID)
}

func (s *OpenAICompatSuite) TestAMarkedToolCallAfterFillerKeepsTheFiller() {
	s.frames = []string{
		textFrame("One moment, checking. <|tool_call>call:book_room{name:<|\"|>Quillon<|\"|>,"),
		textFrame("guests:6,allergen:<|\"|>sesame<|\"|>,notes:{late_checkout:false,tags:[1,<|\"|>a, b<|\"|>]}}<tool_call|>"),
		toolFrame(0, "call-1", "find_room", `{"guests":6}`),
		usageFrame(20, 0, 30, 0, "tool_calls"),
	}
	provider := s.provider(Options{ToolCallText: true})

	response, _ := s.ask(provider, withTools())

	s.Equal("One moment, checking. ", response.OutputText)
	s.Require().Len(response.ToolCalls, 2, "the call written as text and the real one are both made")
	names := []string{response.ToolCalls[0].Name, response.ToolCalls[1].Name}
	s.ElementsMatch([]string{"book_room", "find_room"}, names)
	for _, call := range response.ToolCalls {
		if call.Name == "book_room" {
			s.JSONEq(`{"name":"Quillon","guests":6,"allergen":"sesame","notes":{"late_checkout":false,"tags":[1,"a, b"]}}`, call.Arguments)
		}
	}
}

func (s *OpenAICompatSuite) TestAToolCallWrittenInAnotherSyntaxIsMadeNotSpoken() {
	// What Gemma 4 has sent and callers heard read out.
	cases := map[string]struct {
		frames    []string
		spoken    string
		name      string
		arguments string
	}{
		"call_tool prefix with double quotes": {
			frames:    []string{"Let me open a ticket... call_tool: book_room{name: \"Qui", "llon, party of {6}\", allergen: \"say \\\"none\\\"\"}"},
			spoken:    "Let me open a ticket... ",
			name:      "book_room",
			arguments: `{"name":"Quillon, party of {6}","allergen":"say \"none\""}`,
		},
		"call tool prefix with no arguments": {
			frames:    []string{"One moment. call tool", ": find_room{}"},
			spoken:    "One moment. ",
			name:      "find_room",
			arguments: `{}`,
		},
		"python call in backticks": {
			frames:    []string{"Checking, `call:find_room(guests=4, breakfast=true, ", `time="18:45")`, "`."},
			spoken:    "Checking, ``.",
			name:      "find_room",
			arguments: `{"guests":4,"breakfast":true,"time":"18:45"}`,
		},
		"python call with single quotes": {
			frames:    []string{"call:book_room(name='O\\'Br", "ien (party)', allergen='C:\\temp')"},
			spoken:    "",
			name:      "book_room",
			arguments: `{"name":"O'Brien (party)","allergen":"C:\\temp"}`,
		},
		"python call ending on a bare value": {
			frames:    []string{"call:find_room(time=\"18:45\", guests=4)"},
			spoken:    "",
			name:      "find_room",
			arguments: `{"guests":4,"time":"18:45"}`,
		},
		"bare python call with no prefix": {
			frames:    []string{"One moment. Checking. find_r", "oom(guests=4, time=\"18:45\")"},
			spoken:    "One moment. Checking. ",
			name:      "find_room",
			arguments: `{"guests":4,"time":"18:45"}`,
		},
		"call followed by a space": {
			frames:    []string{"One moment, checking... call find_r", "oom(guests=4, time='18:45')"},
			spoken:    "One moment, checking... ",
			name:      "find_room",
			arguments: `{"guests":4,"time":"18:45"}`,
		},
		"bare brace call with no prefix": {
			frames:    []string{"Booking now. book_room{name: \"Quillon\", guests: 6}"},
			spoken:    "Booking now. ",
			name:      "book_room",
			arguments: `{"name":"Quillon","guests":6}`,
		},
	}
	for name, tc := range cases {
		s.Run(name, func() {
			s.frames = nil
			for _, frame := range tc.frames {
				s.frames = append(s.frames, textFrame(frame))
			}
			s.frames = append(s.frames, usageFrame(20, 0, 30, 0, "stop"))
			provider := s.provider(Options{ToolCallText: true})

			response, _ := s.ask(provider, withTools())

			s.Equal(tc.spoken, response.OutputText)
			s.Require().Len(response.ToolCalls, 1)
			s.Equal(tc.name, response.ToolCalls[0].Name)
			s.JSONEq(tc.arguments, response.ToolCalls[0].Arguments)
		})
	}
}

func (s *OpenAICompatSuite) TestAToolNamedInASentenceIsSpokenNotMade() {
	params := hello()
	params.Tools = []llm.Tool{{Name: "send_note"}, {Name: "check_stock"}}
	cases := map[string]struct {
		frames []string
		spoken string
		made   []string
	}{
		"bare call followed by words": {
			frames: []string{"You can use send_note(to=\"ana\", text=\"hi\")", " to message her."},
			spoken: "You can use send_note(to=\"ana\", text=\"hi\") to message her.",
		},
		"call followed by a space, then words": {
			frames: []string{"I would call send_note{to: \"ana\"} if you want."},
			spoken: "I would call send_note{to: \"ana\"} if you want.",
		},
		"bare call followed by the start of another name": {
			frames: []string{"Try send_note(to=\"ana\") or check"},
			spoken: "Try send_note(to=\"ana\") or check",
		},
		"bare call that ends the answer": {
			frames: []string{"Sending it now. send_note(to=", "\"ana\")"},
			spoken: "Sending it now. ",
			made:   []string{"send_note"},
		},
		"bare call followed by its closing marker": {
			frames: []string{"Sending it now. send_note(to=\"ana\") <tool_", "call|>"},
			spoken: "Sending it now. ",
			made:   []string{"send_note"},
		},
		"bare calls one after another": {
			frames: []string{"One moment. check_stock{} ", "send_note{to: \"ana\"}"},
			spoken: "One moment.  ",
			made:   []string{"check_stock", "send_note"},
		},
		"marked call followed by words": {
			frames: []string{"<|tool_call>call:send_note{to:<|\"|>ana<|\"|>}<tool_call|> Sent."},
			spoken: " Sent.",
			made:   []string{"send_note"},
		},
		"call: prefix followed by words": {
			frames: []string{"call:send_note{to: \"ana\"} Sent."},
			spoken: " Sent.",
			made:   []string{"send_note"},
		},
	}
	for name, tc := range cases {
		s.Run(name, func() {
			s.frames = nil
			for _, frame := range tc.frames {
				s.frames = append(s.frames, textFrame(frame))
			}
			s.frames = append(s.frames, usageFrame(20, 0, 30, 0, "stop"))
			provider := s.provider(Options{ToolCallText: true})

			response, _ := s.ask(provider, params)

			s.Equal(tc.spoken, response.OutputText)
			var made []string
			for _, call := range response.ToolCalls {
				made = append(made, call.Name)
			}
			s.Equal(tc.made, made)
		})
	}
}

func (s *OpenAICompatSuite) TestABraceFormToolCallPreservesSingleQuotedIdentityArguments() {
	// Brace-form identity values in single quotes should decode as exact strings, without
	// quote marks, before they reach the verification tool.
	s.frames = []string{
		textFrame("call:confirm_caller{name:'Ines Moreau',"),
		textFrame("dob:'1979-11-23',member_id:'KTR583027'}"),
		usageFrame(20, 0, 30, 0, "tool_calls"),
	}
	provider := s.provider(Options{ToolCallText: true})
	params := hello()
	params.Tools = []llm.Tool{{Name: "confirm_caller"}}

	response, _ := s.ask(provider, params)

	s.Empty(response.OutputText)
	s.Require().Len(response.ToolCalls, 1)
	s.Equal("confirm_caller", response.ToolCalls[0].Name)
	s.JSONEq(`{"name":"Ines Moreau","dob":"1979-11-23","member_id":"KTR583027"}`, response.ToolCalls[0].Arguments)
	var identity struct {
		Name     string `json:"name"`
		DOB      string `json:"dob"`
		MemberID string `json:"member_id"`
	}
	s.Require().NoError(json.Unmarshal([]byte(response.ToolCalls[0].Arguments), &identity))
	s.Equal("Ines Moreau", identity.Name)
	s.Equal("1979-11-23", identity.DOB)
	s.Equal("KTR583027", identity.MemberID)
}

func (s *OpenAICompatSuite) TestABraceFormToolCallKeepsSingleQuotedValuesTogetherWhileStreaming() {
	s.frames = []string{
		textFrame("call:confirm_caller{'name':'Ines Moreau','dob':'1979-11-23',"),
		textFrame("'member_id':'KTR583027','nickname':'O\\"),
		textFrame("'Brien','note':'comma, brace } and {','bare':O'Brien}"),
		usageFrame(20, 0, 30, 0, "tool_calls"),
	}
	provider := s.provider(Options{ToolCallText: true})
	params := hello()
	params.Tools = []llm.Tool{{Name: "confirm_caller"}}

	response, _ := s.ask(provider, params)

	s.Empty(response.OutputText)
	s.Require().Len(response.ToolCalls, 1)
	s.Equal("confirm_caller", response.ToolCalls[0].Name)
	s.JSONEq(`{"name":"Ines Moreau","dob":"1979-11-23","member_id":"KTR583027","nickname":"O'Brien","note":"comma, brace } and {","bare":"O'Brien"}`, response.ToolCalls[0].Arguments)
}

func (s *OpenAICompatSuite) TestNativeJSONToolArgumentsKeepApostrophesInsideDoubleQuotedValues() {
	s.frames = []string{
		toolFrame(0, "call-1", "confirm_caller", `{"name":"O'Brien","alias":"'Ines Moreau'"}`),
		usageFrame(20, 0, 30, 0, "tool_calls"),
	}
	provider := s.provider(Options{ToolCallText: true})
	params := hello()
	params.Tools = []llm.Tool{{Name: "confirm_caller"}}

	response, _ := s.ask(provider, params)

	s.Require().Len(response.ToolCalls, 1)
	s.Equal("confirm_caller", response.ToolCalls[0].Name)
	s.JSONEq(`{"name":"O'Brien","alias":"'Ines Moreau'"}`, response.ToolCalls[0].Arguments)
}

func (s *OpenAICompatSuite) TestAReplyThatMerelyLooksLikeAToolCallIsSpokenWhole() {
	s.frames = []string{
		textFrame("Please call: me back, or call:"),
		textFrame("check_weather{city:Austin} and call:check"),
		usageFrame(20, 0, 8, 0, "stop"),
	}
	provider := s.provider(Options{ToolCallText: true})

	response, _ := s.ask(provider, withTools())

	s.Equal("Please call: me back, or call:check_weather{city:Austin} and call:check", response.OutputText,
		"a name that is not offered is not a call")
	s.Empty(response.ToolCalls)
}

func (s *OpenAICompatSuite) TestAnOfferedNameWithoutItsArgumentsIsSpoken() {
	s.frames = []string{
		textFrame("Let me check"),
		textFrame(" that. I can find_room for you, or prefind_room(x=1). check"),
		usageFrame(20, 0, 8, 0, "stop"),
	}
	provider := s.provider(Options{ToolCallText: true})

	response, _ := s.ask(provider, withTools())

	s.Equal("Let me check that. I can find_room for you, or prefind_room(x=1). check", response.OutputText,
		"a bare name is a call only when its arguments follow it and it starts a word")
	s.Empty(response.ToolCalls)
}

func (s *OpenAICompatSuite) TestAStrayControlTokenIsNeverSpoken() {
	// Gemma 4's control tokens can arrive as text beside a real call, and none of them is
	// part of the answer.
	s.frames = []string{
		textFrame(`<|"|>`),
		toolFrame(0, "call-1", "find_room", `{"guests":4}`),
		textFrame("\ufffd\nOne <tool_call|>moment."),
		usageFrame(20, 0, 8, 0, "tool_calls"),
	}

	for name, tools := range map[string]llm.ResponseParams{"with tools": withTools(), "without": hello()} {
		provider := s.provider(Options{ToolCallText: true})
		response, _ := s.ask(provider, tools)
		s.Equal("\nOne moment.", response.OutputText, name)
	}
}

func (s *OpenAICompatSuite) TestAStrayChannelMarkerIsNeverSpoken() {
	// A channel's closing marker with no opening one ahead of it is still a marker.
	s.frames = []string{textFrame("<channel|> I found it."), usageFrame(20, 0, 8, 0, "stop")}
	provider := s.provider(Options{ThoughtChannel: true})

	response, _ := s.ask(provider, hello())

	s.Equal(" I found it.", response.OutputText)
}

func (s *OpenAICompatSuite) TestAToolCallThatNeverClosesIsNotSpoken() {
	s.frames = []string{textFrame("Sure. call:book_room{name:<|\"|>Al"), usageFrame(20, 0, 8, 0, "length")}
	provider := s.provider(Options{ToolCallText: true})

	response, events := s.ask(provider, withTools())

	s.Equal("Sure. ", response.OutputText)
	s.Empty(response.ToolCalls)
	s.Contains(reasoningOf(events), "book_room", "it is kept where a caller that wants it can see it")
}

func (s *OpenAICompatSuite) TestAToolCallWrittenAsTextIsLeftAloneForOtherModels() {
	s.frames = []string{textFrame("call:find_room{guests:4}"), usageFrame(20, 0, 8, 0, "stop")}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, withTools())

	s.Equal("call:find_room{guests:4}", response.OutputText)
	s.Empty(response.ToolCalls)
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

func (s *OpenAICompatSuite) TestAPrefillOpensTheReplyForTheModelToContinue() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{Prefill: func(llm.ResponseParams) string { return "<think>\n</think>\n" }})

	s.ask(provider, hello())

	messages, ok := s.requests[0]["messages"].([]any)
	s.Require().True(ok)
	last, ok := messages[len(messages)-1].(map[string]any)
	s.Require().True(ok)
	s.Equal("assistant", last["role"])
	s.Equal("<think>\n</think>\n", last["content"])
	s.Equal(true, s.requests[0]["continue_final_message"], "the model continues the prefill rather than answering after it")
	s.Equal(false, s.requests[0]["add_generation_prompt"])
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

// TestTheCacheKeyIsSentWhereTheProviderWantsIt: chat completions has no field for a cache
// key, so a provider that caches or routes by one says where it goes, a body field or a
// header, and the key is sent there and nowhere else.
func (s *OpenAICompatSuite) TestTheCacheKeyIsSentWhereTheProviderWantsIt() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	request := hello()
	request.PromptCacheKey = "conversation-42"

	inBody := s.provider(Options{CacheKeyField: "prompt_cache_key"})
	s.True(inBody.Capabilities().PromptCacheKey)
	s.ask(inBody, request)
	s.Equal("conversation-42", s.requests[0]["prompt_cache_key"])
	s.Empty(s.headers[0].Get("X-Session-Affinity"))

	inHeader := s.provider(Options{CacheKeyHeader: "x-session-affinity"})
	s.True(inHeader.Capabilities().PromptCacheKey)
	s.ask(inHeader, request)
	s.Equal("conversation-42", s.headers[1].Get("X-Session-Affinity"))
	s.NotContains(s.requests[1], "prompt_cache_key")

	s.ask(inHeader, hello())
	s.Empty(s.headers[2].Values("X-Session-Affinity"), "a request with no key sends no header")
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

// ruling is a schema-bound request like the flow controller's.
func ruling() llm.ResponseParams {
	return llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
		Text: llm.TextParams{Format: llm.FormatJSONSchema, Schema: &llm.JSONSchema{
			Name: "ruling",
			Schema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"floor": map[string]any{"type": "string", "enum": []string{"stop"}}},
				"required":             []string{"floor"},
				"additionalProperties": false,
			},
		}},
	}
}

func (s *OpenAICompatSuite) TestAModelThatTakesASchemaIsHeldToItStrictly() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{Capabilities: llm.Capabilities{JSONSchema: true}})

	s.ask(provider, ruling())

	format, ok := s.requests[0]["response_format"].(map[string]any)
	s.Require().True(ok)
	s.Equal("json_schema", format["type"])
	s.Equal(map[string]any{
		"name":   "ruling",
		"strict": true,
		"schema": map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"floor": map[string]any{"type": "string", "enum": []any{"stop"}}},
			"required":             []any{"floor"},
			"additionalProperties": false,
		},
	}, format["json_schema"])
}

func (s *OpenAICompatSuite) TestAModelThatTakesNoSchemaIsAskedForAJSONObjectAsBefore() {
	// A provider that never claimed schemas is sent what a json_object request has always
	// sent, so it sees nothing new.
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})
	asObject := ruling()
	asObject.Text = llm.TextParams{Format: llm.FormatJSONObject}

	s.ask(provider, asObject)
	s.ask(provider, ruling())

	s.Equal(s.requests[0], s.requests[1])
}

func (s *OpenAICompatSuite) TestClosingAStreamSettlesWhatHadAlreadyArrived() {
	s.hold = make(chan struct{})
	// Usage must be read before cancellation; a frame merely flushed by the server
	// can still be in transit when Close cancels the upstream request.
	s.frames = []string{usageFrame(11, 0, 4, 0, ""), textFrame("I was saying")}
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

func (s *OpenAICompatSuite) TestARejectedHTTPRequestReturnsBeforeStreaming() {
	s.status = http.StatusUnauthorized
	provider := s.provider(Options{})
	stream, err := provider.Create(context.Background(), hello())
	s.Nil(stream)
	s.Require().Error(err)
	s.Contains(err.Error(), "401")
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

func (s *OpenAICompatSuite) TestEarlierThinkingIsSentBackOnlyToAProviderThatReplaysIt() {
	input := []llm.Message{
		{Role: llm.User, Content: "my internet is down"},
		{Role: llm.Assistant, Content: "I can help with that.", Reasoning: "Unverified; ask for the PIN."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "lookup", Arguments: "{}"}}, Reasoning: "Look it up."},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: "found"},
	}
	for replay, want := range map[bool][]any{true: {"Unverified; ask for the PIN.", "Look it up."}, false: {nil, nil}} {
		s.frames = []string{textFrame("done"), usageFrame(30, 0, 4, 0, "stop")}
		s.ask(s.provider(Options{ReplayReasoning: replay}), llm.ResponseParams{Input: input})
		sent := s.sentMessages(len(s.requests) - 1)
		s.Equal(want, []any{sent[1]["reasoning"], sent[2]["reasoning"]}, "replay=%v", replay)
	}
}

func (s *OpenAICompatSuite) TestCallsAnotherModelMadeAreSignedForAProviderThatRefusesThemUnsigned() {
	provider := s.provider(Options{UnsignedCall: "skip"})
	s.frames = []string{textFrame("done"), usageFrame(30, 0, 4, 0, "stop")}

	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{
			{ID: "call-1", Name: "lookup", Arguments: "{}"},
			{ID: "call-2", Name: "lookup", Arguments: "{}"},
		}},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: "one"},
		{Role: llm.ToolResult, ToolCallID: "call-2", Content: "two"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{
			{ID: "call-3", Name: "lookup", Arguments: "{}", Signature: "sig-abc"},
			{ID: "call-4", Name: "lookup", Arguments: "{}"},
		}},
		{Role: llm.ToolResult, ToolCallID: "call-3", Content: "three"},
		{Role: llm.ToolResult, ToolCallID: "call-4", Content: "four"},
	}})

	signatures := func(message map[string]any) []any {
		calls, ok := message["tool_calls"].([]any)
		s.Require().True(ok)
		var signed []any
		for _, call := range calls {
			content, _ := call.(map[string]any)["extra_content"].(map[string]any)
			google, _ := content["google"].(map[string]any)
			signed = append(signed, google["thought_signature"])
		}
		return signed
	}
	sent := s.sentMessages(0)
	s.Equal([]any{"skip", nil}, signatures(sent[0]), "the first call of an unsigned turn carries the placeholder")
	s.Equal([]any{"sig-abc", nil}, signatures(sent[3]), "a turn the provider signed goes back as it was")
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

func (s *OpenAICompatSuite) TestParallelCallsStreamedUnderOneIndexStayApart() {
	// Gemini streams each parallel call whole, all under index 0, told apart by id.
	s.frames = []string{
		toolFrame(0, "call-1", "search_issues", `{"query":"timeout"}`),
		toolFrame(0, "call-2", "search_issues", `{"query":"crash"}`),
		usageFrame(20, 0, 12, 0, "tool_calls"),
	}
	provider := s.provider(Options{})

	response, _ := s.ask(provider, hello())

	s.Require().Len(response.ToolCalls, 2)
	s.Equal("call-1", response.ToolCalls[0].ID)
	s.Equal(`{"query":"timeout"}`, response.ToolCalls[0].Arguments)
	s.Equal("call-2", response.ToolCalls[1].ID)
	s.Equal(`{"query":"crash"}`, response.ToolCalls[1].Arguments)
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

func (s *OpenAICompatSuite) TestAnImagePartIsSentAsAnImageURL() {
	s.frames = []string{textFrame("a rose"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{Capabilities: llm.Capabilities{InputModalities: []string{llm.ModalityImage}}})

	s.ask(provider, llm.ResponseParams{Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Text: "what flower"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}, Detail: "low"}},
		},
	}}})

	messages := s.sentMessages(0)
	s.Require().Len(messages, 1)
	s.Equal("user", messages[0]["role"])

	content, ok := messages[0]["content"].([]any)
	s.Require().True(ok, "mixed text and image have to travel as parts, not a string")
	s.Require().Len(content, 2)

	text, ok := content[0].(map[string]any)
	s.Require().True(ok)
	s.Equal("text", text["type"])
	s.Equal("what flower", text["text"])

	image, ok := content[1].(map[string]any)
	s.Require().True(ok)
	s.Equal("image_url", image["type"])
	url, ok := image["image_url"].(map[string]any)
	s.Require().True(ok)
	s.Equal("low", url["detail"])
	s.Contains(url["url"], "data:image/jpeg;base64,")
}

func (s *OpenAICompatSuite) TestAToolResultImageRidesAsAFollowUpUserTurn() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{Capabilities: llm.Capabilities{InputModalities: []string{llm.ModalityImage}}})

	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "look"}}},
		{Role: llm.ToolResult, ToolCallID: "call-1", Parts: []llm.ContentPart{
			{Text: "2 roses"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff}}},
		}},
	}})

	messages := s.sentMessages(0)
	s.Require().Len(messages, 3)
	s.Equal("tool", messages[1]["role"])
	s.Equal("2 roses", messages[1]["content"])
	s.Equal("user", messages[2]["role"])
	content, ok := messages[2]["content"].([]any)
	s.Require().True(ok)
	image, ok := content[1].(map[string]any)
	s.Require().True(ok)
	s.Equal("image_url", image["type"])
}

func (s *OpenAICompatSuite) TestAnImageIsRefusedWhenTheModelCannotSee() {
	provider := s.provider(Options{})

	_, err := provider.Create(context.Background(), llm.ResponseParams{Input: []llm.Message{{
		Role:  llm.User,
		Parts: []llm.ContentPart{{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{1}}}},
	}}})

	s.Require().Error(err)
	s.Contains(err.Error(), "does not accept image input")
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

func (s *OpenAICompatSuite) TestImagesFollowAllToolResultsInABatch() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{Capabilities: llm.Capabilities{InputModalities: []string{"image"}}})
	s.ask(provider, llm.ResponseParams{Input: []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "one", Name: "look"}, {ID: "two", Name: "look"}}},
		{Role: llm.ToolResult, ToolCallID: "one", Parts: []llm.ContentPart{{Image: &llm.ImagePart{URL: "https://example.com/one.png"}}}},
		{Role: llm.ToolResult, ToolCallID: "two", Parts: []llm.ContentPart{{Image: &llm.ImagePart{URL: "https://example.com/two.png"}}}},
	}})
	messages := s.sentMessages(0)
	s.Require().Len(messages, 4)
	s.Equal("tool", messages[1]["role"])
	s.Equal("tool", messages[2]["role"])
	s.Equal("user", messages[3]["role"])
	encoded, err := json.Marshal(messages[3]["content"])
	s.Require().NoError(err)
	s.Contains(string(encoded), "one.png")
	s.Contains(string(encoded), "two.png")
}
