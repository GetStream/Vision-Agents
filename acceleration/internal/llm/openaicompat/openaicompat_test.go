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
	"time"

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
	s.Require().NoError(provider.Start(context.Background()))
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

// ask sends one request and returns the events up to and including its completion.
func (s *OpenAICompatSuite) ask(provider *LLM, request llm.Request) (
	llm.CompletionComplete, []llm.Event,
) {
	s.Require().NoError(provider.Respond(request))

	var events []llm.Event
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
			if complete, ok := event.(llm.CompletionComplete); ok {
				return complete, events
			}
		case <-deadline:
			s.FailNow("the completion never settled")
			return llm.CompletionComplete{}, events
		}
	}
}

// hello is the simplest request a test can make.
func hello() llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}}
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

	complete, events := s.ask(provider, hello())

	s.Equal("Hello, world", complete.Text)
	s.Equal("stop", complete.FinishReason)

	var deltas []string
	for _, event := range events {
		if delta, ok := event.(llm.TextDelta); ok {
			deltas = append(deltas, delta.Text)
		}
	}
	s.Equal([]string{"Hello", ", world"}, deltas, "the caller can speak the answer as it arrives")
}

func (s *OpenAICompatSuite) TestUsageIsReportedForBilling() {
	s.frames = []string{textFrame("hi"), usageFrame(15, 8, 64, 47, "stop")}
	provider := s.provider(Options{})

	complete, _ := s.ask(provider, hello())

	s.EqualValues(15, complete.InputTokens)
	s.EqualValues(8, complete.CachedInputTokens)
	s.EqualValues(64, complete.OutputTokens)
	s.EqualValues(47, complete.ReasoningTokens)
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

	complete, _ := s.ask(provider, hello())

	s.EqualValues(43, complete.OutputTokens, "the last frame is the total, not one more instalment")
	s.EqualValues(15, complete.InputTokens)
}

func (s *OpenAICompatSuite) TestThinkingIsSeparatedFromTheAnswer() {
	s.frames = []string{
		reasoningFrame("The user said hi, so"),
		textFrame("Hello!"),
		usageFrame(11, 0, 20, 16, "stop"),
	}
	provider := s.provider(Options{Reasoning: true})

	complete, events := s.ask(provider, hello())

	s.Equal("Hello!", complete.Text, "thinking must never be spoken as the reply")

	var thinking string
	for _, event := range events {
		if delta, ok := event.(llm.ReasoningDelta); ok {
			thinking += delta.Text
		}
	}
	s.Equal("The user said hi, so", thinking, "but it is still available to a caller that wants it")
}

func (s *OpenAICompatSuite) TestInstructionsAreSentAsASystemMessage() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, llm.Request{
		Instructions: "Be terse.",
		Messages:     []llm.Message{{Role: llm.User, Content: "hi"}},
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

	s.ask(provider, llm.Request{Messages: []llm.Message{
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

func (s *OpenAICompatSuite) TestExtraBodyReachesTheProvider() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{
		ExtraBody: map[string]any{"chat_template_kwargs": map[string]any{"thinking": false}},
	})

	s.ask(provider, hello())

	args, ok := s.requests[0]["chat_template_kwargs"].(map[string]any)
	s.Require().True(ok, "a provider's own request fields must survive")
	s.Equal(false, args["thinking"])
}

func (s *OpenAICompatSuite) TestMaxTokensAndTemperatureAreOnlySentWhenAsked() {
	s.frames = []string{textFrame("ok"), usageFrame(5, 0, 1, 0, "stop")}
	provider := s.provider(Options{})

	s.ask(provider, hello())
	s.NotContains(s.requests[0], "max_completion_tokens")
	s.NotContains(s.requests[0], "temperature", "zero is a real temperature, so unset must not send one")

	temperature := 0.0
	s.ask(provider, llm.Request{
		Messages:    []llm.Message{{Role: llm.User, Content: "hi"}},
		MaxTokens:   32,
		Temperature: &temperature,
	})
	s.EqualValues(32, s.requests[1]["max_completion_tokens"])
	s.EqualValues(0.0, s.requests[1]["temperature"])
}

func (s *OpenAICompatSuite) TestInterruptSettlesWhatHadAlreadyArrived() {
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("I was saying"), usageFrame(11, 0, 4, 0, "")}
	provider := s.provider(Options{})

	s.Require().NoError(provider.Respond(llm.Request{ID: "c1", Messages: hello().Messages}))

	// Wait for the text to arrive before cutting the model off.
	s.Require().Eventually(func() bool {
		select {
		case event := <-provider.Events():
			_, isDelta := event.(llm.TextDelta)
			return isDelta
		default:
			return false
		}
	}, 5*time.Second, 10*time.Millisecond)

	s.Require().NoError(provider.Interrupt())

	complete := s.awaitCompletion(provider)
	s.True(complete.Interrupted)
	s.Equal("I was saying", complete.Text, "the words already spoken still happened")
	s.EqualValues(4, complete.OutputTokens, "and are still billed")
	close(s.hold)
}

func (s *OpenAICompatSuite) TestInterruptedCompletionIsNotReportedAsAnError() {
	// Barge-in is the design working, so it must not count against the provider's health.
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("cut short")}
	provider := s.provider(Options{})

	s.Require().NoError(provider.Respond(hello()))
	s.Require().Eventually(func() bool {
		select {
		case event := <-provider.Events():
			_, isDelta := event.(llm.TextDelta)
			return isDelta
		default:
			return false
		}
	}, 5*time.Second, 10*time.Millisecond)
	s.Require().NoError(provider.Interrupt())

	complete := s.awaitCompletion(provider)
	s.True(complete.Interrupted)

	select {
	case event := <-provider.Events():
		s.Failf("unexpected event", "an interrupted completion emitted %T", event)
	case <-time.After(100 * time.Millisecond):
	}
	close(s.hold)
}

func (s *OpenAICompatSuite) TestNamingACompletionAbandonsOnlyThatOne() {
	// A caller delegating background work runs several completions at once, and a premise
	// going stale must abandon that work alone rather than everything in flight.
	s.hold = make(chan struct{})
	s.frames = []string{textFrame("working")}
	provider := s.provider(Options{})

	s.Require().NoError(provider.Respond(llm.Request{ID: "keep", Messages: hello().Messages}))
	s.Require().NoError(provider.Respond(llm.Request{ID: "drop", Messages: hello().Messages}))
	s.awaitDeltas(provider, 2)

	s.Require().NoError(provider.Interrupt("drop"))

	complete := s.awaitCompletion(provider)
	s.Equal("drop", complete.CompletionID)
	s.True(complete.Interrupted)

	// The other one is still running: it settles only once the server lets it go.
	select {
	case event := <-provider.Events():
		s.Failf("unexpected event", "the completion that was kept emitted %T", event)
	case <-time.After(100 * time.Millisecond):
	}
	close(s.hold)
	s.Equal("keep", s.awaitCompletion(provider).CompletionID)
}

func (s *OpenAICompatSuite) TestInterruptingNothingInParticularIsNotAnError() {
	provider := s.provider(Options{})

	s.NoError(provider.Interrupt())
	s.NoError(provider.Interrupt("never-existed"))
}

func (s *OpenAICompatSuite) TestAFailedRequestStillSettlesTheCompletion() {
	// A caller waiting on CompletionComplete must never be left hanging, or a turn never
	// ends and the conversation stops.
	s.status = http.StatusUnauthorized
	provider := s.provider(Options{})

	complete, events := s.ask(provider, hello())

	s.Empty(complete.Text)
	s.False(complete.Interrupted)

	var failures []llm.Error
	for _, event := range events {
		if failure, ok := event.(llm.Error); ok {
			failures = append(failures, failure)
		}
	}
	s.Require().NotEmpty(failures, "the failure has to be reported so it can be recorded")
	s.Equal("stream", failures[0].Context)
}

func (s *OpenAICompatSuite) TestRespondBeforeStartIsRefused() {
	provider, err := New(Options{Provider: "p", Model: "m", APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	s.ErrorContains(provider.Respond(hello()), "not started")
}

func (s *OpenAICompatSuite) TestRespondWithoutMessagesIsRefused() {
	provider := s.provider(Options{})

	s.ErrorContains(provider.Respond(llm.Request{}), "at least one message")
}

func (s *OpenAICompatSuite) TestStartEmitsConnectedAndCloseEmitsDisconnected() {
	provider := s.provider(Options{Provider: "test"})

	connected, ok := (<-provider.Events()).(llm.Connected)
	s.Require().True(ok)
	s.Equal("test", connected.Provider)

	s.Require().NoError(provider.Close())

	disconnected, ok := (<-provider.Events()).(llm.Disconnected)
	s.Require().True(ok)
	s.True(disconnected.Clean)

	_, open := <-provider.Events()
	s.False(open, "the channel closes so a consumer's range loop ends")
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

// awaitDeltas drains events until the given number of text deltas have arrived, which is
// how a test knows every completion it started is really under way.
func (s *OpenAICompatSuite) awaitDeltas(provider *LLM, count int) {
	deadline := time.After(5 * time.Second)
	for seen := 0; seen < count; {
		select {
		case event := <-provider.Events():
			if _, ok := event.(llm.TextDelta); ok {
				seen++
			}
		case <-deadline:
			s.FailNow("the deltas never arrived")
			return
		}
	}
}

// awaitCompletion drains events until the completion settles.
func (s *OpenAICompatSuite) awaitCompletion(provider *LLM) llm.CompletionComplete {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			if complete, ok := event.(llm.CompletionComplete); ok {
				return complete
			}
		case <-deadline:
			s.FailNow("the completion never settled")
			return llm.CompletionComplete{}
		}
	}
}
