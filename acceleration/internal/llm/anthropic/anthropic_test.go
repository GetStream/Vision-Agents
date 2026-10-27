package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// sent is one request the stub server received.
type sent struct {
	header http.Header
	body   map[string]any
	raw    []byte
}

// AnthropicSuite talks to a stub of the Messages API. SetupTest sets what it answers with
// and forgets what it was sent.
type AnthropicSuite struct {
	suite.Suite
	server *httptest.Server

	mu       sync.Mutex
	requests []sent
	status   int
	answer   string
}

func TestAnthropicSuite(t *testing.T) {
	suite.Run(t, new(AnthropicSuite))
}

func (s *AnthropicSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		s.mu.Lock()
		s.requests = append(s.requests, sent{header: r.Header.Clone(), body: body, raw: raw})
		status, answer := s.status, s.answer
		s.mu.Unlock()
		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			fmt.Fprint(w, answer)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, answer)
	}))
}

func (s *AnthropicSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = nil
	s.status = http.StatusOK
	s.answer = stream(textBlock(0, "hi"), stop("end_turn", 1))
}

func (s *AnthropicSuite) answerWith(status int, answer string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status, s.answer = status, answer
}

func (s *AnthropicSuite) sent() []sent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

func (s *AnthropicSuite) provider(options Options) llm.LLM {
	options.APIKey = "test"
	options.BaseURL = s.server.URL
	if options.Model == "" {
		options.Model = "claude-opus-5-5"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

func (s *AnthropicSuite) collect(provider llm.LLM, params llm.ResponseParams) (llm.Response, error) {
	stream, err := provider.Create(s.T().Context(), params)
	if err != nil {
		return llm.Response{}, err
	}
	return llm.Collect(stream)
}

func (s *AnthropicSuite) TestAnImageReachesClaude() {
	response, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Text: "what flower"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}}},
		},
	}}})
	s.Require().NoError(err)

	s.Equal("hi", response.OutputText)
	body := s.sent()[0].body
	s.Equal("claude-opus-5-5", body["model"])
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	image := content[1].(map[string]any)
	s.Equal("image", image["type"])
	s.Equal(map[string]any{"type": "base64", "media_type": "image/jpeg", "data": "/9g="}, image["source"])
}

func (s *AnthropicSuite) TestTheKeyAndWorkspaceAreNamedOnEveryRequest() {
	_, err := s.collect(s.provider(Options{WorkspaceID: "wrkspc_test"}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)

	header := s.sent()[0].header
	s.Equal("wrkspc_test", header.Get("anthropic-workspace-id"))
	s.Equal("test", header.Get("x-api-key"))
	s.Empty(header.Get("Authorization"))
}

func (s *AnthropicSuite) TestAReasoningEffortIsRefusedBeforeItIsSent() {
	_, err := s.collect(s.provider(Options{}), llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "think hard"}},
		Reasoning: llm.ReasoningParams{Effort: "high"},
	})

	s.ErrorContains(err, "does not reason")
	s.Empty(s.sent())
}

func (s *AnthropicSuite) TestToolsSystemAndConversationEndOnBreakpoints() {
	_, err := s.collect(s.provider(Options{}), toolRound(1))
	s.Require().NoError(err)

	body := s.sent()[0].body
	tools := body["tools"].([]any)
	s.Nil(tools[0].(map[string]any)["cache_control"], "one breakpoint after the last tool covers them all")
	s.Equal(ephemeral, tools[1].(map[string]any)["cache_control"])
	system := body["system"].([]any)
	s.Equal(ephemeral, system[len(system)-1].(map[string]any)["cache_control"])
	messages := body["messages"].([]any)
	s.Equal(ephemeral, lastBlockOf(messages[len(messages)-1])["cache_control"], "the request's own end is cached for the next round")
	s.Equal(ephemeral, lastBlockOf(messages[0])["cache_control"], "the previous request's end is where it wrote")
	s.Equal(4, strings.Count(string(s.sent()[0].raw), `"cache_control"`))
}

func (s *AnthropicSuite) TestTheNextRoundSendsTheCachedPrefixByteForByte() {
	provider := s.provider(Options{})
	_, err := s.collect(provider, toolRound(1))
	s.Require().NoError(err)
	_, err = s.collect(provider, toolRound(2))
	s.Require().NoError(err)

	var first, second struct {
		Tools    json.RawMessage   `json:"tools"`
		System   json.RawMessage   `json:"system"`
		Messages []json.RawMessage `json:"messages"`
	}
	s.Require().NoError(json.Unmarshal(s.sent()[0].raw, &first))
	s.Require().NoError(json.Unmarshal(s.sent()[1].raw, &second))
	s.Equal(string(first.Tools), string(second.Tools))
	s.Equal(string(first.System), string(second.System))
	s.Require().Greater(len(second.Messages), len(first.Messages))
	// A breakpoint is not part of the prefix it marks, so moving one on is no change.
	for i, message := range first.Messages {
		s.Equal(unmarked(message), unmarked(second.Messages[i]), "message %d of the first round is resent unchanged", i)
	}
	var wrote, read map[string]any
	s.Require().NoError(json.Unmarshal(first.Messages[len(first.Messages)-1], &wrote))
	s.Require().NoError(json.Unmarshal(second.Messages[len(first.Messages)-1], &read))
	s.Equal(ephemeral, lastBlockOf(wrote)["cache_control"])
	s.Equal(ephemeral, lastBlockOf(read)["cache_control"], "the second round reads where the first one wrote")
}

// unmarked is a message as JSON with its breakpoints taken out.
func unmarked(message json.RawMessage) string {
	var decoded map[string]any
	_ = json.Unmarshal(message, &decoded)
	for _, block := range decoded["content"].([]any) {
		delete(block.(map[string]any), "cache_control")
	}
	encoded, _ := json.Marshal(decoded)
	return string(encoded)
}

func (s *AnthropicSuite) TestNoRequestCarriesMoreThanFourBreakpoints() {
	params := toolRound(12)
	params.Input = append([]llm.Message{{Role: llm.System, Content: "Earlier conversation summary: none"}}, params.Input...)
	_, err := s.collect(s.provider(Options{}), params)
	s.Require().NoError(err)

	s.Equal(maxBreakpoints, strings.Count(string(s.sent()[0].raw), `"cache_control"`))
}

func (s *AnthropicSuite) TestExplicitCachingMarksOnlyTheToolsAndInstructions() {
	params := toolRound(3)
	params.PromptCacheOptions.Mode = llm.CacheExplicit
	_, err := s.collect(s.provider(Options{}), params)
	s.Require().NoError(err)

	s.Equal(2, strings.Count(string(s.sent()[0].raw), `"cache_control"`))
	messages := s.sent()[0].body["messages"].([]any)
	s.Nil(lastBlockOf(messages[len(messages)-1])["cache_control"])
}

func (s *AnthropicSuite) TestUsageCountsTheWholePromptWithItsCachedAndWrittenParts() {
	s.answerWith(http.StatusOK, stream(
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":40,"cache_creation_input_tokens":2000,"cache_read_input_tokens":30000,"output_tokens":1}}}`,
		textBlock(0, "done"),
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":120,"output_tokens_details":{"thinking_tokens":80}}}`,
	))

	response, err := s.collect(s.provider(Options{}), toolRound(1))
	s.Require().NoError(err)

	s.Equal(llm.Usage{
		InputTokens:         32040,
		InputTokensDetails:  llm.InputTokensDetails{CachedTokens: 30000, CacheWriteTokens: 2000},
		OutputTokens:        120,
		OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 80},
		TotalTokens:         32160,
	}, response.Usage)
}

func (s *AnthropicSuite) TestAToolCallStreamsWholeBesideTheText() {
	s.answerWith(http.StatusOK, stream(
		textBlock(0, "Looking that up."),
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"roses\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		stop("tool_use", 20),
	))

	response, err := s.collect(s.provider(Options{}), toolRound(1))
	s.Require().NoError(err)

	s.Equal("Looking that up.", response.OutputText)
	s.Equal(llm.StatusCompleted, response.Status, "stopping to call a tool is finishing")
	s.Equal([]llm.ToolCall{{ID: "toolu_1", Name: "lookup", Arguments: `{"query":"roses"}`}}, response.ToolCalls)
}

func (s *AnthropicSuite) TestAToolCallIsReplayedWithItsResultStraightAfterIt() {
	_, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{
		{Role: llm.User, Content: "find roses"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "toolu_1", Name: "lookup", Arguments: `{"query":"roses"}`}}},
		{Role: llm.User, Content: "are you there?"},
		{Role: llm.ToolResult, ToolCallID: "toolu_1", Parts: []llm.ContentPart{
			{Text: "a photo of roses"},
			{Image: &llm.ImagePart{MIME: "image/png", Data: []byte{1}}},
		}},
	}})
	s.Require().NoError(err)

	messages := s.sent()[0].body["messages"].([]any)
	s.Require().Len(messages, 3)
	call := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	s.Equal("tool_use", call["type"])
	s.Equal(map[string]any{"query": "roses"}, call["input"])
	turn := messages[2].(map[string]any)["content"].([]any)
	result := turn[0].(map[string]any)
	s.Equal("tool_result", result["type"], "a result comes first in the turn after its call")
	s.Equal("toolu_1", result["tool_use_id"])
	s.Equal("image", result["content"].([]any)[1].(map[string]any)["type"])
	s.Equal("are you there?", turn[1].(map[string]any)["text"])
}

func (s *AnthropicSuite) TestTheTokenCapEndsTheResponseIncomplete() {
	s.answerWith(http.StatusOK, stream(textBlock(0, "and then"), stop("max_tokens", 16)))

	response, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "go on"}}, MaxOutputTokens: 16})
	s.Require().NoError(err)

	s.Equal(llm.StatusIncomplete, response.Status)
	s.Equal(llm.ReasonMaxOutputTokens, response.IncompleteReason)
	s.EqualValues(16, s.sent()[0].body["max_tokens"])
}

func (s *AnthropicSuite) TestARefusalEndsTheResponseAsFiltered() {
	s.answerWith(http.StatusOK, stream(stop("refusal", 0)))

	response, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)

	s.Equal(llm.ReasonContentFilter, response.IncompleteReason)
}

func (s *AnthropicSuite) TestARejectedRequestFailsCreateWithAnthropicsReason() {
	s.answerWith(http.StatusBadRequest, `{"type":"error","error":{"type":"invalid_request_error","message":"tool_choice any is not supported for this model"}}`)

	_, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}, ToolChoice: "required"})

	s.ErrorContains(err, "tool_choice any is not supported")
	s.Len(s.sent(), 1, "a 400 is not retried")
}

func (s *AnthropicSuite) TestAnErrorMidStreamFailsTheResponse() {
	s.answerWith(http.StatusOK, strings.TrimSuffix(stream(textBlock(0, "par")), "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")+
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")

	response, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})

	s.ErrorContains(err, "Overloaded")
	s.Equal(llm.StatusFailed, response.Status)
	s.Equal("par", response.OutputText)
	s.True(llm.Transient(err), "overloaded is Anthropic's own and passes")
}

func (s *AnthropicSuite) TestARefusalMidStreamIsNotTransient() {
	s.answerWith(http.StatusOK, strings.TrimSuffix(stream(textBlock(0, "par")), "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")+
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Not this\"}}\n\n")

	_, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})

	s.ErrorContains(err, "Not this")
	s.False(llm.Transient(err))
}

func (s *AnthropicSuite) TestTheCompatibilityPathStaysSelectable() {
	s.answerWith(http.StatusOK, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"compat\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	s.T().Setenv(compatEnvVar, "true")

	response, err := s.collect(s.provider(Options{}), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)

	s.Equal("compat", response.OutputText)
	s.Contains(s.sent()[0].body, "stream_options", "it was a chat completions request")
}

// ephemeral is a cache_control breakpoint as it is sent.
var ephemeral = map[string]any{"type": "ephemeral"}

// toolRound is the request after the given number of tool rounds: the instructions, two
// tools, the question, and a call and its result for each round.
func toolRound(rounds int) llm.ResponseParams {
	params := llm.ResponseParams{
		Instructions: "You are a support engineer.",
		Tools: []llm.Tool{
			{Name: "lookup", Description: "Looks a thing up.", Parameters: map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}, "required": []string{"query"}}},
			{Name: "clock", Description: "Says the time."},
		},
		Input: []llm.Message{{Role: llm.User, Content: "Why did the webhook fail?"}},
	}
	for round := 1; round <= rounds; round++ {
		id := fmt.Sprintf("toolu_%d", round)
		params.Input = append(params.Input,
			llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: id, Name: "lookup", Arguments: fmt.Sprintf(`{"query":"round %d"}`, round)}}},
			llm.Message{Role: llm.ToolResult, ToolCallID: id, Content: fmt.Sprintf("result %d", round)},
		)
	}
	return params
}

func lastBlockOf(message any) map[string]any {
	content := message.(map[string]any)["content"].([]any)
	return content[len(content)-1].(map[string]any)
}

// stream is a Messages API event stream: message_start, the given events one per line,
// then message_stop. A first event that is itself a message_start replaces the default.
func stream(events ...string) string {
	var lines []string
	for _, event := range events {
		lines = append(lines, strings.Split(event, "\n")...)
	}
	if len(lines) == 0 || !strings.Contains(lines[0], `"message_start"`) {
		lines = append([]string{`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":5,"output_tokens":1}}}`}, lines...)
	}
	lines = append(lines, `{"type":"message_stop"}`)
	var b strings.Builder
	for _, line := range lines {
		var typed struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(line), &typed)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", typed.Type, line)
	}
	return b.String()
}

// textBlock is a text content block streamed whole, as one event per line.
func textBlock(index int, text string) string {
	quoted, _ := json.Marshal(text)
	return strings.Join([]string{
		fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`, index),
		fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":%s}}`, index, quoted),
		fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, index),
	}, "\n")
}

func stop(reason string, outputTokens int) string {
	return fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q},"usage":{"output_tokens":%d}}`, reason, outputTokens)
}
