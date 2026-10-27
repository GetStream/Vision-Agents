package openai

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type OpenAISuite struct {
	suite.Suite
}

func TestOpenAISuite(t *testing.T) {
	suite.Run(t, new(OpenAISuite))
}

func (s *OpenAISuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
}

func (s *OpenAISuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
}

func (s *OpenAISuite) TestModelIsSentUnqualified() {
	// OpenAI's own model ids carry no owner prefix, unlike the open-weight providers.
	provider, err := New(Options{APIKey: "k", Model: "gpt-5.6-terra"})
	s.Require().NoError(err)

	s.Equal("gpt-5.6-terra", provider.Model())
}

func (s *OpenAISuite) TestADefaultModelIsUsedWhenNoneIsNamed() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	s.Equal(defaultModel, provider.Model())
}

func (s *OpenAISuite) TestAModelDeclaresTheReasoningEffortsItAccepts() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	model := provider.Capabilities()
	s.Contains(model.ReasoningEfforts, "max", "the 5.6 family is the first to take it")
	s.Equal("none", model.DefaultEffort,
		"a conversation cannot afford to wait while the model thinks")
	s.True(model.Store, "the responses endpoint keeps what it generates")
}

func (s *OpenAISuite) TestAnEffortTheModelDoesNotAcceptIsRefused() {
	_, err := New(Options{APIKey: "k", Model: "gpt-5-mini", ReasoningEffort: "max"})

	s.Require().Error(err)
	s.ErrorContains(err, "minimal, low, medium, high")
}

func (s *OpenAISuite) TestAModelIsRecognisedByItsFamily() {
	// A dated snapshot is the same model, so it accepts the same efforts.
	s.Equal(modelCapabilities["gpt-5.6"].ReasoningEfforts,
		capabilitiesFor("gpt-5.6-sol-2026-02-11").ReasoningEfforts)
	s.Equal(fallbackCapabilities, capabilitiesFor("some-future-model"))
}

func (s *OpenAISuite) TestAModelAcceptsImages() {
	s.Contains(capabilitiesFor("gpt-5.6-luna").InputModalities, "image")
	s.Contains(capabilitiesFor("gpt-5.5-chat").InputModalities, "image")
	s.Contains(capabilitiesFor("gpt-6-luna").InputModalities, "image")
	s.Contains(capabilitiesFor("gpt-6-astra").InputModalities, "image")
}

func (s *OpenAISuite) TestGPT6LunaAndSolCanBeToldNotToThink() {
	for _, model := range []string{"gpt-6-luna", "gpt-6-sol"} {
		provider, err := New(Options{APIKey: "k", Model: model})
		s.Require().NoError(err)

		s.Equal("none", provider.Capabilities().DefaultEffort, model)
		s.Contains(provider.Capabilities().ReasoningEfforts, "max", model)
	}
}

func (s *OpenAISuite) TestGPT6AstraCannotBeToldNotToThink() {
	_, err := New(Options{APIKey: "k", Model: "gpt-6-astra", ReasoningEffort: "none"})
	s.ErrorContains(err, "low, medium, high, xhigh, max")

	provider, err := New(Options{APIKey: "k", Model: "gpt-6-astra"})
	s.Require().NoError(err)
	s.Equal("low", provider.Capabilities().DefaultEffort,
		"a request naming no effort must not be sent the none Astra rejects")
}

func (s *OpenAISuite) TestGPT61SolCannotBeToldNotToThink() {
	_, err := New(Options{APIKey: "k", Model: "gpt-6.1-sol", ReasoningEffort: "none"})
	s.ErrorContains(err, "low, medium, high, xhigh")

	provider, err := New(Options{APIKey: "k", Model: "gpt-6.1-sol"})
	s.Require().NoError(err)
	s.Equal("low", provider.Capabilities().DefaultEffort,
		"a request naming no effort must not be sent the none GPT-6.1 Sol rejects")
	s.Contains(provider.Capabilities().InputModalities, "image")
}

func (s *OpenAISuite) TestGPT61SolRefusesMax() {
	_, err := New(Options{APIKey: "k", Model: "gpt-6.1-sol", ReasoningEffort: "max"})
	s.ErrorContains(err, "low, medium, high, xhigh")
}

func (s *OpenAISuite) TestAnImagePartIsSentAsInputImage() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	items := provider.input(llm.ResponseParams{Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Text: "what flower"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}, Detail: "low"}},
		},
	}}})
	s.Require().Len(items, 1)

	raw, err := json.Marshal(items[0])
	s.Require().NoError(err)
	s.Contains(string(raw), `"type":"input_image"`)
	s.Contains(string(raw), `"detail":"low"`)
	s.Contains(string(raw), "data:image/jpeg;base64,")
	s.Contains(string(raw), `"type":"input_text"`)
}

func (s *OpenAISuite) TestAToolResultImageGoesInFunctionCallOutput() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	items := provider.input(llm.ResponseParams{Input: []llm.Message{{
		Role:       llm.ToolResult,
		ToolCallID: "call-1",
		Parts: []llm.ContentPart{
			{Text: "2 roses"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff}}},
		},
	}}})
	s.Require().Len(items, 1)

	raw, err := json.Marshal(items[0])
	s.Require().NoError(err)
	s.Contains(string(raw), `"type":"function_call_output"`)
	s.Contains(string(raw), `"type":"input_image"`)
	s.Contains(string(raw), "2 roses")
}

// TestAnExplicitBreakpointIsSent: in explicit mode OpenAI caches only at the breakpoints a
// request names, so instructions sent without one are never written to the cache.
func (s *OpenAISuite) TestAnExplicitBreakpointIsSent() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	items := provider.input(llm.ResponseParams{
		Instructions:       "Be brief.",
		Input:              []llm.Message{{Role: llm.User, Content: "hi"}},
		PromptCacheOptions: llm.PromptCacheOptions{Mode: llm.CacheExplicit},
	})
	s.Require().Len(items, 2)

	var sent struct {
		Content []map[string]any `json:"content"`
	}
	raw, err := json.Marshal(items[0])
	s.Require().NoError(err)
	s.Require().NoError(json.Unmarshal(raw, &sent))
	s.Require().Len(sent.Content, 1)
	s.Equal("Be brief.", sent.Content[0]["text"])
	s.Equal(map[string]any{"mode": "explicit"}, sent.Content[0]["prompt_cache_breakpoint"])
}

func (s *OpenAISuite) TestImplicitCachingSendsNoBreakpoint() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	items := provider.input(llm.ResponseParams{
		Instructions: "Be brief.",
		Input:        []llm.Message{{Role: llm.User, Content: "hi"}},
	})

	raw, err := json.Marshal(items[0])
	s.Require().NoError(err)
	s.NotContains(string(raw), "prompt_cache_breakpoint")
}

// TestTheTierAndUsageOpenAIReportedReachTheResponse uses a first turn of a cached
// conversation as gpt-5.6-terra answered it, served at the flex tier.
func (s *OpenAISuite) TestTheTierAndUsageOpenAIReportedReachTheResponse() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `event: response.completed
data: {"type":"response.completed","response":{"id":"r","status":"completed","service_tier":"flex",`+
			`"usage":{"input_tokens":2389,"input_tokens_details":{"cache_write_tokens":2386,"cached_tokens":0},`+
			`"output_tokens":22,"output_tokens_details":{"reasoning_tokens":7},"total_tokens":2411}}}

`)
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)

	s.Equal("flex", response.ServiceTier)
	s.Equal(llm.Usage{
		InputTokens:         2389,
		InputTokensDetails:  llm.InputTokensDetails{CacheWriteTokens: 2386},
		OutputTokens:        22,
		OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 7},
		TotalTokens:         2411,
	}, response.Usage)
}

// answering is a server that streams the opening of an answer and then nothing, until the
// request is given up on.
func answering() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.output_text.delta\n"+
			`data: {"type":"response.output_text.delta","item_id":"m","output_index":0,"content_index":0,"delta":"Hello","sequence_number":1}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
}

// drained reads a stream to its end and returns what it settled as.
func drained(stream *llm.Stream) llm.Response {
	for stream.Next() {
	}
	return stream.Response()
}

func (s *OpenAISuite) TestACallersDeadlineCutsTheResponseShort() {
	server := answering()
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()
	ctx, cancel := context.WithTimeout(s.T().Context(), 300*time.Millisecond)
	defer cancel()

	stream, err := provider.Create(ctx, llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)
	response := drained(stream)

	s.True(response.TimedOut)
	s.Equal(llm.StatusFailed, response.Status, "whoever waited still sees it fail")
	s.Equal("Hello", response.OutputText)
	s.ErrorIs(stream.Err(), context.DeadlineExceeded)
}

func (s *OpenAISuite) TestTheProvidersOwnLimitCutsTheResponseShort() {
	server := answering()
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL, Timeout: 300 * time.Millisecond})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)
	response := drained(stream)

	s.True(response.TimedOut)
	s.Equal("Hello", response.OutputText)
}

func (s *OpenAISuite) TestAResponseTheVendorFailedIsNotCutShort() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.failed\n"+
			`data: {"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"server_error","message":"the model fell over"}}}`+"\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)
	response := drained(stream)

	s.False(response.TimedOut)
	s.Equal(llm.StatusFailed, response.Status)
	s.ErrorContains(stream.Err(), "the model fell over")
	s.True(llm.Transient(stream.Err()), "a server error is OpenAI's own")
}

// failedWith is the error a stream ends with when OpenAI sends event as it.
func (s *OpenAISuite) failedWith(event string) error {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+event+"\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	s.Require().NoError(err)
	s.Equal(llm.StatusFailed, drained(stream).Status)
	return stream.Err()
}

func (s *OpenAISuite) TestAnErrorEventOnOpenAIsSideIsTransient() {
	err := s.failedWith(`{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null,"sequence_number":1}`)
	s.ErrorContains(err, "slow down")
	s.True(llm.Transient(err))
}

func (s *OpenAISuite) TestAFailureOfTheRequestItselfIsNotTransient() {
	err := s.failedWith(`{"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"invalid_prompt","message":"not this"}}}`)
	s.ErrorContains(err, "not this")
	s.False(llm.Transient(err))
}

func (s *OpenAISuite) TestTheInstructionsOpenAConversationThatIsReplayed() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	raw, err := json.Marshal(provider.params(llm.ResponseParams{
		Instructions: "You are Ashley.",
		Input:        []llm.Message{{Role: llm.User, Content: "hello"}},
	}))
	s.Require().NoError(err)
	var sent struct {
		PreviousResponseID string `json:"previous_response_id"`
		Input              []struct {
			Role string `json:"role"`
		} `json:"input"`
	}
	s.Require().NoError(json.Unmarshal(raw, &sent))

	s.Empty(sent.PreviousResponseID)
	s.Require().Len(sent.Input, 2)
	s.Equal("developer", sent.Input[0].Role)
	s.Contains(string(raw), "You are Ashley.")
}

func (s *OpenAISuite) TestAContinuedResponseDoesNotSendItsInstructionsAgain() {
	// The stored response already holds the developer message it was sent. Sending it
	// again appended a copy to the chain every turn: a 5k-token prompt became 48k after
	// ten turns of a voice call.
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	raw, err := json.Marshal(provider.params(llm.ResponseParams{
		Instructions:       "You are Ashley.",
		Input:              []llm.Message{{Role: llm.User, Content: "and tomorrow?"}},
		PreviousResponseID: "resp_1",
	}))
	s.Require().NoError(err)
	var sent struct {
		PreviousResponseID string `json:"previous_response_id"`
		Input              []struct {
			Role string `json:"role"`
		} `json:"input"`
	}
	s.Require().NoError(json.Unmarshal(raw, &sent))

	s.Equal("resp_1", sent.PreviousResponseID)
	s.Require().Len(sent.Input, 1)
	s.Equal("user", sent.Input[0].Role)
	s.NotContains(string(raw), "You are Ashley.")
}

func (s *OpenAISuite) TestAContinuedResponseCarryingOnlyAToolResultSendsJustThatResult() {
	provider, err := New(Options{APIKey: "k"})
	s.Require().NoError(err)

	items := provider.input(llm.ResponseParams{
		Instructions:       "You are Ashley.",
		Input:              []llm.Message{{Role: llm.ToolResult, ToolCallID: "call-1", Content: "booked"}},
		PreviousResponseID: "resp_1",
	})
	s.Require().Len(items, 1)

	raw, err := json.Marshal(items[0])
	s.Require().NoError(err)
	s.Contains(string(raw), `"type":"function_call_output"`)
}

// sendMessage is Slack's MCP slack_send_message input schema as its server listed it on
// 2026-10-08: two required properties and four optional ones (AI-969).
var sendMessage = llm.Tool{
	Name:        "slack__slack_send_message",
	Description: "Sends a message to a Slack channel or user.",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"channel_id":       map[string]any{"type": "string"},
			"message":          map[string]any{"type": "string"},
			"thread_ts":        map[string]any{"type": "string"},
			"draft_id":         map[string]any{"type": "string"},
			"reply_broadcast":  map[string]any{"type": "boolean"},
			"unfurl_app_links": map[string]any{"type": "boolean"},
		},
		"required": []any{"channel_id", "message"},
	},
}

// sentTool is the tool as the request to OpenAI carried it.
type sentTool struct {
	Strict     *bool          `json:"strict"`
	Parameters map[string]any `json:"parameters"`
}

// send offers one tool to a server that records the request, and returns the tool it got.
func (s *OpenAISuite) send(tool llm.Tool) sentTool {
	sent := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		sent <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "post hello"}},
		Tools: []llm.Tool{tool},
	})
	s.Require().NoError(err)
	_, err = llm.Collect(stream)
	s.Require().NoError(err)

	var request struct {
		Tools []sentTool `json:"tools"`
	}
	s.Require().NoError(json.Unmarshal(<-sent, &request))
	s.Require().Len(request.Tools, 1)
	return request.Tools[0]
}

// sentText asks one question in the given shape and returns the text config the request
// carried.
func (s *OpenAISuite) sentText(model string, text llm.TextParams) map[string]any {
	sent := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		sent <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "k", BaseURL: server.URL, Model: model})
	s.Require().NoError(err)
	defer provider.Close()

	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "rule"}},
		Text:  text,
	})
	s.Require().NoError(err)
	_, err = llm.Collect(stream)
	s.Require().NoError(err)

	var request struct {
		Text map[string]any `json:"text"`
	}
	s.Require().NoError(json.Unmarshal(<-sent, &request))
	return request.Text
}

// floorSchema is a schema-bound answer like the flow controller's.
var floorSchema = &llm.JSONSchema{Name: "ruling", Schema: map[string]any{
	"type":                 "object",
	"properties":           map[string]any{"floor": map[string]any{"type": "string", "enum": []string{"stop"}}},
	"required":             []string{"floor"},
	"additionalProperties": false,
}}

func (s *OpenAISuite) TestGPT56LunaIsHeldToASchemaStrictly() {
	text := s.sentText("gpt-5.6-luna", llm.TextParams{Format: llm.FormatJSONSchema, Schema: floorSchema})

	s.Equal(map[string]any{
		"type":   "json_schema",
		"name":   "ruling",
		"strict": true,
		"schema": map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"floor": map[string]any{"type": "string", "enum": []any{"stop"}}},
			"required":             []any{"floor"},
			"additionalProperties": false,
		},
	}, text["format"])
}

func (s *OpenAISuite) TestEveryKnownFamilyTakesASchema() {
	for family := range modelCapabilities {
		provider, err := New(Options{APIKey: "k", Model: family})
		s.Require().NoError(err)

		s.Truef(provider.Capabilities().JSONSchema, "%s takes Structured Outputs", family)
	}
}

func (s *OpenAISuite) TestAnUnknownModelIsAskedForAJSONObjectInsteadOfASchema() {
	// An unrecognised family claims nothing it might not do, so it gets json_object, which
	// is what a JSON request has always sent.
	asObject := s.sentText("o9-preview", llm.TextParams{Format: llm.FormatJSONObject})
	asSchema := s.sentText("o9-preview", llm.TextParams{Format: llm.FormatJSONSchema, Schema: floorSchema})

	s.Equal(map[string]any{"type": "json_object"}, asObject["format"])
	s.Equal(asObject, asSchema)
}

// TestAToolsOptionalArgumentsStayOptional is a tool offered as the caller described it. Left
// to itself, the Responses API turns a tool into strict mode by making every property
// required, so the model has to fill thread_ts with "" and Slack refuses the post.
func (s *OpenAISuite) TestAToolsOptionalArgumentsStayOptional() {
	sent := s.send(sendMessage)

	s.Require().NotNil(sent.Strict, "an omitted strict is strict mode on the Responses API")
	s.False(*sent.Strict)
	s.Equal([]any{"channel_id", "message"}, sent.Parameters["required"])
	s.NotContains(sent.Parameters, "additionalProperties")
}

// TestAnOptionalPropertyInsideAListCounts is the same normalization one level down: OpenAI
// also marks every property of an object in an array required.
func (s *OpenAISuite) TestAnOptionalPropertyInsideAListCounts() {
	sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": map[string]any{"type": "array", "items": map[string]any{
				"type":       "object",
				"properties": map[string]any{"name": map[string]any{"type": "string"}, "note": map[string]any{"type": "string"}},
				"required":   []string{"name"},
			}},
		},
		"required": []string{"items"},
	}})

	s.Require().NotNil(sent.Strict)
	s.False(*sent.Strict)
}

// TestAnOptionalPropertyInsideAnyOfCounts is the same normalization inside a union branch:
// OpenAI also marks every property of an object under anyOf required.
func (s *OpenAISuite) TestAnOptionalPropertyInsideAnyOfCounts() {
	sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"r": map[string]any{"anyOf": []any{
				map[string]any{
					"type":       "object",
					"properties": map[string]any{"a": map[string]any{}, "b": map[string]any{}},
					"required":   []any{"a"},
				},
				map[string]any{"type": "string"},
			}},
		},
		"required": []any{"r"},
	}})

	s.Require().NotNil(sent.Strict)
	s.False(*sent.Strict)
}

// TestAToolWithEveryPropertyRequiredIsSentAsBefore leaves strict out, as before AI-969:
// OpenAI's strict normalization takes nothing from a schema with no optional property.
func (s *OpenAISuite) TestAToolWithEveryPropertyRequiredIsSentAsBefore() {
	sent := s.send(llm.Tool{Name: "get_weather", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []string{"city"},
	}})

	s.Nil(sent.Strict)
}

// TestAToolWithEveryPropertyRequiredDecodedFromJSONIsSentAsBefore: a decoded MCP schema has
// required as []any, and it must count as much as a []string does.
func (s *OpenAISuite) TestAToolWithEveryPropertyRequiredDecodedFromJSONIsSentAsBefore() {
	sent := s.send(llm.Tool{Name: "get_weather", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []any{"city"},
	}})

	s.Nil(sent.Strict)
}

// TestADefinitionNamedLikeAKeywordIsASchema: a name under $defs, definitions, patternProperties
// or dependentSchemas is the author's, so default or enum there is no data keyword and
// properties is no properties map.
func (s *OpenAISuite) TestADefinitionNamedLikeAKeywordIsASchema() {
	optional := map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{}}}
	allRequired := map[string]any{
		"type":       "object",
		"properties": map[string]any{"x": map[string]any{}},
		"required":   []string{"x"},
	}
	for _, holder := range []string{"$defs", "definitions", "patternProperties", "dependentSchemas"} {
		for _, name := range []string{"default", "enum", "const", "examples"} {
			s.Run(holder+"/"+name+" with an optional property", func() {
				sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"a": map[string]any{}},
					"required":   []string{"a"},
					holder:       map[string]any{name: optional},
				}})

				s.Require().NotNil(sent.Strict)
				s.False(*sent.Strict)
			})
		}
		s.Run(holder+"/properties with every property required", func() {
			sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"a": map[string]any{}},
				"required":   []string{"a"},
				holder:       map[string]any{"properties": allRequired},
			}})

			s.Nil(sent.Strict)
		})
	}
}

// TestAPropertiesKeyThatIsDataIsNotASchema reads an object under default, enum, const or
// examples as a value: its properties key names no property, so nothing is optional.
func (s *OpenAISuite) TestAPropertiesKeyThatIsDataIsNotASchema() {
	data := map[string]any{"properties": map[string]any{"x": 1}}
	for _, keyword := range []string{"default", "enum", "const", "examples"} {
		s.Run(keyword, func() {
			value := any(data)
			if keyword == "enum" || keyword == "examples" {
				value = []any{data}
			}
			sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"config": map[string]any{"type": "object", keyword: value}},
				"required":   []string{"config"},
			}})

			s.Nil(sent.Strict)
		})
	}
}

// TestAnArgumentNamedPropertiesIsAProperty reads a property called properties as one more
// property, not as a properties keyword holding the schema's other keywords.
func (s *OpenAISuite) TestAnArgumentNamedPropertiesIsAProperty() {
	sent := s.send(llm.Tool{Name: "post", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"properties": map[string]any{"type": "string"}},
		"required":   []string{"properties"},
	}})

	s.Nil(sent.Strict)
}

// TestNumberedKeysShareTheLoadAndARateLimitedOneRests reads a pool from OPENAI_API_KEY_1 and
// _2. The first is refused with 429, so the turn is answered on the second, and the next turn
// skips the first while it rests.
func (s *OpenAISuite) TestNumberedKeysShareTheLoadAndARateLimitedOneRests() {
	first, second := "openai-"+rand.Text(), "openai-"+rand.Text()
	s.T().Setenv(apiKeyEnvVar+"_1", first)
	s.T().Setenv(apiKeyEnvVar+"_2", second)
	keys := make(chan string, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		keys <- key
		if key == first {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"slow down"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{BaseURL: server.URL})
	s.Require().NoError(err)
	defer provider.Close()

	for range 2 {
		stream, err := provider.Create(s.T().Context(), llm.ResponseParams{
			Input: []llm.Message{{Role: llm.User, Content: "hello"}},
		})
		s.Require().NoError(err)
		_, err = llm.Collect(stream)
		s.Require().NoError(err)
	}

	close(keys)
	var seen []string
	for key := range keys {
		seen = append(seen, key)
	}
	s.Equal([]string{first, second, second}, seen)
}
