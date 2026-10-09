package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

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
