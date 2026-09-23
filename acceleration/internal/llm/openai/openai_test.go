package openai

import (
	"encoding/json"
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
