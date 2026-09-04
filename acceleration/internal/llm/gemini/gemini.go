// Package gemini reaches Google's Gemini models over their OpenAI-compatible endpoint.
//
// Google serves chat completions alongside their own API, which means these models need no
// implementation of their own: the difference from OpenAI is a base URL, a key and how
// reasoning is asked for.
package gemini

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// apiKeyEnvVar holds the credentials when Options does not. It is the name the Python side
// of this repository already uses for the same key.
const apiKeyEnvVar = "GOOGLE_API_KEY"

// defaultBaseURL is Google's OpenAI-compatible endpoint. The trailing slash is theirs: the
// compatibility layer lives under the versioned path rather than replacing it.
const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai/"

// defaultModel is used when the caller names no model. Flash-Lite is the fast, cheap tier,
// which is what a conversation wants; a caller who wants a better answer names one.
const defaultModel = "gemini-3.5-flash-lite"

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL overrides the endpoint, for a gateway or a test server.
	BaseURL string
	// ReasoningEffort is how long the model may think before answering, one of minimal,
	// low, medium or high. Empty means minimal, because thinking spends the latency
	// budget before the first word and a live conversation has none to spend.
	//
	// It cannot be turned off altogether: unlike the 2.5 series, every Gemini 3 model
	// thinks, and minimal is as low as the setting goes.
	ReasoningEffort string
	Logger          *slog.Logger
}

// minimalEffort is the least thinking a Gemini 3 model will do.
const minimalEffort = "minimal"

// capabilities is what these models accept. There is no none: every Gemini 3 model thinks.
var capabilities = llm.Capabilities{
	ReasoningEfforts: []string{minimalEffort, "low", "medium", "high"},
	DefaultEffort:    minimalEffort,
	// Google reports thinking as tokens rather than streaming it, so there is no
	// reasoning text for a caller to separate out.
	StreamsReasoning: false,
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("gemini: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = minimalEffort
	}

	model := capabilities
	model.DefaultEffort = options.ReasoningEffort

	return openaicompat.New(openaicompat.Options{
		Provider:     ProviderName,
		Model:        options.Model,
		StatsModel:   options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: model,
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			return map[string]any{"reasoning_effort": effort}
		},
		Logger: options.Logger,
	})
}
