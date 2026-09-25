// Package xai serves Grok through xAI's OpenAI-compatible chat completions endpoint.
package xai

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "xai"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "XAI_API_KEY"

const defaultBaseURL = "https://api.x.ai/v1"

const defaultModel = "grok-4.7"

// efforts are what Grok 4.7 accepts. It cannot be told not to think: none is a 400.
var efforts = []string{"low", "medium", "high", "xhigh"}

// Options configures the provider.
type Options struct {
	APIKey, Model, BaseURL string
	// ReasoningEffort is sent when a request names none. Empty means low rather than the
	// high xAI default to, because on the live path the first word matters more.
	ReasoningEffort string
	Logger          *slog.Logger
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("xai: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = "low"
	}

	return openaicompat.New(openaicompat.Options{
		Provider: ProviderName,
		Model:    options.Model,
		APIKey:   options.APIKey,
		BaseURL:  options.BaseURL,
		Capabilities: llm.Capabilities{
			ReasoningEfforts: efforts,
			DefaultEffort:    options.ReasoningEffort,
			StreamsReasoning: true,
			InputModalities:  []string{llm.ModalityImage},
		},
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			return map[string]any{"reasoning_effort": effort}
		},
		Logger: options.Logger,
	})
}
