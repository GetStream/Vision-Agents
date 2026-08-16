// Package openai reaches OpenAI's own chat completions API.
package openai

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "openai"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "OPENAI_API_KEY"

// defaultBaseURL is OpenAI's endpoint.
const defaultBaseURL = "https://api.openai.com/v1"

// defaultModel is used when the caller names no model. The cheapest, fastest tier is the
// right default for a conversation; a caller who wants a better answer names one.
const defaultModel = "gpt-5.6-luna"

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL overrides the endpoint, which is useful for a gateway in front of OpenAI.
	BaseURL string
	Logger  *slog.Logger
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("openai: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}

	return openaicompat.New(openaicompat.Options{
		Provider: ProviderName,
		Model:    options.Model,
		APIKey:   options.APIKey,
		BaseURL:  options.BaseURL,
		Logger:   options.Logger,
	})
}
