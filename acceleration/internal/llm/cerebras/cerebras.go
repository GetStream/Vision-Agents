// Package cerebras reaches models on Cerebras' OpenAI-compatible inference API.
//
// Gemma 4 31B is what this is for: a live-path voice model that does not need a Baseten
// deployment. The existing gemma provider talks to that deployment and prepends google/;
// Cerebras serves the bare id, so the two stay separate rather than sharing a name.
package cerebras

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "cerebras"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "CEREBRAS_API_KEY"

// baseURLEnvVar overrides the endpoint, for a dedicated deployment rather than the public
// inference API.
const baseURLEnvVar = "CEREBRAS_BASE_URL"

// defaultBaseURL is Cerebras' public OpenAI-compatible endpoint.
const defaultBaseURL = "https://api.cerebras.ai/v1"

// defaultModel is Gemma 4 31B, the only Gemma 4 Cerebras currently hosts.
const defaultModel = "gemma-4-31b"

// Options configures the provider.
type Options struct {
	APIKey  string
	Model   string
	BaseURL string
	Logger  *slog.Logger
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("cerebras: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
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
