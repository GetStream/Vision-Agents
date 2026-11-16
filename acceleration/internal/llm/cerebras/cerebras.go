// Package cerebras reaches models on Cerebras' OpenAI-compatible inference API.
//
// Cerebras serves the bare model id, unlike the gemma provider, which prepends google/, so
// the two stay separate rather than sharing a name. There is no default model: Gemma 4 31B,
// the one this was added for, has been archived by Cerebras, so a model is always named.
package cerebras

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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
		options.APIKey = keypool.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("cerebras: " + apiKeyEnvVar + " is required"))
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
