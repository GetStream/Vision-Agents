// Package ollama reaches open-weight models on Ollama Cloud's OpenAI-compatible API.
//
// Ollama spells its own model ids, as in deepseek-v4.1-flash:1017, and switches thinking
// with reasoning_effort rather than the chat template arguments other hosts read. It
// streams thinking as "reasoning", which openaicompat does not read, so a model entry
// that asks to think is refused rather than answered with its reasoning dropped.
package ollama

import (
	"errors"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "ollama"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "OLLAMA_API_KEY",
	BaseURLEnvVar:  "OLLAMA_BASE_URL",
	DefaultBaseURL: "https://ollama.com/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) {
	if options.Thinking {
		return nil, stack.Wrap(errors.New(ProviderName + ": thinking is not supported, because its reasoning would not be streamed"))
	}
	if options.APIKey == "" {
		options.APIKey = keypool.Getenv(Host.APIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New(ProviderName + ": " + Host.APIKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		return nil, stack.Wrap(errors.New(ProviderName + ": model is required"))
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(Host.BaseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = Host.DefaultBaseURL
	}

	return openaicompat.New(openaicompat.Options{
		Provider:      ProviderName,
		Model:         options.Model,
		APIKey:        options.APIKey,
		BaseURL:       options.BaseURL,
		RequestFields: requestFields,
		Logger:        options.Logger,
	})
}

// requestFields turns thinking off, which is what Ollama documents "none" as.
func requestFields(llm.ResponseParams, string) map[string]any {
	return map[string]any{"reasoning_effort": "none"}
}
