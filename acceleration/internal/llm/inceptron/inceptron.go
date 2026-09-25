// Package inceptron reaches open-weight models on Inceptron's OpenAI-compatible API.
package inceptron

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "inceptron"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "INCEPTRON_API_KEY",
	BaseURLEnvVar:  "INCEPTRON_BASE_URL",
	DefaultBaseURL: "https://api.inceptron.io/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
