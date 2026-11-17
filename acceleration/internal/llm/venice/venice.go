// Package venice reaches open-weight models on Venice's OpenAI-compatible API.
//
// Venice replaces every separator in a model name with a hyphen, so GLM 5.3 Flash is
// z-ai-glm-5-3-flash and there is no Hugging Face name to fall back on.
package venice

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "venice"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "VENICE_API_KEY",
	BaseURLEnvVar:  "VENICE_BASE_URL",
	DefaultBaseURL: "https://api.venice.ai/api/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
