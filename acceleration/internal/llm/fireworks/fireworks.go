// Package fireworks reaches open-weight models on Fireworks' OpenAI-compatible API.
//
// Fireworks addresses a model by the account that published it, so the id is the path
// accounts/fireworks/models/glm-5p3-flash rather than a name. A dot is not allowed in
// that path, which is why a version reads 5p3.
package fireworks

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "fireworks"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "FIREWORKS_API_KEY",
	BaseURLEnvVar:  "FIREWORKS_BASE_URL",
	DefaultBaseURL: "https://api.fireworks.ai/inference/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
