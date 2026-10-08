// Package sailresearch reaches open-weight models on Sail Research's serverless API.
//
// Sail serves the Hugging Face name unchanged and answers on the Responses endpoint as
// well as chat completions. This provider uses chat completions, which is what the rest
// of the open-weight hosts have in common.
package sailresearch

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "sailresearch"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "SAIL_API_KEY",
	BaseURLEnvVar:  "SAIL_BASE_URL",
	DefaultBaseURL: "https://api.sailresearch.com/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
