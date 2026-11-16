// Package phala reaches open-weight models on Phala's RedPill API.
//
// Phala runs its GPUs inside trusted execution environments and attests each response, so
// it is the host to reach for when the prompt must not be readable by the machine serving
// it. The endpoint is redpill.ai, which is the product name rather than the company's.
package phala

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "phala"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "PHALA_API_KEY",
	BaseURLEnvVar:  "PHALA_BASE_URL",
	DefaultBaseURL: "https://api.redpill.ai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
