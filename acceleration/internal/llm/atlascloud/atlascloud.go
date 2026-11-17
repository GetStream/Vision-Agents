// Package atlascloud reaches open-weight models on AtlasCloud's OpenAI-compatible API.
package atlascloud

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "atlascloud"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "ATLASCLOUD_API_KEY",
	BaseURLEnvVar:  "ATLASCLOUD_BASE_URL",
	DefaultBaseURL: "https://api.atlascloud.ai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
