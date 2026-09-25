// Package digitalocean reaches open-weight models on DigitalOcean's serverless inference.
//
// DigitalOcean drops the publisher and keeps the version, so GLM 5.3 Flash is
// glm-5.3-flash and DeepSeek V4.1 Flash is deepseek-v4.1-flash.
package digitalocean

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "digitalocean"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "DIGITALOCEAN_INFERENCE_KEY",
	BaseURLEnvVar:  "DIGITALOCEAN_BASE_URL",
	DefaultBaseURL: "https://inference.do-ai.run/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
