// Package siliconflow reaches open-weight models on SiliconFlow's OpenAI-compatible API.
//
// SiliconFlow serves the Hugging Face name unchanged, so zai-org/GLM-5.3-Flash is the id
// here as well as on the hub.
//
// Its docs give one thinking switch for every model, a top-level enable_thinking, rather
// than the per-family fields openweights sends. Nobody has run a key against it to see
// whether those are also accepted, so check before declaring a model here in router.yaml.
package siliconflow

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "siliconflow"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "SILICONFLOW_API_KEY",
	BaseURLEnvVar:  "SILICONFLOW_BASE_URL",
	DefaultBaseURL: "https://api.siliconflow.com/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
