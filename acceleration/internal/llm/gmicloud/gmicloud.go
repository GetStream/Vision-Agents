// Package gmicloud reaches open-weight models on GMI Cloud's Inference Engine.
//
// The endpoint is gmi-serving.com rather than gmicloud.ai: the company's site and its
// inference API are on different domains.
//
// GMI's catalogue trails the other hosts. It stops at GLM 5.1 and MiniMax M2.7, and carries
// no Kimi K3 or Qwen 3.8, so the newest open weights have to come from somewhere else.
package gmicloud

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gmicloud"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "GMI_API_KEY",
	BaseURLEnvVar:  "GMI_BASE_URL",
	DefaultBaseURL: "https://api.gmi-serving.com/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
