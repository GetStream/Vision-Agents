// Package coreweave reaches open-weight models on CoreWeave Serverless Inference.
//
// The endpoint is Weights & Biases', because CoreWeave owns W&B and serves its serverless
// catalogue through it: the key is a W&B key and the host is api.inference.wandb.ai.
// CoreWeave's own api.coreweave.com is the management API for dedicated deployments,
// where each gateway gets a URL of its own and there is no shared catalogue to route to.
package coreweave

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "coreweave"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "COREWEAVE_API_KEY",
	BaseURLEnvVar:  "COREWEAVE_BASE_URL",
	DefaultBaseURL: "https://api.inference.wandb.ai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
