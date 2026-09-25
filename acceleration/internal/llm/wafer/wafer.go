// Package wafer reaches open-weight models on Wafer's Serverless API.
//
// Wafer drops the publisher from the name, so the model is GLM-5.2 rather than
// zai-org/GLM-5.2. It also offers zero data retention per request, through a Wafer-ZDR
// header this provider does not send: retention is something a model entry declares,
// not something a turn decides.
package wafer

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "wafer"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "WAFER_API_KEY",
	BaseURLEnvVar:  "WAFER_BASE_URL",
	DefaultBaseURL: "https://pass.wafer.ai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
