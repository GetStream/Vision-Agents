// Package together reaches open-weight models on Together's OpenAI-compatible API.
//
// Together serves the Hugging Face name unchanged. It does not honour GLM's thinking
// switch -- its template reasons whatever the request says -- so a GLM entry routed here
// pays for tokens nobody reads before the answer starts.
package together

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "together"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "TOGETHER_API_KEY",
	BaseURLEnvVar:  "TOGETHER_BASE_URL",
	DefaultBaseURL: "https://api.together.xyz/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
