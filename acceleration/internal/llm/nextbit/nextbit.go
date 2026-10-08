// Package nextbit reaches open-weight models on NextBit's OpenAI-compatible API.
//
// NextBit names a model the way a container registry names an image -- deepseek:v4-flash,
// glm:5.3-flash -- so the publisher and the version are separated by a colon rather than
// a slash.
package nextbit

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "nextbit"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "NEXTBIT_API_KEY",
	BaseURLEnvVar:  "NEXTBIT_BASE_URL",
	DefaultBaseURL: "https://api.nextbit256.com/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
