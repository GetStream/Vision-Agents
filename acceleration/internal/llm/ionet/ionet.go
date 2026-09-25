// Package ionet reaches open-weight models on io.net's Intelligence API.
//
// io.net rents GPUs from whoever has them rather than owning a fleet, which is why the
// catalogue is broad and the endpoint lives under a domain nobody would guess.
package ionet

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "ionet"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "IONET_API_KEY",
	BaseURLEnvVar:  "IONET_BASE_URL",
	DefaultBaseURL: "https://api.intelligence.io.solutions/api/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
