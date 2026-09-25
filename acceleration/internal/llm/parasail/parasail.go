// Package parasail reaches open-weight models on Parasail's OpenAI-compatible API.
//
// Parasail gives each deployment a name of its own -- parasail-glm-53-flash -- and also
// answers to the Hugging Face name. The routing config names the deployment, because that
// is what Parasail's own model pages call the endpoint.
package parasail

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "parasail"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "PARASAIL_API_KEY",
	BaseURLEnvVar:  "PARASAIL_BASE_URL",
	DefaultBaseURL: "https://api.parasail.io/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
