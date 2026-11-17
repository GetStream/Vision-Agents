// Package deepinfra reaches open-weight models on DeepInfra's OpenAI-compatible API.
//
// DeepInfra serves more of the models this router declares than any other host, and it
// serves them under the name they carry on Hugging Face, so a model entry reads the same
// here as the weights do.
package deepinfra

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "deepinfra"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "DEEPINFRA_API_KEY",
	BaseURLEnvVar:  "DEEPINFRA_BASE_URL",
	DefaultBaseURL: "https://api.deepinfra.com/v1/openai",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
