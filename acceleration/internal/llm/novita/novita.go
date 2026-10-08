// Package novita reaches open-weight models on Novita's OpenAI-compatible API.
//
// Novita is the only host in this router that serves Tencent's Hy4 preview alongside the
// rest of the catalogue. It lowercases the Hugging Face name, so zai-org/GLM-5.3-Flash is
// zai-org/glm-5.3-flash here.
package novita

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "novita"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:       ProviderName,
	APIKeyEnvVar:   "NOVITA_API_KEY",
	BaseURLEnvVar:  "NOVITA_BASE_URL",
	DefaultBaseURL: "https://api.novita.ai/openai/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
