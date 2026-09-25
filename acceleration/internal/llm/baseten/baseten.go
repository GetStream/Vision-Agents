// Package baseten reaches open-weight models on Baseten's shared Model APIs.
//
// The deepseek and gemma providers already run on Baseten: one on the same shared Model
// APIs, the other on a deployment of our own. This is the rest of what Baseten hosts --
// GLM, Kimi, Nemotron -- under the one key all three share.
package baseten

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "baseten"

// Host is where this provider's requests go and what pays for them.
var Host = openweights.Host{
	Provider:     ProviderName,
	APIKeyEnvVar: "BASETEN_API_KEY",
	// Not BASETEN_BASE_URL: that one already names a dedicated deployment of a single
	// model, and pointing the shared Model APIs at it would send every model here.
	BaseURLEnvVar:  "BASETEN_INFERENCE_BASE_URL",
	DefaultBaseURL: "https://inference.baseten.co/v1",
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options openweights.Options) (*openaicompat.LLM, error) { return Host.New(options) }
