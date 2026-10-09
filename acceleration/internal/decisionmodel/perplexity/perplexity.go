// Package perplexity asks Perplexity's Decider models for typed judgements.
//
// Perplexity's Decisions API takes the System One protocol, and it is the only place
// Decider V1 is served now that OpenRouter has moved to V1.1. It refuses a field it does
// not know, which systemone already never sends.
package perplexity

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
)

// ProviderName is how this is named in stats and configuration.
const ProviderName = "perplexity"

// Endpoint is Perplexity's Decisions API.
var Endpoint = systemone.Endpoint{
	Provider:      ProviderName,
	APIKeyEnvVar:  "PERPLEXITY_API_KEY",
	BaseURLEnvVar: "PERPLEXITY_BASE_URL",
	BaseURL:       "https://api.perplexity.ai",
	Path:          "/v1/decisions",
	DefaultModel:  "pplx-decider-v1.1-27b",
}

// New returns a client for one Decider model.
func New(options systemone.Options) (*systemone.Client, error) {
	return systemone.New(Endpoint, options)
}
