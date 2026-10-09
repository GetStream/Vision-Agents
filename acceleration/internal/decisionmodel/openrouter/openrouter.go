// Package openrouter asks the decision models OpenRouter serves for typed judgements.
//
// OpenRouter's Decisions API takes the System One protocol as is, and it is the one key
// that reaches every vendor's decision model: Jev, GPT-6 Luna Decisions, Clef, Decider,
// Mercury Decide and the rest. A model is named the way OpenRouter names it, with the
// maker in front, e.g. "cloudflare/clef-flash".
package openrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
)

// ProviderName is how this is named in stats and configuration.
const ProviderName = "openrouter"

// Endpoint is OpenRouter's Decisions API. It is under /api/alpha rather than /api/v1, which
// is OpenRouter's name for it rather than a stage this deployment opted into.
var Endpoint = systemone.Endpoint{
	Provider:      ProviderName,
	APIKeyEnvVar:  "OPENROUTER_API_KEY",
	BaseURLEnvVar: "OPENROUTER_BASE_URL",
	BaseURL:       "https://openrouter.ai",
	Path:          "/api/alpha/decisions",
	DefaultModel:  "typesafe/jev-1.13",
}

// New returns a client for one model on OpenRouter.
func New(options systemone.Options) (*systemone.Client, error) {
	return systemone.New(Endpoint, options)
}
