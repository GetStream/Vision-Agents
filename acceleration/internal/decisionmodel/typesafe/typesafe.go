// Package typesafe asks TypeSafe's System One models for typed judgements.
//
// Jev is the model; TypeSafe is the vendor, which is why this is registered under the
// vendor's name with the model beside it, the way every other provider here is.
package typesafe

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
)

// ProviderName is how this is named in stats and configuration.
const ProviderName = "typesafe"

// DefaultModel is the alias for the newest stable System One model. It moves when a release
// ships, so anything with thresholds tuned against one version should name that version.
const DefaultModel = "jev-latest"

// Endpoint is TypeSafe's own API, the one the protocol was written for.
var Endpoint = systemone.Endpoint{
	Provider:      ProviderName,
	APIKeyEnvVar:  "TYPESAFE_API_KEY",
	BaseURLEnvVar: "TYPESAFE_BASE_URL",
	BaseURL:       "https://api.typesafe.ai",
	Path:          "/v1/systemone",
	DefaultModel:  DefaultModel,
}

// New returns a client for one TypeSafe model.
func New(options systemone.Options) (*systemone.Client, error) {
	return systemone.New(Endpoint, options)
}
