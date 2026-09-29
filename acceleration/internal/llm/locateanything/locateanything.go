// Package locateanything reaches NVIDIA's LocateAnything-3B deployed on Baseten.
//
// LocateAnything is a grounding model rather than a chat model: it is shown an image and
// a description and answers with boxes, written as <ref>car</ref><box><x1><y1><x2><y2></box>
// on a 0 to 1000 grid. It is routed as an LLM because that is the protocol its deployment
// speaks, and declared a specialist so no shortcut ever hands it a conversation.
//
// The Truss recipe is in deploy/locate-anything. Until someone pushes it,
// LOCATE_ANYTHING_BASE_URL is unset and New fails.
package locateanything

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "locateanything"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "BASETEN_API_KEY"

// baseURLEnvVar is the deployment's OpenAI-compatible endpoint, which Baseten prints once
// the model is pushed.
const baseURLEnvVar = "LOCATE_ANYTHING_BASE_URL"

// defaultModel is used when the caller names no model.
const defaultModel = "LocateAnything-3B"

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL is the deployment endpoint, up to and including /v1.
	BaseURL string
	Logger  *slog.Logger
}

// New builds the provider. It fails when the deployment endpoint is unknown.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("locateanything: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.BaseURL == "" {
		return nil, errors.New("locateanything: " + baseURLEnvVar + " is required: see deploy/locate-anything")
	}

	return openaicompat.New(openaicompat.Options{
		Provider:     ProviderName,
		Model:        options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: llm.Capabilities{InputModalities: []string{llm.ModalityImage}},
		Logger:       options.Logger,
	})
}
