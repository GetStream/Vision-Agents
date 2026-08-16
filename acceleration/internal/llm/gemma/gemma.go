// Package gemma reaches a Gemma 4 model deployed on Baseten.
//
// Unlike DeepSeek, Gemma is not on Baseten's shared Model APIs: it has to be deployed,
// which gives it its own endpoint. The Truss recipe is in deploy/gemma-4. Until someone
// pushes it, GEMMA_BASE_URL is unset and New fails, so routing moves to the next candidate
// and a capability shortcut still resolves.
package gemma

import (
	"errors"
	"log/slog"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemma"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "BASETEN_API_KEY"

// baseURLEnvVar is the deployment's OpenAI-compatible endpoint, which Baseten prints once
// the model is pushed.
const baseURLEnvVar = "GEMMA_BASE_URL"

// modelOwner is the prefix vLLM serves these under, matching the Hugging Face repo. The
// routing config names the model alone, so the provider puts its owner back on.
const modelOwner = "google/"

// defaultModel is used when the caller names no model.
const defaultModel = "gemma-4-E2B-it"

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL is the deployment endpoint, up to and including /v1.
	BaseURL string
	Logger  *slog.Logger
}

// New builds the provider. It fails when the deployment endpoint is unknown, which is what
// makes an undeployed Gemma fail over rather than hang.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("gemma: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.BaseURL == "" {
		return nil, errors.New("gemma: " + baseURLEnvVar + " is required: see deploy/gemma-4")
	}

	return openaicompat.New(openaicompat.Options{
		Provider:   ProviderName,
		Model:      upstreamModel(options.Model),
		StatsModel: options.Model,
		APIKey:     options.APIKey,
		BaseURL:    options.BaseURL,
		Logger:     options.Logger,
	})
}

// upstreamModel returns the id the deployment expects, leaving an already-qualified one
// alone.
func upstreamModel(model string) string {
	if strings.Contains(model, "/") {
		return model
	}
	return modelOwner + model
}
