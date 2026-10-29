// Package gemma reaches a Gemma 4 model served by vLLM on our own deployment: Nebius,
// Baseten, or any other OpenAI-compatible host.
//
// Unlike DeepSeek, Gemma is not on Baseten's shared Model APIs: it has to be deployed,
// which gives it its own endpoint. The recipe for both hosts is in deploy/gemma-4. While
// GEMMA_BASE_URL is unset, New fails, so routing moves to the next candidate
// and a capability shortcut still resolves.
package gemma

import (
	"errors"
	"log/slog"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemma"

// apiKeyEnvVar holds the credentials when Options does not, for whichever host serves the
// deployment.
const apiKeyEnvVar = "GEMMA_API_KEY"

// basetenAPIKeyEnvVar is read when apiKeyEnvVar is empty, so a deployment on Baseten needs
// no key of its own. A Nebius endpoint has its own token, so it needs GEMMA_API_KEY.
const basetenAPIKeyEnvVar = "BASETEN_API_KEY"

// baseURLEnvVar is the deployment's OpenAI-compatible endpoint root, ending in /v1: the
// host prints the URL once the model is deployed.
const baseURLEnvVar = "GEMMA_BASE_URL"

// modelOwner is the prefix vLLM serves these under, matching the Hugging Face repo. The
// routing config names the model alone, so the provider puts its owner back on.
const modelOwner = "google/"

// conversationTemperature is what a request that names none is answered at. The
// deployment's own default is Gemma's 1.0, at which it announces a tool call and then skips
// it, and rules differently on the same words.
const conversationTemperature = 0.2

// flowPurpose is the flow controller's requests, which are a classification and answered
// at zero.
const flowPurpose = "flow"

// defaultModel is used when the caller names no model.
const defaultModel = "gemma-4-26B-A4B-it"

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
		options.APIKey = keypool.Getenv(apiKeyEnvVar, basetenAPIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("gemma: " + apiKeyEnvVar + " or " + basetenAPIKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.BaseURL == "" {
		return nil, stack.Wrap(errors.New("gemma: " + baseURLEnvVar + " is required: see deploy/gemma-4"))
	}

	return openaicompat.New(openaicompat.Options{
		Provider:   ProviderName,
		Model:      upstreamModel(options.Model),
		StatsModel: options.Model,
		APIKey:     options.APIKey,
		BaseURL:    options.BaseURL,
		// The live path wants the first word, not a think-then-speak turn. The deployment
		// also defaults thinking off; this pins it on every request so a client cannot
		// enable it by accident.
		RequestFields: func(params llm.ResponseParams, _ string) map[string]any {
			fields := map[string]any{
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			}
			if params.Temperature == nil {
				fields["temperature"] = conversationTemperature
				if params.Purpose == flowPurpose {
					fields["temperature"] = 0.0
				}
			}
			return fields
		},
		// vLLM constrains decoding to the schema, which kept the flow controller to its
		// own values where json_object let it answer "continue" for a disposition.
		Capabilities: llm.Capabilities{JSONSchema: true},
		// Thinking off still leaves an empty thought channel in some answers.
		ThoughtChannel: true,
		// The thought channel can take the opening of a tool call with it, and the rest
		// arrives as text.
		ToolCallText: true,
		Logger:       options.Logger,
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
