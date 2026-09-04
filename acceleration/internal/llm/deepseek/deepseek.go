// Package deepseek reaches DeepSeek models on Baseten's shared Model APIs.
//
// Nothing has to be deployed for these: Baseten hosts them behind one OpenAI-compatible
// endpoint that every model on it shares, so a Baseten API key is the whole setup.
package deepseek

import (
	"errors"
	"log/slog"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "deepseek"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "BASETEN_API_KEY"

// baseURLEnvVar overrides the endpoint, for a dedicated deployment rather than the shared
// Model APIs.
const baseURLEnvVar = "DEEPSEEK_BASE_URL"

// defaultBaseURL is Baseten's shared Model APIs endpoint.
const defaultBaseURL = "https://inference.baseten.co/v1"

// modelOwner is the prefix Baseten expects on a DeepSeek model id. The routing config
// names the model alone, so the provider puts its owner back on.
const modelOwner = "deepseek-ai/"

// defaultModel is used when the caller names no model.
const defaultModel = "DeepSeek-V4-Flash-0731"

// efforts are what the chat template accepts once thinking is on.
var efforts = []string{"low", "medium", "high"}

// Options configures the provider.
type Options struct {
	APIKey  string
	Model   string
	BaseURL string
	// Thinking lets the model reason before answering. It is off by default: reasoning
	// spends the whole token budget and most of the latency before the first word of the
	// answer, which is the wrong trade for a live conversation.
	Thinking bool
	// ReasoningEffort tunes how long the model thinks when Thinking is on. Empty leaves
	// the model's own default.
	ReasoningEffort string
	Logger          *slog.Logger
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("deepseek: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}

	model := llm.Capabilities{StreamsReasoning: options.Thinking}
	if options.Thinking {
		model.ReasoningEfforts = efforts
		model.DefaultEffort = options.ReasoningEffort
	}

	return openaicompat.New(openaicompat.Options{
		Provider:     ProviderName,
		Model:        upstreamModel(options.Model),
		StatsModel:   options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: model,
		// The chat template decides whether the model thinks, so the switch travels as a
		// template argument rather than as a standard request field.
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			args := map[string]any{"thinking": options.Thinking}
			if effort != "" {
				args["reasoning_effort"] = effort
			}
			return map[string]any{"chat_template_kwargs": args}
		},
		Logger: options.Logger,
	})
}

// upstreamModel returns the id Baseten expects, leaving an already-qualified one alone.
func upstreamModel(model string) string {
	if strings.Contains(model, "/") {
		return model
	}
	return modelOwner + model
}
