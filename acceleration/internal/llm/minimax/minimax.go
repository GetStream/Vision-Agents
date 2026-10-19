// Package minimax reaches MiniMax's own models on MiniMax's OpenAI-compatible API.
//
// M3.1 Flash has no published weights yet, so this is the only place it runs. It always
// thinks: a request that turns thinking off is refused with a 400, so the lowest effort
// is the fastest it can answer.
package minimax

import (
	"errors"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "minimax"

// apiKeyEnvVars hold the credentials when Options does not, in order.
var apiKeyEnvVars = []string{"MINIMAX_API_KEY", "MINIMAX"}

// baseURLEnvVar overrides the endpoint, for MiniMax's mainland host.
const baseURLEnvVar = "MINIMAX_BASE_URL"

const defaultBaseURL = "https://api.minimax.io/v1"

const defaultModel = "MiniMax-M3.1-Flash"

// efforts are what reasoning_effort accepts. M3.1 Flash defaults to max, which is far too
// slow for a turn, so a request that names none is sent at low.
var efforts = []string{"low", "medium", "high", "xhigh", "max"}

const defaultEffort = "low"

// temperature is what a request that names none is answered at, Gemma's 0.2.
const temperature = 0.2

// Options configures the provider.
type Options struct {
	APIKey  string
	Model   string
	BaseURL string
	// ReasoningEffort is the effort a request that names none is sent at.
	ReasoningEffort string
	Logger          *slog.Logger
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	for _, name := range apiKeyEnvVars {
		if options.APIKey != "" {
			break
		}
		options.APIKey = keypool.Getenv(name)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("minimax: MINIMAX_API_KEY is required"))
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
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = defaultEffort
	}
	return openaicompat.New(openaicompat.Options{
		Provider: ProviderName,
		Model:    options.Model,
		APIKey:   options.APIKey,
		BaseURL:  options.BaseURL,
		Capabilities: llm.Capabilities{
			ReasoningEfforts: efforts,
			DefaultEffort:    options.ReasoningEffort,
			StreamsReasoning: true,
		},
		RequestFields: requestFields(options.ReasoningEffort),
		Logger:        options.Logger,
	})
}

func requestFields(fallback string) func(llm.ResponseParams, string) map[string]any {
	return func(params llm.ResponseParams, effort string) map[string]any {
		if effort == "" {
			effort = fallback
		}
		fields := map[string]any{"reasoning_effort": effort}
		if params.Temperature == nil {
			fields["temperature"] = temperature
		}
		return fields
	}
}
