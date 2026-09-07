// Package gemini reaches Google's Gemini models over their OpenAI-compatible endpoint.
//
// Google serves chat completions alongside their own API, which means these models need no
// implementation of their own: the difference from OpenAI is a base URL, a key and how
// reasoning is asked for.
package gemini

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// apiKeyEnvVar holds the credentials when Options does not. It is the name the Python side
// of this repository already uses for the same key.
const apiKeyEnvVar = "GOOGLE_API_KEY"

// defaultBaseURL is Google's OpenAI-compatible endpoint. The trailing slash is theirs: the
// compatibility layer lives under the versioned path rather than replacing it.
const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai/"

// defaultModel is used when the caller names no model. It is the same model llm-fast
// prefers, so the provider does not give a second answer to the question of which Gemini a
// conversation gets.
const defaultModel = "gemini-3.8-flash"

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL overrides the endpoint, for a gateway or a test server.
	BaseURL string
	// ReasoningEffort is how long the model may think before answering. Empty leaves the
	// model's own floor, because thinking spends the latency budget before the first word
	// and a live conversation has none to spend.
	//
	// It cannot be turned off altogether: unlike the 2.5 series, every Gemini 3 model
	// thinks, and the floor is as low as the setting goes.
	ReasoningEffort string
	Logger          *slog.Logger
}

// minimalEffort is the least thinking the Gemini 3 models up to 3.5 will do.
const minimalEffort = "minimal"

// lowEffort is the floor from 3.8 Flash onwards, which took minimal away.
const lowEffort = "low"

// modelCapabilities is what each model family accepts.
//
// 3.8 Flash made thinking a level rather than a budget and dropped minimal with it, so low
// is as far down as it goes. Asking for minimal there is a 400 halfway through a call
// rather than a slower turn, which is why the floor is a table and not one constant.
var modelCapabilities = map[string]llm.Capabilities{
	"gemini-3.8": {
		ReasoningEfforts: []string{lowEffort, "medium", "high"},
		DefaultEffort:    lowEffort,
		StreamsReasoning: false,
	},
}

// fallbackCapabilities is the older Gemini 3 line, where minimal is still the floor. There
// is no none: every Gemini 3 model thinks.
var fallbackCapabilities = llm.Capabilities{
	ReasoningEfforts: []string{minimalEffort, lowEffort, "medium", "high"},
	DefaultEffort:    minimalEffort,
	// Google reports thinking as tokens rather than streaming it, so there is no
	// reasoning text for a caller to separate out.
	StreamsReasoning: false,
}

// capabilitiesFor returns what one model accepts, by longest matching family.
func capabilitiesFor(model string) llm.Capabilities {
	best := ""
	for family := range modelCapabilities {
		if strings.HasPrefix(model, family) && len(family) > len(best) {
			best = family
		}
	}
	if best == "" {
		return fallbackCapabilities
	}
	return modelCapabilities[best]
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("gemini: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}

	model := capabilitiesFor(options.Model)
	if options.ReasoningEffort != "" {
		if err := model.Validate(
			llm.ResponseParams{Reasoning: llm.ReasoningParams{Effort: options.ReasoningEffort}},
		); err != nil {
			return nil, fmt.Errorf("gemini: %s: %w", options.Model, err)
		}
		model.DefaultEffort = options.ReasoningEffort
	}

	return openaicompat.New(openaicompat.Options{
		Provider:     ProviderName,
		Model:        options.Model,
		StatsModel:   options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: model,
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			return map[string]any{"reasoning_effort": effort}
		},
		Logger: options.Logger,
	})
}
