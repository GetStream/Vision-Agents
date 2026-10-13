// Package moondream serves Moondream's vision model through its OpenAI-compatible endpoint.
package moondream

import (
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

const ProviderName = "moondream"

type Options struct {
	APIKey, Model, BaseURL string
	Logger                 *slog.Logger
}

func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("MOONDREAM_API_KEY")
	}
	if options.Model == "" {
		options.Model = "moondream3-preview"
	}
	if options.BaseURL == "" {
		options.BaseURL = "https://api.moondream.ai/v1/"
	}
	// The endpoint accepts tools and never calls one, so this answers questions about a
	// picture and nothing else. It does not reason.
	capabilities := llm.Capabilities{InputModalities: []string{llm.ModalityImage}}
	return openaicompat.New(openaicompat.Options{Provider: ProviderName, Model: options.Model, APIKey: options.APIKey, BaseURL: options.BaseURL, Capabilities: capabilities, Logger: options.Logger})
}
