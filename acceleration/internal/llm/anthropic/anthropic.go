// Package anthropic serves Claude through Anthropic's OpenAI-compatible endpoint.
package anthropic

import (
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

const ProviderName = "anthropic"

type Options struct {
	APIKey, Model, BaseURL string
	Logger                 *slog.Logger
}

func New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if options.Model == "" {
		options.Model = "claude-opus-5"
	}
	if options.BaseURL == "" {
		options.BaseURL = "https://api.anthropic.com/v1/"
	}
	// Every current Claude model reads images, and the compatibility endpoint takes them as
	// image_url parts. It ignores reasoning_effort, so no effort is declared and a request
	// naming one is refused rather than answered at the model's default.
	capabilities := llm.Capabilities{InputModalities: []string{llm.ModalityImage}}
	return openaicompat.New(openaicompat.Options{Provider: ProviderName, Model: options.Model, APIKey: options.APIKey, BaseURL: options.BaseURL, Capabilities: capabilities, Logger: options.Logger})
}
