// Package anthropic serves Claude through Anthropic's OpenAI-compatible endpoint.
package anthropic

import (
	"log/slog"
	"os"

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
	return openaicompat.New(openaicompat.Options{Provider: ProviderName, Model: options.Model, APIKey: options.APIKey, BaseURL: options.BaseURL, Logger: options.Logger})
}
