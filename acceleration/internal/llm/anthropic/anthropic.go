// Package anthropic serves Claude through Anthropic's Messages API.
//
// It speaks the native API rather than the OpenAI-compatible endpoint because only the
// native one caches: the compatibility endpoint takes no cache_control, so every turn of a
// long tool loop was billed as a fresh prompt. The compatibility path stays selectable with
// ANTHROPIC_OPENAI_COMPAT until it is retired.
package anthropic

import (
	"log/slog"
	"os"
	"strconv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

const ProviderName = "anthropic"

// compatEnvVar switches the provider back to Anthropic's OpenAI-compatible endpoint.
const compatEnvVar = "ANTHROPIC_OPENAI_COMPAT"

// workspaceHeader names the workspace a request bills to.
const workspaceHeader = "anthropic-workspace-id"

type Options struct {
	APIKey, Model, BaseURL string
	// WorkspaceID names the workspace a key that is not scoped to one bills to. Anthropic
	// refuses such a key without it.
	WorkspaceID string
	// Compat serves the model over the OpenAI-compatible endpoint, which does not cache.
	// ANTHROPIC_OPENAI_COMPAT=true sets it for every provider built.
	Compat bool
	Logger *slog.Logger
}

func New(options Options) (llm.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = keypool.Getenv("ANTHROPIC_API_KEY")
	}
	if options.WorkspaceID == "" {
		options.WorkspaceID = os.Getenv("ANTHROPIC_WORKSPACE_ID")
	}
	if options.Model == "" {
		options.Model = "claude-opus-5"
	}
	if compat, err := strconv.ParseBool(os.Getenv(compatEnvVar)); err == nil && compat {
		options.Compat = true
	}
	if options.Compat {
		return newCompat(options)
	}
	return newMessages(options)
}

// newCompat serves the model over the OpenAI-compatible endpoint.
func newCompat(options Options) (*openaicompat.LLM, error) {
	if options.BaseURL == "" {
		options.BaseURL = "https://api.anthropic.com/v1/"
	}
	var headers map[string]string
	if options.WorkspaceID != "" {
		headers = map[string]string{workspaceHeader: options.WorkspaceID}
	}
	// Every current Claude model reads images, and the compatibility endpoint takes them as
	// image_url parts. It ignores reasoning_effort, so no effort is declared and a request
	// naming one is refused rather than answered at the model's default.
	capabilities := llm.Capabilities{InputModalities: []string{llm.ModalityImage}}
	return openaicompat.New(openaicompat.Options{Provider: ProviderName, Model: options.Model, APIKey: options.APIKey, BaseURL: options.BaseURL, Headers: headers, Capabilities: capabilities, Logger: options.Logger})
}
