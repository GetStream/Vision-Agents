// Package meta serves Meta's Muse models through their hosted Model API.
package meta

import (
	"errors"
	"fmt"
	"log/slog"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const ProviderName = "meta"
const DefaultModel = "muse-spark-1.3"
const apiKeyEnvVar = "META_API_KEY"
const defaultBaseURL = "https://api.meta.ai/v1"

// models are the Standard-tier ids, which Meta do not train on. The contributor variants
// do, and are refused.
var models = []string{DefaultModel, "muse-wonder-1.0"}

type Options struct {
	APIKey          string
	Model           string
	BaseURL         string
	ReasoningEffort string
	Logger          *slog.Logger
}

// LLM is the shared streaming implementation, with Meta's capabilities.
type LLM struct {
	*openaicompat.LLM
}

func New(options Options) (*LLM, error) {
	if options.APIKey == "" {
		options.APIKey = keypool.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("meta: META_API_KEY is required"))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if !slices.Contains(models, options.Model) {
		return nil, stack.Wrap(fmt.Errorf("meta: unsupported model %q", options.Model))
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = "low"
	}
	capabilities := llm.Capabilities{
		// max is Standard-tier only; the contributor models answer it with a 400.
		ReasoningEfforts: []string{"minimal", "low", "medium", "high", "xhigh", "max"},
		DefaultEffort:    options.ReasoningEffort,
		InputModalities:  []string{llm.ModalityImage},
		// Chat Completions reports reasoning token counts, not reasoning text.
		StreamsReasoning: false,
		// Meta answers any other tool_choice with a 400. Listing only auto keeps a request
		// that needs the rest refused here, and makes the harness drop the tools of a reply
		// that must not call one instead of sending none.
		ToolChoices: []string{"auto"},
	}
	if err := capabilities.Validate(llm.ResponseParams{Reasoning: llm.ReasoningParams{Effort: options.ReasoningEffort}}); err != nil {
		return nil, err
	}
	provider, err := openaicompat.New(openaicompat.Options{
		Provider:     ProviderName,
		Model:        options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: capabilities,
		RequestFields: func(_ llm.ResponseParams, effort string) map[string]any {
			return map[string]any{"reasoning_effort": effort}
		},
		Logger: options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &LLM{LLM: provider}, nil
}
