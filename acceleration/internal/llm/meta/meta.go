// Package meta serves Muse Spark through Meta's hosted Model API.
package meta

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

const ProviderName = "meta"
const DefaultModel = "muse-spark-1.3"
const apiKeyEnvVar = "META_API_KEY"
const defaultBaseURL = "https://api.meta.ai/v1"

type Options struct {
	APIKey          string
	Model           string
	BaseURL         string
	ReasoningEffort string
	Logger          *slog.Logger
}

// LLM adds Meta's tool-choice restrictions to the shared streaming implementation.
type LLM struct {
	*openaicompat.LLM
}

func New(options Options) (*LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("meta: META_API_KEY is required")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.Model != DefaultModel {
		return nil, fmt.Errorf("meta: unsupported model %q", options.Model)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = "low"
	}
	capabilities := llm.Capabilities{
		ReasoningEfforts: []string{"minimal", "low", "medium", "high", "xhigh"},
		DefaultEffort:    options.ReasoningEffort,
		InputModalities:  []string{llm.ModalityImage},
		// Chat Completions reports reasoning token counts, not reasoning text.
		StreamsReasoning: false,
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

func (l *LLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	// Do not silently weaken required/none to auto: that changes execution semantics.
	if params.ToolChoice != "" && params.ToolChoice != "auto" {
		return nil, fmt.Errorf("meta: tool choice %q is unsupported; use auto or omit tools", params.ToolChoice)
	}
	return l.LLM.Create(ctx, params)
}
