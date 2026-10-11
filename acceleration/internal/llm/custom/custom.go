// Package custom reaches a model a customer serves themselves: a fine-tune on a host of
// open weights, or vLLM or SGLang on their own GPUs. Each speaks OpenAI's chat completions,
// so this is openaicompat pointed at the customer's endpoint.
//
// The weights are usually one of the open families the hosts serve, under a name of the
// customer's choosing, so the family is still read out of the model id: a fine-tune of Qwen
// is told not to think the way Qwen is.
package custom

import (
	"errors"
	"log/slog"

	"github.com/openai/openai-go/v3/option"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is what a customer's own model is routed and recorded under.
const ProviderName = "custom"

// keyless is sent to an endpoint that takes no key. The client always sends one, and a
// server started without --api-key ignores it.
const keyless = "none"

// Options configures one of a customer's models.
type Options struct {
	// Name is what the customer calls the model, which is what stats record.
	Name string
	// Model is the id the endpoint serves the weights under.
	Model string
	// BaseURL is the endpoint root, up to and including /v1.
	BaseURL string
	// APIKey is empty for an endpoint that takes none.
	APIKey     string
	HTTPClient option.HTTPClient
	Logger     *slog.Logger
}

// New builds the provider. It performs no network access.
func New(options Options) (*openaicompat.LLM, error) {
	if options.Model == "" || options.BaseURL == "" {
		return nil, stack.Wrap(errors.New("custom: a model and a base url are required"))
	}
	if options.APIKey == "" {
		options.APIKey = keyless
	}

	return openaicompat.New(openaicompat.Options{
		Provider:      ProviderName,
		Model:         options.Model,
		StatsModel:    options.Name,
		APIKey:        options.APIKey,
		BaseURL:       options.BaseURL,
		Capabilities:  openweights.Capabilities(options.Model, false, ""),
		RequestFields: openweights.RequestFields(options.Model, false),
		HTTPClient:    options.HTTPClient,
		Logger:        options.Logger,
	})
}
