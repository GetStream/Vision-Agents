package llmrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/atlascloud"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/baidu"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/baseten"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/cerebras"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/cloudflare"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/coreweave"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/deepinfra"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/deepseek"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/digitalocean"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/fireworks"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemma"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gmicloud"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/inceptron"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/ionet"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/meta"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/nextbit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/novita"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/parasail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/phala"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/sailresearch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/siliconflow"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/together"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/venice"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/wafer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[Provider]() }

// DefaultRegistry returns a registry with every LLM provider this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(meta.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(meta.New(meta.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	registry.Register(openai.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(openai.New(openai.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	registry.Register(deepseek.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(deepseek.New(deepseek.Options{
			Model:           spec.Model,
			Thinking:        spec.Thinking,
			ReasoningEffort: spec.ReasoningEffort,
			Logger:          spec.Logger,
		}))
	})

	registry.Register(gemini.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(gemini.New(gemini.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	registry.Register(gemma.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(gemma.New(gemma.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	registry.Register(cerebras.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(cerebras.New(cerebras.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	registry.Register(anthropic.ProviderName, func(spec routing.Spec) (Provider, error) {
		return Started(anthropic.New(anthropic.Options{Model: spec.Model, Logger: spec.Logger}))
	})

	// The hosts of open-weight models. They serve the same weights over the same
	// protocol and differ only in what they charge and what they keep, so the model
	// entry decides everything about a request except which of them answers it.
	//
	// Each is registered through its own package's constructor rather than through
	// openweights.Host, because a host with something of its own to do -- Cloudflare
	// builds its endpoint out of an account id -- does it there.
	hosts := map[string]func(openweights.Options) (*openaicompat.LLM, error){
		atlascloud.ProviderName:   atlascloud.New,
		baidu.ProviderName:        baidu.New,
		baseten.ProviderName:      baseten.New,
		cloudflare.ProviderName:   cloudflare.New,
		coreweave.ProviderName:    coreweave.New,
		deepinfra.ProviderName:    deepinfra.New,
		digitalocean.ProviderName: digitalocean.New,
		fireworks.ProviderName:    fireworks.New,
		gmicloud.ProviderName:     gmicloud.New,
		inceptron.ProviderName:    inceptron.New,
		ionet.ProviderName:        ionet.New,
		nextbit.ProviderName:      nextbit.New,
		novita.ProviderName:       novita.New,
		parasail.ProviderName:     parasail.New,
		phala.ProviderName:        phala.New,
		sailresearch.ProviderName: sailresearch.New,
		siliconflow.ProviderName:  siliconflow.New,
		together.ProviderName:     together.New,
		venice.ProviderName:       venice.New,
		wafer.ProviderName:        wafer.New,
	}
	for name, build := range hosts {
		registry.Register(name, func(spec routing.Spec) (Provider, error) {
			return Started(build(openweights.Options{
				Model:           spec.Model,
				Thinking:        spec.Thinking,
				ReasoningEffort: spec.ReasoningEffort,
				Logger:          spec.Logger,
			}))
		})
	}
	return registry
}
