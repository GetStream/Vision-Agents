package llmrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/deepseek"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemma"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[llm.LLM]() }

// DefaultRegistry returns a registry with every LLM provider this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(openai.ProviderName, func(spec routing.Spec) (llm.LLM, error) {
		return openai.New(openai.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(deepseek.ProviderName, func(spec routing.Spec) (llm.LLM, error) {
		return deepseek.New(deepseek.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(gemini.ProviderName, func(spec routing.Spec) (llm.LLM, error) {
		return gemini.New(gemini.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(gemma.ProviderName, func(spec routing.Spec) (llm.LLM, error) {
		return gemma.New(gemma.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}
