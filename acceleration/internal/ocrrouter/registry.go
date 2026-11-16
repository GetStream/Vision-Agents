package ocrrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr/mistral"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[ocr.Provider]() }

// DefaultRegistry returns a registry with every OCR provider this build supports: Mistral
// OCR, and the vision language models the LLM router builds, each built the way the LLM
// router builds it and refused unless it accepts an image.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(mistral.ProviderName, func(spec routing.Spec) (ocr.Provider, error) {
		return mistral.New(mistral.Options{Model: spec.Model})
	})

	models := llmrouter.DefaultRegistry()
	for _, name := range []string{gemini.ProviderName, openai.ProviderName, anthropic.ProviderName} {
		registry.Register(name, func(spec routing.Spec) (ocr.Provider, error) {
			model, err := models.Build(name, spec)
			if err != nil {
				return nil, err
			}
			return newVision(model)
		})
	}

	return registry
}
