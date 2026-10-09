package decisionrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/openrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/perplexity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/typesafe"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[decisionmodel.Provider]() }

// DefaultRegistry returns a registry with every decision model this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(typesafe.ProviderName, func(spec routing.Spec) (decisionmodel.Provider, error) {
		return typesafe.New(systemone.Options{Model: spec.Model, Logger: spec.Logger})
	})
	registry.Register(openrouter.ProviderName, func(spec routing.Spec) (decisionmodel.Provider, error) {
		return openrouter.New(systemone.Options{Model: spec.Model, Logger: spec.Logger})
	})
	registry.Register(perplexity.ProviderName, func(spec routing.Spec) (decisionmodel.Provider, error) {
		return perplexity.New(systemone.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}
