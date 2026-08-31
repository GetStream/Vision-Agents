package searchrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/exa"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/perplexity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/tavily"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[search.Provider]() }

// DefaultRegistry returns a registry with every search provider this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(tavily.ProviderName, func(spec routing.Spec) (search.Provider, error) {
		return tavily.New(tavily.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(exa.ProviderName, func(spec routing.Spec) (search.Provider, error) {
		return exa.New(exa.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(perplexity.ProviderName, func(spec routing.Spec) (search.Provider, error) {
		return perplexity.New(perplexity.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}
