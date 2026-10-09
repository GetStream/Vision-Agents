package lcmrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm/typesafe"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[lcm.Provider]() }

// DefaultRegistry returns a registry with every classifier this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(typesafe.ProviderName, func(spec routing.Spec) (lcm.Provider, error) {
		return typesafe.New(typesafe.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}
