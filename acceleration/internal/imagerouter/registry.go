package imagerouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen/fal"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen/google"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[imagegen.Provider]() }

// DefaultRegistry returns a registry with every image provider this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(fal.ProviderName, func(spec routing.Spec) (imagegen.Provider, error) {
		return fal.New(fal.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(google.ProviderName, func(spec routing.Spec) (imagegen.Provider, error) {
		return google.New(google.Options{Model: spec.Model})
	})

	return registry
}
