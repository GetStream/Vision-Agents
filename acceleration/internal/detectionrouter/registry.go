package detectionrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/moondream"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/roboflow"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detection/ultralytics"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[detection.Provider]() }

// DefaultRegistry returns a registry with every detection provider this build supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(moondream.ProviderName, func(spec routing.Spec) (detection.Provider, error) {
		return moondream.New(moondream.Options{Model: spec.Model})
	})

	registry.Register(roboflow.ProviderName, func(spec routing.Spec) (detection.Provider, error) {
		return roboflow.New(roboflow.Options{Model: spec.Model})
	})

	registry.Register(ultralytics.ProviderName, func(spec routing.Spec) (detection.Provider, error) {
		return ultralytics.New(ultralytics.Options{Model: spec.Model})
	})

	return registry
}
