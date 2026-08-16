package routing

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
)

// Spec is what the router asks a factory to build. It carries the session-level settings
// a caller can influence; anything else is the factory's own default. Fields that mean
// nothing to a modality are ignored by its factories.
type Spec struct {
	Model string
	// LanguageHints narrow multilingual models.
	LanguageHints []string
	// Voice selects the speaker for modalities that produce audio.
	Voice string
	// EagerTurns asks a transcriber that can do it for provisional end-of-turn signals.
	// A provider without them ignores it, so asking is never an error.
	EagerTurns bool
	Logger     *slog.Logger
}

// Factory builds an unstarted provider.
type Factory[P Provider] func(spec Spec) (P, error)

// Registry maps a provider name to the factory that builds it. Registration is separate
// from configuration so a deployment can declare capabilities for a provider it has no
// credentials for and simply never route to it.
type Registry[P Provider] struct {
	mu        sync.RWMutex
	factories map[string]Factory[P]
}

// NewRegistry returns an empty registry.
func NewRegistry[P Provider]() *Registry[P] {
	return &Registry[P]{factories: make(map[string]Factory[P])}
}

// Register adds or replaces the factory for a provider.
func (r *Registry[P]) Register(provider string, factory Factory[P]) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[provider] = factory
}

// Build constructs a provider.
func (r *Registry[P]) Build(provider string, spec Spec) (P, error) {
	var zero P

	r.mu.RLock()
	factory, ok := r.factories[provider]
	r.mu.RUnlock()

	if !ok {
		return zero, fmt.Errorf("routing: no factory registered for provider %q", provider)
	}

	built, err := factory(spec)
	if err != nil {
		return zero, err
	}
	if any(built) == nil {
		return zero, errors.New("routing: factory returned no provider")
	}
	return built, nil
}

// Has reports whether a provider can be built.
func (r *Registry[P]) Has(provider string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.factories[provider]
	return ok
}
