package routing

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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
	// Keyterms are the words a modality that recognises speech should expect.
	Keyterms []string
	// STT, TTS and Search carry the rest of what the request asked for. A factory reads
	// its own modality's block and ignores the others, and only ever sees terms its model
	// declared, since routing does not offer a request to a model that cannot serve it.
	STT    options.STT
	TTS    options.TTS
	Search options.Search
	// Overwrites is this provider's own block from the request's overwrites, and nobody
	// else's. A factory that has one decodes it into a struct of its own with Settings,
	// which is what makes an unrecognised field an error rather than a setting that was
	// accepted and never sent.
	Overwrites json.RawMessage
	Logger     *slog.Logger
}

// Settings decodes this provider's overwrites into a struct of its own.
//
// Unknown fields are rejected, which is the whole point: an escape hatch that silently
// dropped a misspelt setting would be worse than not having one, because the caller would
// believe they had changed something. A spec with no overwrites leaves the struct alone.
func (s Spec) Settings(into any) error {
	if len(s.Overwrites) == 0 {
		return nil
	}

	decoder := json.NewDecoder(bytes.NewReader(s.Overwrites))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(into); err != nil {
		return fmt.Errorf("routing: overwrites for %s: %w", s.Model, err)
	}
	return nil
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
