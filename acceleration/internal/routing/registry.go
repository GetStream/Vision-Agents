package routing

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Spec is what the router asks a factory to build. It carries the session-level settings
// a caller can influence; anything else is the factory's own default. Fields that mean
// nothing to a modality are ignored by its factories.
type Spec struct {
	Model string
	// Thinking enables the model's reasoning mode when the provider supports one.
	// It is declared by the model entry rather than inferred from an alias so a direct
	// model target has the same behavior as a shortcut.
	Thinking bool
	// ReasoningEffort bounds the amount of reasoning a thinking model performs.
	ReasoningEffort string
	// LanguageHints narrow multilingual models.
	LanguageHints []string
	// Voice selects the speaker for modalities that produce audio.
	Voice string
	// Keyterms are the words a modality that recognises speech should expect.
	Keyterms []string
	// Tools are what a modality that converses may call.
	Tools []llm.Tool
	// STT, TTS, STS and Search carry the rest of what the request asked for. A factory
	// reads its own modality's block and ignores the others, and only ever sees terms its
	// model declared, since routing does not offer a request to a model that cannot serve
	// it.
	STT    options.STT
	TTS    options.TTS
	STS    options.STS
	Search options.Search
	// DataPolicy is the request's own policy merged with the floor its customer's policies
	// set. The policy in an option block above is the caller's alone.
	DataPolicy options.DataPolicy
	// Overwrites is this provider's own block from the request's overwrites, and nobody
	// else's. A factory that has one decodes it into a struct of its own with Settings,
	// which is what makes an unrecognised field an error rather than a setting that was
	// accepted and never sent.
	Overwrites json.RawMessage
	// Endpoint is where a customer's own model is served, and empty for every other. It
	// dials public addresses only, since a tenant wrote it.
	Endpoint Endpoint
	// BaseURL is the base URL of the instance's regional API, which the deployment's routing
	// config set, and empty for the vendor's default host. Only a factory registered as
	// honoring OverrideBaseURL is ever given one.
	BaseURL string
	// APIKey is the instance's own key, in place of the vendor's variable, and empty for that.
	// Only a factory registered as honoring OverrideAPIKey is ever given one.
	APIKey string
	// Region is where the instance runs its work, never empty. A factory registered as
	// honoring OverrideRegion picks its vendor's host for it.
	Region string
	Logger *slog.Logger
}

// Override is something of an instance's own a factory builds with in place of its default.
// A factory declares the ones it honors when it is registered, and a router refuses to start
// with an instance that sets one its factory would drop.
type Override string

const (
	// OverrideAPIKey is Spec.APIKey, an instance's own key, from its api_key_env.
	OverrideAPIKey Override = "api_key"
	// OverrideBaseURL is Spec.BaseURL, an instance's regional endpoint.
	OverrideBaseURL Override = "base_url"
	// OverrideRegion is Spec.Region read by the factory itself to reach the vendor in that
	// region, so an instance outside the default region needs no endpoint.
	OverrideRegion Override = "region"
)

// ErrRequest marks a build error as the caller's mistake, a setting the model cannot take,
// rather than the provider's fault. The router records no stat for it: health is shared by
// every customer, and one caller's bad value must not mark a model unavailable for all.
var ErrRequest = errors.New("routing: the request was refused")

// ErrOverwrite says a provider could not read the overwrites it was given: a field it does
// not have, or a value of the wrong type. It is a request error, and the router answers
// with it rather than falling over to a vendor that would never read the block at all.
var ErrOverwrite = errors.New("routing: overwrites refused")

// refused marks err as a request error without changing what it says.
type refused struct{ err error }

func (r refused) Error() string   { return r.err.Error() }
func (r refused) Unwrap() []error { return []error{r.err, ErrRequest} }

// Refused marks a factory's error as the caller's mistake, see ErrRequest.
func Refused(err error) error { return refused{err: err} }

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
		return Refused(stack.Wrap(fmt.Errorf("%w for %s: %w", ErrOverwrite, s.Model, err)))
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
	// honors is what each provider's factory builds with of an instance's own.
	honors map[string][]Override
	// probes is set by Probe. ready is what each probe of an instance answered.
	probes bool
	ready  map[instance]error
}

// NewRegistry returns an empty registry.
func NewRegistry[P Provider]() *Registry[P] {
	return &Registry[P]{factories: make(map[string]Factory[P]), honors: make(map[string][]Override), ready: make(map[instance]error)}
}

// Register adds or replaces the factory for a provider, with the overrides it honors: the
// ones it builds with when a Spec carries them. A factory that honors none takes the
// vendor's default host and key, in the default region.
func (r *Registry[P]) Register(provider string, factory Factory[P], honors ...Override) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[provider] = factory
	r.honors[provider] = slices.Clone(honors)
}

// Honors reports whether the provider's factory builds with that override. A provider with
// no factory honors nothing.
func (r *Registry[P]) Honors(provider string, override Override) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return slices.Contains(r.honors[provider], override)
}

// Probe makes Ready build what it is asked about. A registry of real vendors probes: their
// factories read the deployment's config and do nothing else. A registry of stubs that count
// their builds, block in them or hand out one shared provider does not.
func (r *Registry[P]) Probe() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.probes = true
}

// Ready says why this deployment cannot build a provider's model, and nil when it can. A
// vendor whose key or endpoint the deployment lacks fails in its factory, before anything
// is dialled, so building the model and closing it unstarted is the whole check.
//
// Each model is built once and the answer kept: config is read from the environment, which
// does not change while the process runs, and a factory may take a turn on a shared key
// ring (deepgram.New calls Next), which a build per request would skew. A registry that
// does not Probe, and a provider with no factory, answer nil.
func (r *Registry[P]) Ready(provider, model string) error {
	return r.ReadyAs(provider, Spec{Model: model, Region: pricing.DefaultRegion})
}

// ReadyAs is Ready for one instance of the model, built with its own endpoint, key and
// region: an instance with a key of its own can be built where the vendor's variable is unset.
func (r *Registry[P]) ReadyAs(provider string, spec Spec) error {
	named := instance{name: provider + "/" + spec.Model, region: spec.Region}
	r.mu.RLock()
	_, ok := r.factories[provider]
	probes := r.probes
	err, probed := r.ready[named]
	r.mu.RUnlock()
	if !ok || !probes {
		return nil
	}
	if probed {
		return err
	}

	built, err := r.Build(provider, Spec{Model: spec.Model, BaseURL: spec.BaseURL, APIKey: spec.APIKey, Region: spec.Region})
	if err == nil {
		built.Close()
	}
	r.mu.Lock()
	r.ready[named] = err
	r.mu.Unlock()
	return err
}

// Build constructs a provider.
func (r *Registry[P]) Build(provider string, spec Spec) (P, error) {
	var zero P

	r.mu.RLock()
	factory, ok := r.factories[provider]
	r.mu.RUnlock()

	if !ok {
		return zero, stack.Wrap(fmt.Errorf("routing: no factory registered for provider %q", provider))
	}

	built, err := factory(spec)
	if err != nil {
		return zero, stack.Wrap(err)
	}
	if any(built) == nil {
		return zero, stack.Wrap(errors.New("routing: factory returned no provider"))
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
