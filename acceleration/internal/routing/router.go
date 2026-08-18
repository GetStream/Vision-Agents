// Package routing picks a provider for a request and records what happened. It knows
// nothing about any particular modality.
//
// A caller asks for either a concrete "provider/model" or a capability shortcut such as
// en-low-latency. Shortcuts resolve against the capabilities declared in config and are
// then ranked by live health, so a degraded provider drops down the list without anyone
// editing config. Selection walks the ranked list until one provider starts, which is
// where failover happens.
//
// What a started provider then does with audio, text or tokens is the modality's business:
// each one wraps Select in a session that knows how to read its own events and turn them
// into stat rows.
package routing

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Provider is the little a router needs from a model provider: it can be opened, closed
// and named. Everything that makes a modality itself lives on the modality's own
// interface.
type Provider interface {
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "elevenlabs".
	Provider() string
	// Model is the model identifier used in stats.
	Model() string
}

// Inspector is the part of a router that does not depend on what it routes: which
// providers exist and where a target would go. It is what lets one HTTP surface serve
// every modality, since Router is generic and so a different type per modality.
type Inspector interface {
	Modality() Modality
	Config() ModalityConfig
	Providers(ctx context.Context) []Candidate
	Resolve(ctx context.Context, target string, languageHints []string) ([]Candidate, error)
}

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options[P Provider] struct {
	Modality Modality
	Config   ModalityConfig
	Registry *Registry[P]
	Store    *store.Store
	Live     *live.Client
	Logger   *slog.Logger
}

// Router selects providers and records per-request statistics.
type Router[P Provider] struct {
	modality Modality
	config   ModalityConfig
	registry *Registry[P]
	recorder *Recorder
	live     *live.Client
	logger   *slog.Logger
}

// Request is what a caller wants served.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// AgentID is the agent the work is for. Empty outside a conversation.
	AgentID string
	// CallID is the call the work happens in. Empty outside a conversation.
	CallID string
	// Tags are the customer's own cost labels, recorded on every row the session writes.
	Tags Tags
	// Target is a "provider/model" name or a capability shortcut.
	Target string
	// LanguageHints narrow multilingual models.
	LanguageHints []string
	// Voice selects the speaker for modalities that produce audio.
	Voice string
}

// Owner returns who the request is billed to and how it is labelled.
func (r Request) Owner() Owner {
	return Owner{CustomerID: r.CustomerID, AgentID: r.AgentID, CallID: r.CallID, Tags: r.Tags}
}

// Candidate is one option for serving a request, in preference order.
type Candidate struct {
	Config ProviderConfig
	Health live.Health
}

// New validates the options and returns a Router.
func New[P Provider](options Options[P]) (*Router[P], error) {
	if options.Modality == "" {
		return nil, errors.New("routing: modality is required")
	}
	if err := options.Config.Validate(); err != nil {
		return nil, err
	}
	if options.Registry == nil {
		return nil, errors.New("routing: registry is required")
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &Router[P]{
		modality: options.Modality,
		config:   options.Config,
		registry: options.Registry,
		recorder: NewRecorder(options.Modality, options.Store, options.Live, logger),
		live:     options.Live,
		logger:   logger,
	}, nil
}

// Modality is what this router serves.
func (r *Router[P]) Modality() Modality { return r.modality }

// Config returns the capability configuration in use.
func (r *Router[P]) Config() ModalityConfig { return r.config }

// Recorder is where a modality's session writes its stat rows.
func (r *Router[P]) Recorder() *Recorder { return r.recorder }

// Logger returns the router's logger.
func (r *Router[P]) Logger() *slog.Logger { return r.logger }

// Close stops the background stat writer.
func (r *Router[P]) Close() { r.recorder.Close() }

// Providers returns every configured provider with its live health, in config order.
func (r *Router[P]) Providers(ctx context.Context) []Candidate {
	candidates := make([]Candidate, 0, len(r.config.Providers))
	for _, provider := range r.config.Providers {
		candidates = append(candidates, Candidate{Config: provider, Health: r.health(ctx, provider)})
	}
	return candidates
}

// Resolve returns the candidates for a target, best first.
//
// A concrete "provider/model" resolves to itself. A capability shortcut resolves to every
// provider that meets its requirements, ranked by availability, then error rate, then
// average latency. Providers with no recent history keep their config order rather than
// jumping the queue on an unmeasured zero latency.
func (r *Router[P]) Resolve(ctx context.Context, target string, languageHints []string) ([]Candidate, error) {
	if target == "" {
		return nil, errors.New("routing: target is required")
	}

	if provider, ok := r.config.Provider(target); ok {
		return []Candidate{{Config: provider, Health: r.health(ctx, provider)}}, nil
	}

	alias, ok := r.config.Aliases[target]
	if !ok {
		return nil, fmt.Errorf("routing: unknown target %q", target)
	}

	var candidates []Candidate
	for _, provider := range r.config.Providers {
		if !alias.matches(provider) {
			continue
		}
		// A hinted language the model cannot handle rules it out, whatever the alias says.
		if !provider.Speaks(languageHints) {
			continue
		}
		candidates = append(candidates, Candidate{Config: provider, Health: r.health(ctx, provider)})
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("routing: no provider satisfies %q for languages %s", target, strings.Join(languageHints, ","))
	}

	rank(candidates)
	return candidates, nil
}

// Select builds and starts the best provider that will have it, falling back to the next
// candidate when one fails to start. Failures are recorded so the next request routes
// around them. The caller owns the returned provider and must Close it.
func (r *Router[P]) Select(ctx context.Context, request Request) (P, ProviderConfig, error) {
	var zero P

	if request.CustomerID == "" {
		return zero, ProviderConfig{}, errors.New("routing: customer id is required")
	}
	if err := request.Tags.Validate(); err != nil {
		return zero, ProviderConfig{}, err
	}

	candidates, err := r.Resolve(ctx, request.Target, request.LanguageHints)
	if err != nil {
		return zero, ProviderConfig{}, err
	}

	var failures []error
	for _, candidate := range candidates {
		if !r.registry.Has(candidate.Config.Provider) {
			failures = append(failures, fmt.Errorf("%s: no factory registered", candidate.Config.Name()))
			continue
		}

		provider, err := r.startCandidate(ctx, request, candidate)
		if err == nil {
			return provider, candidate.Config, nil
		}

		r.logger.Warn("provider failed to start, trying the next candidate",
			"modality", r.modality, "target", request.Target, "provider", candidate.Config.Name(), "error", err)
		failures = append(failures, fmt.Errorf("%s: %w", candidate.Config.Name(), err))
	}

	return zero, ProviderConfig{}, fmt.Errorf("routing: every candidate for %q failed: %w",
		request.Target, errors.Join(failures...))
}

func (r *Router[P]) startCandidate(ctx context.Context, request Request, candidate Candidate) (P, error) {
	var zero P

	spec := Spec{
		Model:         candidate.Config.Model,
		LanguageHints: request.LanguageHints,
		Voice:         request.Voice,
		Logger:        r.logger,
	}

	provider, err := r.registry.Build(candidate.Config.Provider, spec)
	if err != nil {
		r.recorder.Record(candidate.Config, Stat{
			Owner:     request.Owner(),
			StartedAt: time.Now().UTC(),
			Success:   false,
			ErrorCode: "build_failed",
		})
		return zero, err
	}

	startedAt := time.Now()
	if err := provider.Start(ctx); err != nil {
		provider.Close()
		r.recorder.Record(candidate.Config, Stat{
			Owner:     request.Owner(),
			StartedAt: startedAt.UTC(),
			LatencyMs: MsSince(startedAt),
			Success:   false,
			ErrorCode: "start_failed",
		})
		return zero, err
	}

	return provider, nil
}

// health reads live health, treating a Redis failure as "no information" so a broken
// stats path cannot take routing down with it.
func (r *Router[P]) health(ctx context.Context, provider ProviderConfig) live.Health {
	if r.live == nil {
		return live.Health{Provider: provider.Provider, Model: provider.Model, Available: true}
	}

	health, err := r.live.Health(ctx, string(r.modality), provider.Provider, provider.Model)
	if err != nil {
		r.logger.Warn("health lookup failed, treating provider as unmeasured",
			"modality", r.modality, "provider", provider.Name(), "error", err)
		return live.Health{Provider: provider.Provider, Model: provider.Model, Available: true}
	}
	return health
}

// rank orders candidates best first. The sort is stable, so equally-ranked candidates keep
// the order they were declared in.
func rank(candidates []Candidate) {
	sort.SliceStable(candidates, func(i, j int) bool {
		left, right := candidates[i].Health, candidates[j].Health

		if left.Available != right.Available {
			return left.Available
		}
		if left.ErrorRate() != right.ErrorRate() {
			return left.ErrorRate() < right.ErrorRate()
		}
		return latencyRank(left) < latencyRank(right)
	})
}

// latencyRank keeps unmeasured providers from winning on a latency of zero.
func latencyRank(health live.Health) float64 {
	if health.Requests == 0 {
		return math.Inf(1)
	}
	return health.LatencyMsAvg
}

// MsSince returns the elapsed milliseconds, which is how every latency is measured.
func MsSince(started time.Time) float64 {
	return float64(time.Since(started).Microseconds()) / 1000
}
