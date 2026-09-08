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
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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

// ErrVoiceNotPrepared says a customer's own voice exists but this provider was never
// given it. The router treats that as a reason to try the next candidate rather than as a
// failure, because another provider may well have it.
var ErrVoiceNotPrepared = errors.New("routing: this provider has not been given that voice")

// VoiceResolver turns what a caller asked for into the id one provider knows the voice by.
//
// It exists because a customer's own voice is one row to us and a different id at every
// provider, and the router only learns which provider it has once it is picking between
// them. A name that is not a customer's own voice is returned unchanged, so a provider's
// library voices keep working without going near this.
type VoiceResolver interface {
	ResolveVoice(ctx context.Context, customerID, provider, voice string) (string, error)
}

// Options configures a Router. Store, Live and Voices are optional: without them the
// router still routes, it just stops recording and stops resolving custom voices.
type Options[P Provider] struct {
	Modality Modality
	Config   ModalityConfig
	Registry *Registry[P]
	Store    *store.Store
	Live     *live.Client
	Voices   VoiceResolver
	Logger   *slog.Logger
}

// Router selects providers and records per-request statistics.
type Router[P Provider] struct {
	modality Modality
	config   ModalityConfig
	registry *Registry[P]
	recorder *Recorder
	live     *live.Client
	voices   VoiceResolver
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
	// Providers is a priority list of where to try, in the order given, and it wins over
	// Target when it holds anything. Each entry is a bare provider name, a
	// "provider/model" or a capability shortcut, and each is expanded where it stands, so
	// the caller's order is the order candidates are tried in. Health only demotes what
	// is unavailable rather than reordering the list.
	Providers []string
	// Realtime restricts a priority list to one half of a vendor's models: a socket must
	// not be served by a batch model, and a recording should not be streamed at a live
	// one. Nil does not filter, which is what a concrete target has always done.
	Realtime *bool
	// LanguageHints narrow multilingual models.
	LanguageHints []string
	// Voice selects the speaker for modalities that produce audio.
	Voice string
	// Keyterms are the words a modality that recognises speech should expect.
	Keyterms []string
	// Terms are the optional terms this request asks for beyond a target and a language.
	// A candidate that has not declared one of them is not a candidate, so a term is
	// either honoured or the request fails saying nothing can serve it.
	Terms []options.Term
	// DataPolicy is what the caller requires of what happens to their data afterwards.
	// It narrows candidates the same way a term does, and for the same reason: being
	// refused is better than being served by a provider who keeps what they were sent.
	DataPolicy options.DataPolicy
	// STT, TTS and Search are the per-modality options, handed to the factory of
	// whichever modality this router serves.
	STT    options.STT
	TTS    options.TTS
	Search options.Search
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
		voices:   options.Voices,
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
// jumping the queue on an unmeasured zero latency. A shortcut that names a preferred model
// puts that one first instead, for as long as it is available.
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
	prefer(candidates, alias.Prefer)
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

	candidates, err := r.candidates(ctx, request)
	if err != nil {
		return zero, ProviderConfig{}, err
	}
	candidates, err = serving(candidates, request.Terms)
	if err != nil {
		return zero, ProviderConfig{}, err
	}
	candidates, err = permitted(candidates, request.DataPolicy)
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

// candidates is where a request may go, best first: a priority list where one was given,
// and the ranked resolution of a single target otherwise.
func (r *Router[P]) candidates(ctx context.Context, request Request) ([]Candidate, error) {
	if len(request.Providers) == 0 {
		return r.Resolve(ctx, request.Target, request.LanguageHints)
	}
	return r.resolveChain(ctx, request)
}

// resolveChain expands a priority list in the order it was written.
//
// Each entry is expanded where it stands and the results are concatenated, so a caller
// who said one vendor comes first is asked about the second only once the first is out.
// Within a single entry the old rules still apply: a shortcut ranks its own members by
// health, because naming a shortcut is declining to choose between them.
//
// Health does not reorder the list itself. A caller who wanted the fastest available
// model has a shortcut for that; a priority list means the order is the point, and the
// most an unavailable provider earns is a place at the back.
func (r *Router[P]) resolveChain(ctx context.Context, request Request) ([]Candidate, error) {
	var chain []Candidate
	var refusals []error

	for _, target := range request.Providers {
		found, err := r.resolveEntry(ctx, target, request.LanguageHints)
		if err != nil {
			refusals = append(refusals, err)
			continue
		}
		for _, candidate := range found {
			if request.Realtime != nil && candidate.Config.Realtime != *request.Realtime {
				continue
			}
			// First position wins, so naming a vendor and then a shortcut they are in
			// keeps them where the caller put them.
			if slices.ContainsFunc(chain, func(held Candidate) bool {
				return held.Config.Name() == candidate.Config.Name()
			}) {
				continue
			}
			chain = append(chain, candidate)
		}
	}

	if len(chain) == 0 {
		return nil, fmt.Errorf("routing: nothing in the priority list %s can serve this request: %w",
			strings.Join(request.Providers, ", "), errors.Join(refusals...))
	}

	demote(chain)
	return chain, nil
}

// resolveEntry is Resolve, plus the bare vendor name only a priority list may hold.
//
// The vendor name is tried last, so a name that is somehow both an alias and a provider
// still means the alias, which is what it means everywhere else.
func (r *Router[P]) resolveEntry(ctx context.Context, target string, languageHints []string) ([]Candidate, error) {
	candidates, err := r.Resolve(ctx, target, languageHints)
	if err == nil {
		return candidates, nil
	}

	var named []Candidate
	for _, provider := range r.config.Providers {
		if provider.Provider != target || !provider.Speaks(languageHints) {
			continue
		}
		named = append(named, Candidate{Config: provider, Health: r.health(ctx, provider)})
	}
	if len(named) == 0 {
		return nil, err
	}
	return named, nil
}

func (r *Router[P]) startCandidate(ctx context.Context, request Request, candidate Candidate) (P, error) {
	var zero P

	voice, err := r.voice(ctx, request, candidate.Config.Provider)
	if err != nil {
		return zero, err
	}

	spec := Spec{
		Model:         candidate.Config.Model,
		LanguageHints: request.LanguageHints,
		Voice:         voice,
		Keyterms:      request.Keyterms,
		STT:           request.STT,
		TTS:           request.TTS,
		Search:        request.Search,
		Overwrites:    request.STT.Overwrites[candidate.Config.Provider],
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

// voice is the id this candidate knows the requested voice by. Without a resolver, or for
// a name that is not a customer's own voice, it is what was asked for.
func (r *Router[P]) voice(ctx context.Context, request Request, provider string) (string, error) {
	if r.voices == nil || request.Voice == "" {
		return request.Voice, nil
	}
	return r.voices.ResolveVoice(ctx, request.CustomerID, provider, request.Voice)
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

// serving narrows candidates to the ones that can express every term the request names.
//
// A request asking for something none of them can do fails here rather than being served
// by a provider that ignores the term: a transcript that was quietly not diarized is
// worse than one that was refused, because nothing about it says so.
func serving(candidates []Candidate, terms []options.Term) ([]Candidate, error) {
	if len(terms) == 0 {
		return candidates, nil
	}

	kept := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Config.Supports(terms) {
			kept = append(kept, candidate)
		}
	}
	if len(kept) > 0 {
		return kept, nil
	}

	// Which terms to name is the ones nothing offered, since a request refused for
	// asking two things of which one is available should say which one is not.
	var unserved []string
	for _, term := range terms {
		if !slices.ContainsFunc(candidates, func(candidate Candidate) bool {
			return candidate.Config.Supports([]options.Term{term})
		}) {
			unserved = append(unserved, string(term))
		}
	}
	if len(unserved) == 0 {
		unserved = []string{"that combination of terms"}
	}
	return nil, fmt.Errorf("routing: no provider can express %s", strings.Join(unserved, ", "))
}

// permitted narrows candidates to the ones allowed to do this work at all.
//
// It reads like serving and fails like it, but it is a different kind of no. A term is
// something a provider cannot express; a data policy is something they are not permitted
// to be asked. Both end in a refusal naming what went unmet, because a request that said
// "not somewhere that trains on this" and was answered anyway has been answered wrongly
// in a way nothing about the transcript would show.
func permitted(candidates []Candidate, policy options.DataPolicy) ([]Candidate, error) {
	if !policy.Asks() {
		return candidates, nil
	}

	kept := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Config.Permits(policy) {
			kept = append(kept, candidate)
		}
	}
	if len(kept) > 0 {
		return kept, nil
	}

	// Which half to name is the half nothing met, since a request refused for asking two
	// things of which one is available should say which one is not.
	var unmet []string
	if policy.AllowTraining != nil && !*policy.AllowTraining {
		if !slices.ContainsFunc(candidates, func(candidate Candidate) bool {
			return candidate.Config.DataPolicy.TrainsOnData == options.ClaimNo
		}) {
			unmet = append(unmet, "not training on what it is sent")
		}
	}
	if policy.Retention != "" {
		ceiling := options.DataPolicy{Retention: policy.Retention}
		if !slices.ContainsFunc(candidates, func(candidate Candidate) bool {
			return candidate.Config.Permits(ceiling)
		}) {
			unmet = append(unmet, retentionUnmet(policy.Retention))
		}
	}
	if len(unmet) == 0 {
		unmet = []string{"that combination of data policy requirements"}
	}
	return nil, fmt.Errorf("routing: no provider meets your data policy: none offers %s",
		strings.Join(unmet, " and "))
}

// retentionUnmet names a retention requirement nothing offered, in the words it was asked
// in: asking for none is asking to be kept by nobody, not for a window of length zero.
func retentionUnmet(required options.Retention) string {
	if required == options.RetentionNone {
		return "keeping nothing at all"
	}
	return fmt.Sprintf("a retention of %s or less", required)
}

// demote moves unavailable candidates to the back of a priority list without disturbing
// the order of the rest.
//
// It is the whole of what health is allowed to do to a list the caller ordered by hand.
// A provider that is down should be tried last rather than first, and a provider that is
// merely slower than the next one down the list is still the one that was asked for.
func demote(candidates []Candidate) {
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].Health.Available && !candidates[j].Health.Available
	})
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

// prefer moves the alias's pinned model to the front, leaving the rest in ranked order
// behind it.
//
// Only while it is available: a pin says which model this deployment wants, not that a
// request should fail with it. Once it is back the ranking stops mattering again, which is
// the point of pinning in the first place.
func prefer(candidates []Candidate, name string) {
	if name == "" {
		return
	}
	at := slices.IndexFunc(candidates, func(candidate Candidate) bool {
		return candidate.Config.Name() == name && candidate.Health.Available
	})
	if at <= 0 {
		return
	}
	pinned := candidates[at]
	copy(candidates[1:at+1], candidates[:at])
	candidates[0] = pinned
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
