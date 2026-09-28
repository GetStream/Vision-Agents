// Package searchrouter routes web search traffic and records what each search cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the search shape: which providers exist, and
// what counts as a unit of work, which for search is simply one question asked.
package searchrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of search providers a build can construct.
type Registry = routing.Registry[search.Provider]

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Gate     routing.Gate
	Logger   *slog.Logger
}

// Request is what a caller wants a search provider for.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// AgentID is the agent the work is for. Empty outside a conversation.
	AgentID string
	// CallID is the call the work happens in. Empty outside a conversation.
	CallID string
	// Tags are the customer's own cost labels, recorded on every row this session writes.
	Tags routing.Tags
	// Target is a "provider/model" name or a capability shortcut.
	Target string
	// LanguageHints narrow the candidates to providers that cover them.
	LanguageHints []string
	// Options is the rest of what the caller asked for: a domain filter, a category, how
	// fresh the answer has to be. A term named here narrows the candidates to the
	// providers that declared it, so it is either honoured or the request is refused.
	Options options.Search
}

// Router selects a search provider and opens sessions.
type Router struct {
	*routing.Router[search.Provider]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[search.Provider]{
		Modality: routing.Search,
		Config:   options.Config,
		Registry: options.Registry,
		Store:    options.Store,
		Live:     options.Live,
		Gate:     options.Gate,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Router{Router: core}, nil
}

// Start selects a provider and opens a session, falling back to the next candidate when
// one fails to start. One session answers many searches.
//
// A search that fails is retried further down only when the caller wrote a priority list.
// On a live call a second provider's latency on top of the first one's failure is a
// longer silence than saying it could not check, and a list is the caller choosing that
// wait; a single target has not.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        request.Target,
		Providers:     request.Options.Providers,
		LanguageHints: request.LanguageHints,
		Terms:         request.Options.Terms(),
		Search:        request.Options,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}

	session := newSession(provider, config, core.Owner(), r.Recorder())
	if len(core.Providers) == 0 {
		return session, nil
	}
	session.fallback = func(ctx context.Context, query search.Query, failed routing.ProviderConfig) (*Session, search.Result, error) {
		candidates, err := r.Candidates(ctx, core)
		if err != nil {
			return nil, search.Result{}, err
		}
		var failures []error
		for _, candidate := range candidates {
			if ctx.Err() != nil {
				return nil, search.Result{}, ctx.Err()
			}
			if candidate.Config.Name() == failed.Name() {
				continue
			}
			// The list would win over the target, so it is cleared to ask this one alone.
			alternative := core
			alternative.Target = candidate.Config.Name()
			alternative.Providers = nil
			provider, selected, err := r.Select(ctx, alternative)
			if err != nil {
				failures = append(failures, err)
				continue
			}
			next := newSession(provider, selected, core.Owner(), r.Recorder())
			found, err := next.ask(ctx, provider, selected, query)
			if err != nil {
				_ = provider.Close()
				failures = append(failures, fmt.Errorf("%s: %w", selected.Name(), err))
				continue
			}
			return next, found, nil
		}
		return nil, search.Result{}, errors.Join(append(failures, errors.New("searchrouter: no fallback provider available"))...)
	}
	return session, nil
}
