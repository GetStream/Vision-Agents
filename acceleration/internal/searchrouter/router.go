// Package searchrouter routes web search traffic and records what each search cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the search shape: which providers exist, and
// what counts as a unit of work, which for search is simply one question asked.
package searchrouter

import (
	"context"
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
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Router{Router: core}, nil
}

// Start selects a provider and opens a session, falling back to the next candidate when
// one fails to start. One session answers many searches.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        request.Target,
		LanguageHints: request.LanguageHints,
		Terms:         request.Options.Terms(),
		Search:        request.Options,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}
