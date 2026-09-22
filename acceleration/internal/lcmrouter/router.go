// Package lcmrouter routes judgement traffic and records what each judgement cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the classifier shape: which providers exist,
// and what counts as a unit of work, which here is one request however many questions it
// carried, because that is how a classifier is billed.
package lcmrouter

import (
	"context"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of classifiers a build can construct.
type Registry = routing.Registry[lcm.Provider]

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Logger   *slog.Logger
}

// Request is what a caller wants a classifier for.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// AgentID is the agent the work is for. Empty outside a conversation.
	AgentID string
	// CallID is the call the work happens in. Empty outside a conversation.
	CallID string
	// Tags are the customer's own cost labels, recorded on every row this session writes.
	Tags routing.Tags
	// Target is a "provider/model" name or a capability shortcut. Empty takes the default
	// from Options.
	Target string
	// Options is the rest of what the caller asked for, which for a classifier is only
	// where to send it.
	Options options.Classifier
}

// Router selects a classifier and opens sessions.
type Router struct {
	*routing.Router[lcm.Provider]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[lcm.Provider]{
		Modality: routing.LCM,
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

// Start selects a classifier and opens a session, falling back to the next candidate when
// one fails to start. One session answers many requests.
//
// No language hints are passed, and that is deliberate rather than an omission. A
// classifier is asked questions in whatever language they were written in and is not
// configured per language, so narrowing candidates by the conversation's language would
// refuse a judgement for a reason that has nothing to do with the model.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	target := request.Target
	if target == "" {
		target = request.Options.Route()
	}

	core := routing.Request{
		CustomerID: request.CustomerID,
		AgentID:    request.AgentID,
		CallID:     request.CallID,
		Tags:       request.Tags,
		Target:     target,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}
