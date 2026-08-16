// Package llmrouter routes large-language-model traffic and records what each completion
// cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the LLM shape: which providers exist, what a
// session does with a conversation, and which events count as a unit of work.
package llmrouter

import (
	"context"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of LLM providers a build can construct.
type Registry = routing.Registry[llm.LLM]

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Logger   *slog.Logger
}

// Request is what a caller wants a model for.
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
	// LanguageHints narrow the candidates to models that cover them.
	LanguageHints []string
}

// Router selects an LLM provider and opens completion sessions.
type Router struct {
	*routing.Router[llm.LLM]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[llm.LLM]{
		Modality: routing.LLM,
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

// Start selects a provider and opens a session, falling back to the next candidate when one
// fails to start. One session answers many turns.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        request.Target,
		LanguageHints: request.LanguageHints,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}
