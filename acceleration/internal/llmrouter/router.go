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

// Provider is an llm.LLM as the generic router sees it.
//
// The router opens every modality the same way, and an LLM has nothing to open: a response
// is one request, so there is no connection to make before the first one. Start is here to
// satisfy that shape and does nothing, which is what Started wraps a provider to say.
type Provider interface {
	llm.LLM
	Start(ctx context.Context) error
}

// Registry is the set of LLM providers a build can construct.
type Registry = routing.Registry[Provider]

// Started adapts a provider to the router's shape by giving it a Start that opens nothing.
// It takes a constructor's two results so a registry entry stays one line.
func Started[P llm.LLM](provider P, err error) (Provider, error) {
	if err != nil {
		return nil, err
	}
	return started{LLM: provider}, nil
}

type started struct{ llm.LLM }

func (started) Start(context.Context) error { return nil }

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

// Router selects an LLM provider and opens sessions.
type Router struct {
	*routing.Router[Provider]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[Provider]{
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
// fails to build. One session answers many turns.
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
