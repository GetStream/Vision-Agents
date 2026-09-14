// Package llmrouter routes large-language-model traffic and records what each completion
// cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the LLM shape: which providers exist, what a
// session does with a conversation, and which events count as a unit of work.
package llmrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
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
	// Quota caps what one end user may spend in a day. Absent means nothing is capped.
	Quota  *quota.Limiter
	Logger *slog.Logger
}

// Request is what a caller wants a model for.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// Caller is the end user the work is for, when the request came from a device rather
	// than from the customer's own backend. It is what daily limits are counted against,
	// and it is empty for work nothing is counted against.
	Caller routing.Caller
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
	// InputModalities restrict candidates to models that accept those extra input kinds.
	InputModalities []string
}

// Router selects an LLM provider and opens sessions.
type Router struct {
	*routing.Router[Provider]
	quota *quota.Limiter
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[Provider]{
		Validate: func(provider Provider, config routing.ProviderConfig) error {
			for _, modality := range config.InputModalities {
				if !provider.Capabilities().Accepts(modality) {
					return fmt.Errorf("%s declares unsupported input modality %s", config.Name(), modality)
				}
			}
			return nil
		},
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
	return &Router{Router: core, quota: options.Quota}, nil
}

// Start selects a provider and opens a session, falling back to the next candidate when one
// fails to build. One session answers many turns.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:      request.CustomerID,
		Caller:          request.Caller,
		AgentID:         request.AgentID,
		CallID:          request.CallID,
		Tags:            request.Tags,
		Target:          request.Target,
		LanguageHints:   request.LanguageHints,
		InputModalities: request.InputModalities,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}

	session := newSession(provider, config, core.Owner(), r.Recorder(), r.quota)
	session.fallback = func(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
		candidates, err := r.Resolve(ctx, request.Target, request.LanguageHints)
		if err != nil {
			return nil, err
		}
		var failures []error
		for _, candidate := range candidates {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if candidate.Config.Name() == config.Name() {
				continue
			}
			alternative := core
			alternative.Target = candidate.Config.Name()
			provider, selected, err := r.Select(ctx, alternative)
			if err != nil {
				failures = append(failures, err)
				continue
			}
			// The child serves a response the parent already allowed, so it reaches for
			// create rather than Create: the limit is asked once per response, not once
			// per provider tried. It still holds the limiter, because whichever provider
			// ends up answering is the one whose tokens have to be debited.
			child := newSession(provider, selected, core.Owner(), r.Recorder(), r.quota)
			if !session.addChild(child) {
				_ = child.Close()
				return nil, errors.New("llmrouter: session is closed")
			}
			stream, err := child.create(ctx, params)
			if err != nil {
				session.releaseChild(child)
				failures = append(failures, err)
				continue
			}
			return stream.Observe(func(event llm.Event) {
				if _, done := event.(llm.ResponseCompleted); done {
					session.releaseChild(child)
				}
			}), nil
		}
		return nil, errors.Join(append(failures, errors.New("llmrouter: no fallback provider available"))...)
	}
	return session, nil
}
