// Package stsrouter routes speech-to-speech traffic and records what each reply cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the speech-to-speech shape: which providers
// exist, what a session does with the caller's audio, and which events count as a unit of
// work.
package stsrouter

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// Registry is the set of speech-to-speech providers a build can construct.
type Registry = routing.Registry[sts.STS]

// streaming is what every provider here is, for a priority list that names a vendor rather
// than one of their models. There is no batch half of a conversation.
var streaming = true

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
//
// There is no voice resolver on purpose. None of these models takes a cloned voice, so a
// customer's own voice asked for here is refused up front rather than looked up at every
// candidate and found at none of them.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Logger   *slog.Logger
}

// Request is what a caller wants a conversation held by.
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
	// LanguageHints narrow multilingual models.
	LanguageHints []string
	// Tools are what the model may call. They are given as the session opens because
	// some models take them then and never again.
	Tools []llm.Tool
	// Options is the rest of what the caller asked for: a voice, instructions, a turn
	// detector. A term named here narrows the candidates to the models that declared it,
	// so it is either honoured or the request is refused.
	Options options.STS
}

// Router selects a speech-to-speech provider and opens conversations.
type Router struct {
	*routing.Router[sts.STS]
}

// New validates the options and returns a Router.
//
// Beyond what every router checks, the config is held to what the code behind each model
// can send: a model declared as calling tools whose provider package cannot is refused here,
// at boot, rather than at the first call that asks. The check runs again on every session
// against the provider actually built, for a provider this build does not know by name.
func New(options Options) (*Router, error) {
	for _, provider := range options.Config.Providers {
		capabilities, known := capabilitiesFor(provider.Provider, provider.Model)
		if !known {
			continue
		}
		if err := declared(capabilities, provider); err != nil {
			return nil, fmt.Errorf("stsrouter: %w", err)
		}
	}

	core, err := routing.New(routing.Options[sts.STS]{
		Validate: func(provider sts.STS, config routing.ProviderConfig) error {
			return declared(provider.Capabilities(), config)
		},
		Modality: routing.STS,
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

// Start selects a provider and opens a conversation, falling back to the next candidate
// when one fails to start. One session holds many turns.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:      request.CustomerID,
		AgentID:         request.AgentID,
		CallID:          request.CallID,
		Tags:            request.Tags,
		Target:          request.Target,
		Providers:       request.Options.Providers,
		Realtime:        &streaming,
		LanguageHints:   request.LanguageHints,
		InputModalities: request.Options.InputModalities(),
		Voice:           request.Options.Voice,
		Tools:           request.Tools,
		Terms:           request.Options.Terms(),
		DataPolicy:      request.Options.DataPolicy,
		STS:             request.Options,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}

// declared reports the first thing a config promises of a model that the code behind it
// cannot deliver. A term declared and not sent is the one thing terms exist to prevent.
func declared(capabilities sts.Capabilities, config routing.ProviderConfig) error {
	for _, term := range config.Terms {
		if !capabilities.Expresses(term) {
			return fmt.Errorf("%s declares %s, which its provider cannot express", config.Name(), term)
		}
	}
	for _, modality := range config.InputModalities {
		if !capabilities.Accepts(modality) {
			return fmt.Errorf("%s declares unsupported input modality %s", config.Name(), modality)
		}
	}
	return nil
}
