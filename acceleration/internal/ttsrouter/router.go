// Package ttsrouter routes text-to-speech traffic and records what each utterance cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the text-to-speech shape: which providers
// exist, what a session does with text, and which events count as a unit of work.
package ttsrouter

import (
	"context"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// Registry is the set of text-to-speech providers a build can construct.
type Registry = routing.Registry[tts.TTS]

// Options configures a Router. Store, Live and Voices are optional: without them the
// router still routes, it just stops recording and stops resolving custom voices.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	// Recorders is the batch half, for NewRecordings. The two registries are separate
	// because a streaming voice and a voice that returns a file are different endpoints
	// at the same vendor.
	Recorders *Recorders
	Store     *store.Store
	Live      *live.Client
	Voices    routing.VoiceResolver
	Logger    *slog.Logger
}

// Request is what a caller wants said.
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
	// Voice selects the speaker. Its meaning is the provider's, since a voice id is not
	// portable between them.
	Voice string
	// Options is the rest of what the caller asked for: a speed, an emotion, an output
	// format. A term named here narrows the candidates to the voices that declared it, so
	// it is either honoured or the request is refused.
	Options options.TTS
}

// Router selects a text-to-speech provider and opens synthesis sessions.
type Router struct {
	*routing.Router[tts.TTS]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[tts.TTS]{
		Modality: routing.TTS,
		Config:   options.Config,
		Registry: options.Registry,
		Store:    options.Store,
		Live:     options.Live,
		Voices:   options.Voices,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Router{Router: core}, nil
}

// Start selects a provider and opens a session, falling back to the next candidate when
// one fails to start. One session says many things.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        request.Target,
		LanguageHints: request.LanguageHints,
		Voice:         request.Voice,
		Terms:         request.Options.Terms(),
		TTS:           request.Options,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}
