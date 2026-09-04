// Package sttrouter routes speech-to-text traffic and records what each turn cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the speech-to-text shape: which providers
// exist, what a session does with audio, and which events count as a unit of work.
package sttrouter

import (
	"context"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Registry is the set of speech-to-text providers a build can construct.
type Registry = routing.Registry[stt.STT]

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	// Transcribers is the batch half, for NewRecordings. The two registries are separate
	// because a vendor's recording model is not its streaming one.
	Transcribers *Transcribers
	Store        *store.Store
	Live         *live.Client
	Logger       *slog.Logger
}

// Request is what a caller wants transcribed.
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
	// Keyterms are the business-specific words the transcriber should expect. A provider
	// that cannot be told about vocabulary ignores them.
	Keyterms []string
	// Options is the rest of what the caller asked for: partials, endpointing,
	// diarization, redaction. A term named here narrows the candidates to the models that
	// declared it, so it is either honoured or the request is refused.
	Options options.STT
}

// Router selects a speech-to-text provider and opens transcription sessions.
type Router struct {
	*routing.Router[stt.STT]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[stt.STT]{
		Modality: routing.STT,
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
// one fails to start.
func (r *Router) Start(ctx context.Context, request Request) (*Session, error) {
	core := routing.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        request.Target,
		LanguageHints: request.LanguageHints,
		Keyterms:      request.Keyterms,
		Terms:         request.Options.Terms(),
		STT:           request.Options,
	}
	provider, config, err := r.Select(ctx, core)
	if err != nil {
		return nil, err
	}
	return newSession(provider, config, core.Owner(), r.Recorder()), nil
}
