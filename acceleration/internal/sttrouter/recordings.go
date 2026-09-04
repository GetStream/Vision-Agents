package sttrouter

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
)

// Transcribers is the set of batch transcription providers a build can construct.
type Transcribers = routing.Registry[stt.Transcriber]

// NewTranscriberRegistry returns an empty registry.
func NewTranscriberRegistry() *Transcribers { return routing.NewRegistry[stt.Transcriber]() }

// DefaultTranscriberRegistry returns a registry with every batch transcriber this build
// supports.
//
// It is a separate registry from the streaming one because a vendor's batch model is a
// different model: Deepgram's live path is Flux, whose job is deciding where a turn ended
// while somebody is still talking, and its recording path is Nova, which has the whole
// file in front of it.
func DefaultTranscriberRegistry() *Transcribers {
	registry := NewTranscriberRegistry()

	registry.Register(deepgram.ProviderName, func(spec routing.Spec) (stt.Transcriber, error) {
		return deepgram.NewPrerecorded(deepgram.PrerecordedOptions{
			Model:  spec.Model,
			Logger: spec.Logger,
		})
	})

	return registry
}

// Recordings routes whole recordings to a batch transcriber.
type Recordings struct {
	*routing.Router[stt.Transcriber]
}

// NewRecordings validates the options and returns a router.
func NewRecordings(options Options) (*Recordings, error) {
	core, err := routing.New(routing.Options[stt.Transcriber]{
		Modality: routing.STT,
		Config:   options.Config,
		Registry: options.Transcribers,
		Store:    options.Store,
		Live:     options.Live,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Recordings{Router: core}, nil
}

// Recording is a whole recording to transcribe and how to bill it.
type Recording struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// Tags are the customer's own cost labels.
	Tags routing.Tags
	// Options are what the caller asked for, target included.
	Options options.STT
	// Source is the recording itself.
	Source stt.Recording
}

// Transcribe selects a provider and transcribes the whole recording, falling back to the
// next candidate when one fails to start.
//
// Unlike a live session this records its own stat row: there is one unit of work rather
// than a turn at a time, and it is finished by the time this returns.
func (r *Recordings) Transcribe(
	ctx context.Context,
	recording Recording,
) (stt.Transcription, routing.ProviderConfig, error) {
	request := routing.Request{
		CustomerID:    recording.CustomerID,
		Tags:          recording.Tags,
		Target:        recording.Options.Target,
		LanguageHints: recording.Options.Languages,
		Keyterms:      recording.Options.Keyterms,
		Terms:         recording.Options.Terms(),
		STT:           recording.Options,
	}

	provider, config, err := r.Select(ctx, request)
	if err != nil {
		return stt.Transcription{}, routing.ProviderConfig{}, err
	}
	defer provider.Close()

	startedAt := time.Now()
	transcription, err := provider.Transcribe(ctx, recording.Source)
	stat := routing.Stat{
		Owner:     request.Owner(),
		StartedAt: startedAt.UTC(),
		LatencyMs: routing.MsSince(startedAt),
		Success:   err == nil,
		Usage:     routing.Usage{AudioMs: transcription.AudioDurationMs},
	}
	if err != nil {
		stat.ErrorCode = "transcribe_failed"
	}
	r.Recorder().Record(config, stat)
	if err != nil {
		return stt.Transcription{}, config, err
	}
	return transcription, config, nil
}
