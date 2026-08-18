// Package stt defines the minimal speech-to-text contract shared by every provider.
//
// Only the pieces the router actually needs are standardised: audio in, transcript
// revisions out, plus identity and lifecycle. Anything provider-specific stays
// on the concrete type, which also exposes a Client method returning the underlying
// SDK client so callers are never boxed in by this interface.
package stt

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// SampleRate is the only rate the providers accept. LiveKit decodes to this for us.
const SampleRate = 16_000

// Mode describes how a transcript relates to the ones before it.
type Mode string

const (
	// ModeDelta means the text appends to what came before.
	ModeDelta Mode = "delta"
	// ModeReplacement means the text replaces the current in-progress transcript.
	ModeReplacement Mode = "replacement"
	// ModeFinal means the turn is settled and the text will not change.
	ModeFinal Mode = "final"
)

// Participant identifies who is speaking.
type Participant struct {
	ID     string
	UserID string
	Name   string
}

// PcmData is a chunk of signed 16-bit PCM audio. Callers validate it against SampleRate.
type PcmData = audio.PcmData

// STT is a streaming speech-to-text provider.
//
// Start opens the upstream connection, ProcessAudio feeds it, and Events carries
// transcript revisions back. Events is closed by Close.
type STT interface {
	Start(ctx context.Context) error
	ProcessAudio(pcm PcmData, participant Participant) error
	Events() <-chan Event
	Close() error

	// Provider is the stable provider name used in stats, e.g. "deepgram".
	Provider() string
	// Model is the model identifier used in stats, e.g. "flux-general-en".
	Model() string
}
