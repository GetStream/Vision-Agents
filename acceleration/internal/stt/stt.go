// Package stt defines the minimal speech-to-text contract shared by every provider.
//
// Only the pieces the router actually needs are standardised: audio in, transcript
// revisions out, plus identity and lifecycle. Anything provider-specific stays
// on the concrete type, which also exposes a Client method returning the underlying
// SDK client so callers are never boxed in by this interface.
package stt

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// SampleRate is the only rate the providers accept. LiveKit decodes to this for us.
const SampleRate = 16_000

// MaxKeyterms is how many terms a provider will be told about. It is the lowest limit
// among the providers that accept any, so a list under it works everywhere.
const MaxKeyterms = 100

// CleanKeyterms drops the blanks and the surrounding space from a caller's list, so a
// term nobody meant to add does not take up one of the places a provider allows.
func CleanKeyterms(terms []string) []string {
	if len(terms) == 0 {
		return nil
	}
	kept := make([]string, 0, len(terms))
	for _, term := range terms {
		if trimmed := strings.TrimSpace(term); trimmed != "" {
			kept = append(kept, trimmed)
		}
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

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
