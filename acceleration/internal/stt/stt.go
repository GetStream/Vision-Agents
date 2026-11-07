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

// SampleRate is the only rate the providers accept. Inbound Opus is decoded to this.
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

// Metering is how a transcriber's work is measured for its bill.
type Metering int

const (
	// MeteredByVendor is a transcriber whose vendor reports on every settled turn exactly
	// what it bills for it, so the turns it reports are the whole bill: audio it took after
	// its last turn, before a refusal or the end of the session, is billed by no row. A turn
	// it says nothing for is billed on the router's count of the audio streamed since the
	// last turn, marked as an estimate, and raises the alarm, since that vendor's adapter is
	// what needs fixing.
	MeteredByVendor Metering = iota
	// MeteredByRouter is a transcriber whose vendor bills every second of audio it is sent and
	// reports none of it, so the audio the router streams to it is the bill: every
	// millisecond, between turns and after the last one included, and never an estimate.
	MeteredByRouter
)

// Metered is a transcriber that says how it is measured. One that does not is
// MeteredByVendor.
type Metered interface {
	Metering() Metering
}

// MeteringOf is how the provider is measured.
func MeteringOf(provider STT) Metering {
	if metered, ok := provider.(Metered); ok {
		return metered.Metering()
	}
	return MeteredByVendor
}

// Retuner is a provider whose vocabulary can change on a stream already open. One that
// cannot be retuned keeps the keyterms it started with.
type Retuner interface {
	// SetKeyterms replaces the terms the transcriber expects, for the rest of the stream.
	SetKeyterms(terms []string) error
}
