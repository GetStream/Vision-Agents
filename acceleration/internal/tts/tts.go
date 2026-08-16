// Package tts defines the minimal text-to-speech contract shared by every provider.
//
// Only the pieces the router actually needs are standardised: text in, PCM audio out,
// barge-in, and identity. Anything provider-specific stays on the concrete type, which
// also exposes a Client method returning the underlying connection or SDK client so
// callers are never boxed in by this interface.
package tts

import (
	"context"
)

// Request is one thing to say.
//
// A streaming provider is fed a sentence in pieces: several requests sharing an ID, the
// last with Final set. A non-streaming provider gets one final request per utterance.
type Request struct {
	// ID correlates every event belonging to one synthesis. Providers generate one when
	// it is empty.
	ID string
	// Text is what to say, or the next piece of it.
	Text string
	// Voice overrides the session's voice. Providers that bind a voice per connection
	// report an error rather than quietly saying it in the wrong one.
	Voice string
	// Language overrides the session's language. Empty lets the model infer it.
	Language string
	// Final closes the utterance. It is false for a partial text delta.
	Final bool
}

// TTS is a streaming text-to-speech provider.
//
// Start opens the upstream connection, Synthesize feeds it text, and Events carries audio
// and synthesis boundaries back. Events is closed by Close.
type TTS interface {
	Start(ctx context.Context) error
	Synthesize(request Request) error
	// Interrupt drops queued and in-flight audio so the agent can stop mid-sentence when
	// the user starts talking. It is not an error to interrupt when nothing is playing.
	Interrupt() error
	Events() <-chan Event
	Close() error

	// Provider is the stable provider name used in stats, e.g. "elevenlabs".
	Provider() string
	// Model is the model identifier used in stats, e.g. "eleven_flash_v2_5".
	Model() string
	// Streaming reports whether the provider accepts partial text deltas. When false, a
	// caller must buffer a sentence and send it as one final request.
	Streaming() bool
}
