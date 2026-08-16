// Package llm defines the minimal large-language-model contract shared by every provider.
//
// Only the pieces the router actually needs are standardised: messages in, streamed text
// out, barge-in, and identity. Anything provider-specific stays on the concrete type,
// which also exposes a Client method returning the underlying SDK client so callers are
// never boxed in by this interface.
//
// Tool calling is deliberately absent. Standardising a tool schema across providers is a
// larger question than routing needs answered, and the Client escape hatch reaches the
// provider's own tool support in the meantime.
package llm

import (
	"context"
)

// Role is who authored a message.
type Role string

const (
	// System carries the agent's instructions.
	System Role = "system"
	// User is what the person said.
	User Role = "user"
	// Assistant is what the model said on an earlier turn.
	Assistant Role = "assistant"
)

// Message is one turn of a conversation.
type Message struct {
	Role    Role
	Content string
}

// Request is one completion to generate.
//
// The whole conversation is passed every time rather than held by the provider, because
// routing may send consecutive turns to different providers and a conversation that lives
// in the caller survives a failover.
type Request struct {
	// ID correlates every event belonging to one completion. Providers generate one when
	// it is empty.
	ID string
	// Instructions is the system prompt. It is separate from Messages so a caller cannot
	// forget it on a retry, and providers prepend it themselves.
	Instructions string
	// Messages is the conversation so far, oldest first, ending with what to respond to.
	Messages []Message
	// MaxTokens caps the response. Zero leaves the provider's own default in place.
	MaxTokens int
	// Temperature controls randomness. Nil leaves the provider's own default in place,
	// which is not the same as zero.
	Temperature *float64
}

// LLM is a streaming large-language-model provider.
//
// Start prepares the provider, Respond asks for a completion, and Events carries text
// deltas and completion boundaries back. Events is closed by Close.
type LLM interface {
	Start(ctx context.Context) error
	Respond(request Request) error
	// Interrupt abandons the named completions, or every completion in flight when given
	// none, so the agent can stop mid-sentence when the user starts talking. Naming them
	// is what lets one caller run several completions at once and abandon only the one
	// whose premise has gone stale. It is not an error to interrupt when nothing is
	// running.
	Interrupt(completionIDs ...string) error
	Events() <-chan Event
	Close() error

	// Provider is the stable provider name used in stats, e.g. "deepseek".
	Provider() string
	// Model is the model identifier used in stats, e.g. "DeepSeek-V4-Flash-0731".
	Model() string
	// Reasoning reports whether the model emits ReasoningDelta events before its answer.
	// A caller on the live path uses it to decide whether to wait for the first
	// TextDelta or to show thinking in the meantime.
	Reasoning() bool
}
