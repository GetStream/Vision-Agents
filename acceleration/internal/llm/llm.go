// Package llm defines the minimal large-language-model contract shared by every provider.
//
// Only the pieces the router actually needs are standardised: messages in, streamed text
// out, tool calls, barge-in, and identity. Anything provider-specific stays on the
// concrete type, which also exposes a Client method returning the underlying SDK client so
// callers are never boxed in by this interface.
//
// Tools are here because an agent on a phone call has to do things the conversation cannot
// do for it: hand the caller to a human, press a digit at a menu. Every provider this
// routes to speaks the OpenAI tool schema, so standardising it turned out to be the small
// question it was once assumed not to be.
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
	// ToolResult is what running a tool returned, answering an earlier assistant turn.
	ToolResult Role = "tool"
)

// Message is one turn of a conversation.
type Message struct {
	Role    Role
	Content string
	// ToolCalls are what an assistant turn asked to have run. A turn that called a tool
	// has to be replayed with them, because the provider rejects a tool result that
	// answers nothing.
	ToolCalls []ToolCall
	// ToolCallID names the call a ToolResult message answers.
	ToolCallID string
}

// Tool is something the model may do instead of, or alongside, saying something.
type Tool struct {
	// Name is how the model asks for it.
	Name string
	// Description is what the model is told the tool does, which is the whole of how it
	// decides when to reach for one.
	Description string
	// Parameters is a JSON Schema object describing the arguments. It is untyped because
	// a schema is untyped: the shape is whatever the tool accepts.
	Parameters map[string]any
}

// ToolCall is the model asking for a tool to be run.
type ToolCall struct {
	// ID correlates the call with the result sent back, and is what the provider matches
	// a Tool message against.
	ID string
	// Name is which tool was asked for.
	Name string
	// Arguments is the JSON object the model filled in, left as text because it is the
	// tool that knows what shape to expect. A model may produce arguments that do not
	// parse, and the caller is better placed than this package to say what that means.
	Arguments string
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
	// JSON asks for one JSON object instead of prose. A caller that parses the answer
	// needs it: a model handed a conversation and asked about it will otherwise sometimes
	// carry the conversation on instead.
	JSON bool
	// Temperature controls randomness. Nil leaves the provider's own default in place,
	// which is not the same as zero.
	Temperature *float64
	// Tools the model may call. An empty list sends none, which is what a request that
	// only wants prose wants: a model offered a tool will eventually reach for it.
	Tools []Tool
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
