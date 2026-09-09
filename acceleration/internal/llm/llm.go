// Package llm defines the minimal large-language-model contract shared by every provider.
//
// The shape follows OpenAI's Responses API: one call creates one response, and the events
// it produces are pulled from the stream that call returns. Anything provider-specific
// stays on the concrete type, which also exposes a Client method returning the underlying
// SDK client so callers are never boxed in by this interface.
//
// Tools are here because an agent on a phone call has to do things the conversation cannot
// do for it: hand the caller to a human, press a digit at a menu. Every provider this
// routes to speaks the OpenAI tool schema, so standardising it turned out to be the small
// question it was once assumed not to be.
package llm

import (
	"context"
	"strings"
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

// Message is one turn of a conversation, and one item of a request's input.
//
// The Responses API models a call and its result as items of their own rather than as
// fields on a message. They are kept on the message here because the providers reached
// over an OpenAI-compatible chat endpoint need that shape anyway, and because the history
// this comes from is held as turns everywhere else in the agent.
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
	// Signature is provider state that has to be handed back with the call when the turn
	// is replayed. Gemini signs the calls it makes and rejects a conversation that
	// returns one unsigned, which is how a tool result gets answered rather than
	// refused. It is opaque and empty for providers that do not sign.
	Signature string
}

// LLM is a streaming large-language-model provider.
//
// Create asks for one response and returns the stream it arrives on. There is no shared
// event channel and nothing to interrupt by name: a response is abandoned by closing its
// own stream, which is what barge-in does.
type LLM interface {
	Create(ctx context.Context, params ResponseParams) (*Stream, error)

	// Provider is the stable provider name used in stats, e.g. "deepseek".
	Provider() string
	// Model is the model identifier used in stats, e.g. "DeepSeek-V4-Flash-0731".
	Model() string
	// Capabilities is what this model accepts, so a caller can ask for reasoning or
	// persistence only where they mean something.
	Capabilities() Capabilities
	// Close abandons everything in flight and releases the provider.
	Close() error
}

// Unfence strips the code fence a model puts around JSON it was asked for.
//
// Asking for JSON only mostly works, and a fenced object is the way it mostly fails. The
// answer is otherwise exactly what was asked for, so it is read rather than rejected.
func Unfence(answer string) string {
	trimmed := strings.TrimSpace(answer)
	trimmed = strings.TrimPrefix(trimmed, "```json")
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimSuffix(trimmed, "```")
	return strings.TrimSpace(trimmed)
}
