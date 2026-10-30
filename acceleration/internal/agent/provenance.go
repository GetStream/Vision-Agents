package agent

import (
	"context"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// Provenance is what led the model to a tool call: whether a scheduled task fired the turn,
// and what it had read since the person last spoke or wrote. A tool whose effect outlives the
// conversation can refuse a call only text somebody else wrote could have asked for.
type Provenance struct {
	// Scheduled is a call in a turn a scheduled task fired, or in a round answering its tools.
	Scheduled bool
	// ToolsRead names, once each, the tools whose results the model had read since the
	// person's last turn.
	ToolsRead []string
	// Delegated is a call in a turn the answer of delegated work started.
	Delegated bool
}

type provenanceKey struct{}

// WithProvenance says what led the model to the tool call ctx runs.
func WithProvenance(ctx context.Context, provenance Provenance) context.Context {
	return context.WithValue(ctx, provenanceKey{}, provenance)
}

// ProvenanceOf is what led the model to the tool call ctx runs.
func ProvenanceOf(ctx context.Context) Provenance {
	provenance, _ := ctx.Value(provenanceKey{}).(Provenance)
	return provenance
}

// toolsReadSince names the tools whose results history holds after its last turn of the
// person's, each once, in the order they came back.
func toolsReadSince(history []llm.Message) []string {
	start := len(history)
	for start > 0 && history[start-1].Role != llm.User {
		start--
	}
	called := map[string]string{}
	var read []string
	for _, message := range history[start:] {
		for _, call := range message.ToolCalls {
			called[call.ID] = call.Name
		}
		if message.Role != llm.ToolResult {
			continue
		}
		if name := called[message.ToolCallID]; name != "" && !slices.Contains(read, name) {
			read = append(read, name)
		}
	}
	return read
}
