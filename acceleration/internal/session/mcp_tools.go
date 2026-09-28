package session

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
)

// connectorToolRunner runs prefixed MCP tools itself and hands everything else to the caller bridge.
type connectorToolRunner struct {
	mcp  *mcp.Runtime
	next agent.ToolRunner
}

func (r *connectorToolRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if r.mcp != nil && r.mcp.Owns(call.Name) {
		text, err := r.mcp.Call(ctx, call)
		return llm.TextParts(text), err
	}
	if r.next != nil {
		return r.next.Run(ctx, call)
	}
	return nil, errUnknownTool(call.Name)
}

func errUnknownTool(name string) error {
	return &toolError{name: name}
}

type toolError struct{ name string }

func (e *toolError) Error() string {
	return "session: " + e.name + " is not a tool this session can run"
}
