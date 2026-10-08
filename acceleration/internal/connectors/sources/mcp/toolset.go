package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// nonText is what the model reads for a result whose content has no text: an image or a
// resource, which a tool result cannot carry yet. The prototype's wording
// (internal/mcp/mcp.go:285 at cf62af0d).
const nonText = "The tool returned content that is not text, which the agent cannot read"

// Toolset is the granted tools of one MCP server, opened for one session. It is safe for
// concurrent use: its tools are fixed when Open returns, and the SDK's session takes calls at
// once.
type Toolset struct {
	session *mcp.ClientSession
	// timeout bounds one call; zero leaves it to the caller's context.
	timeout time.Duration
	tools   []llm.Tool
	// targets are the offered tools by the name the model calls them.
	targets map[string]target
}

var _ core.Toolset = (*Toolset)(nil)

// target is one offered tool: its name at the server and its input schema.
type target struct {
	name      string
	validator *jsonschema.Schema
}

// Tools are the tools the model is offered.
func (t *Toolset) Tools() []llm.Tool {
	return t.tools
}

// Call runs one offered tool. A name this toolset did not offer, or arguments its input schema
// refuses, never reach the server. A result past core.MaxResultBytes is cut and ends with
// core.TruncatedMarker. A result the server marks isError is a *core.ToolError, whatever its
// content: an error with no text is still an error.
func (t *Toolset) Call(ctx context.Context, call llm.ToolCall) (core.Result, error) {
	offered, found := t.targets[call.Name]
	if !found {
		return core.Result{}, stack.Wrap(fmt.Errorf("mcp: %q is not a tool this connection offers", call.Name))
	}
	raw := strings.TrimSpace(call.Arguments)
	if raw == "" {
		raw = "{}"
	}
	arguments, err := jsonschema.UnmarshalJSON(strings.NewReader(raw))
	if err != nil {
		return core.Result{}, stack.Wrap(fmt.Errorf("mcp: %s: the arguments are not JSON: %w", call.Name, err))
	}
	if err := offered.validator.Validate(arguments); err != nil {
		return core.Result{}, stack.Wrap(fmt.Errorf("mcp: %s: the arguments do not match the tool's input schema", call.Name))
	}
	if t.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, t.timeout)
		defer cancel()
	}
	// The arguments go as the model wrote them, so no number is rounded through a float.
	result, err := t.session.CallTool(ctx, &mcp.CallToolParams{Name: offered.name, Arguments: json.RawMessage(raw)})
	if err != nil {
		return core.Result{}, stack.Wrap(fmt.Errorf("mcp: %s: %w", call.Name, err))
	}
	text, err := resultText(result)
	if err != nil {
		return core.Result{}, err
	}
	if result.IsError {
		if text == "" {
			text = "The tool reported an error and said nothing more"
		}
		return core.Result{}, stack.Wrap(&core.ToolError{Message: text})
	}
	return core.Result{Parts: llm.TextParts(text)}, nil
}

// Close ends the session with the server.
func (t *Toolset) Close() {
	_ = t.session.Close()
}

// resultText is what the model reads of a result: its text parts, one per line; without any,
// its structured content as JSON; without that, nonText when there was content. It is cut to
// core.MaxResultBytes, marker included, at a UTF-8 boundary.
func resultText(result *mcp.CallToolResult) (string, error) {
	var parts []string
	for _, content := range result.Content {
		if text, ok := content.(*mcp.TextContent); ok && text.Text != "" {
			parts = append(parts, text.Text)
		}
	}
	text := strings.Join(parts, "\n")
	if text == "" && result.StructuredContent != nil {
		structured, err := json.Marshal(result.StructuredContent)
		if err != nil {
			return "", stack.Wrap(fmt.Errorf("mcp: structured result: %w", err))
		}
		text = string(structured)
	}
	if text == "" && len(result.Content) > 0 {
		text = nonText
	}
	return core.CutResult(text), nil
}

// offered is a tool as the model is shown it.
func offered(name, description string, schema map[string]any) llm.Tool {
	return llm.Tool{Name: name, Description: description, Parameters: schema}
}
