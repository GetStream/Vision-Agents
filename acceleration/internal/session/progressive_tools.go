package session

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// summaryLimit caps the description a deferred tool is offered with, in runes.
const summaryLimit = 200

// firstCallNote ends a deferred tool's description, so the model expects its first call to
// come back with instructions rather than a result.
const firstCallNote = "Its first call runs nothing and returns how to use it."

// progressively offers the deferred tools among tools by the first line of what each does
// and its arguments without their descriptions, and puts a runner in front of next that
// answers the first call to each with all of it instead of running it. A server that writes
// a page of examples into a tool's description then costs that page only in the
// conversations that use the tool, at the price of one more model turn when they do.
func progressively(tools, deferred []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
	if len(deferred) == 0 {
		return tools, next
	}
	runner := &progressiveRunner{next: next, full: map[string]harness.Tool{}, described: map[string]bool{}}
	for _, tool := range deferred {
		runner.full[tool.Name] = tool
	}
	offered := slices.Clone(tools)
	for i, tool := range offered {
		if _, ok := runner.full[tool.Name]; ok {
			offered[i] = briefTool(tool)
		}
	}
	return offered, runner
}

// progressiveRunner answers the first call to a deferred tool with how to use it, and
// hands every other call to next.
type progressiveRunner struct {
	next agent.ToolRunner
	// full are the deferred tools as their servers describe them, by name.
	full map[string]harness.Tool

	mu        sync.Mutex
	described map[string]bool
}

func (r *progressiveRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if tool, ok := r.full[call.Name]; ok && r.firstCall(call.Name) {
		return llm.TextParts(howToUse(tool)), nil
	}
	return r.next.Run(ctx, call)
}

func (r *progressiveRunner) firstCall(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.described[name] {
		return false
	}
	r.described[name] = true
	return true
}

func howToUse(tool harness.Tool) string {
	schema, _ := json.Marshal(tool.Parameters)
	return fmt.Sprintf("%s did not run. This is how to use it; call it again with arguments that follow it.\n\n%s\n\nInput schema: %s",
		tool.Name, tool.Description, schema)
}

// briefTool is tool as a deferred one is offered.
func briefTool(tool harness.Tool) harness.Tool {
	tool.Description = strings.TrimSpace(summary(tool.Description) + " " + firstCallNote)
	if tool.Parameters != nil {
		tool.Parameters = bareSchema(tool.Parameters)
	}
	return tool
}

// summary is the first line of a description, cut at a word to summaryLimit runes.
func summary(description string) string {
	var first string
	for line := range strings.Lines(description) {
		if first = strings.TrimSpace(line); first != "" {
			break
		}
	}
	runes := []rune(first)
	if len(runes) <= summaryLimit {
		return first
	}
	cut := string(runes[:summaryLimit])
	if space := strings.LastIndex(cut, " "); space > 0 {
		cut = cut[:space]
	}
	return cut + "…"
}

// bareSchema is a JSON Schema without the words written for a reader: every description,
// title and example, at every depth. What a call has to look like stays: the property
// names, types, enums and what is required.
func bareSchema(schema map[string]any) map[string]any {
	bare := make(map[string]any, len(schema))
	for key, value := range schema {
		switch key {
		case "description", "title", "examples":
			continue
		case "properties", "patternProperties", "$defs", "definitions":
			if named, ok := value.(map[string]any); ok {
				each := make(map[string]any, len(named))
				for name, sub := range named {
					each[name] = bareValue(sub)
				}
				value = each
			}
		case "items", "additionalProperties", "not", "contains", "if", "then", "else":
			value = bareValue(value)
		case "anyOf", "oneOf", "allOf", "prefixItems":
			if list, ok := value.([]any); ok {
				each := make([]any, len(list))
				for i, sub := range list {
					each[i] = bareValue(sub)
				}
				value = each
			}
		}
		bare[key] = value
	}
	return bare
}

func bareValue(value any) any {
	if schema, ok := value.(map[string]any); ok {
		return bareSchema(schema)
	}
	return value
}
