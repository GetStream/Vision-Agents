// Package sandbox runs code the model wrote somewhere it cannot do any harm.
//
// It is offered to the subagent rather than to the voice model. Code execution is slow
// and a conversation is not: a model that stops to run something mid-sentence has stopped
// talking, which is the one thing a voice agent may not do. The slower model has already
// left the live path, so it is free to take its time.
package sandbox

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ToolName is what the model asks for when it wants to run something.
const ToolName = "run_code"

// Result is what running a piece of code produced.
type Result struct {
	// Output is everything the code printed. A sandbox runs it the way a shell would, so
	// what went to output and what went to errors arrive interleaved, as a person reading
	// a terminal would see them.
	Output string
	// ExitCode is what the process returned. Zero is success.
	ExitCode int
}

// Sandbox runs code somewhere isolated from everything that matters.
type Sandbox interface {
	// Run executes a piece of Python and returns what it printed. An error means the code
	// could not be run at all; code that ran and failed is a Result with a non-zero exit.
	Run(ctx context.Context, code string) (Result, error)
	// Close releases whatever was held open to run code. Safe to call twice.
	Close() error
}

// Tool describes running code to a model.
func Tool() llm.Tool {
	return llm.Tool{
		Name:        ToolName,
		Description: "Run a short Python program and read back what it printed. Use it for anything you would otherwise have to work out in your head, such as arithmetic, dates or parsing. Print what you want to know.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"code": map[string]any{
					"type":        "string",
					"description": "The Python to run. Print the answer; nothing else comes back.",
				},
			},
			"required": []string{"code"},
		},
	}
}
