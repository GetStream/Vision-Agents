package openaicompat

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/stretchr/testify/assert"
)

func TestResultsAreMovedStraightAfterTheirCall(t *testing.T) {
	call := llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "confirm_caller"}}}
	spoke := llm.Message{Role: llm.User, Content: "New member ID is LMN204816."}
	result := llm.Message{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":true}`}

	assert.Equal(t, []llm.Message{call, result, spoke}, StrictToolOrder([]llm.Message{call, spoke, result}))
	assert.Equal(t, []llm.Message{call, result, spoke}, StrictToolOrder([]llm.Message{call, result, spoke}),
		"an order that is already right is left alone")
}

func TestACallWithNoResultIsAnsweredAndAnOrphanResultBecomesANote(t *testing.T) {
	call := llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1"}, {ID: "c2"}}}
	first := llm.Message{Role: llm.ToolResult, ToolCallID: "c1", Content: "ok"}
	orphan := llm.Message{Role: llm.ToolResult, ToolCallID: "c0", Content: "earlier"}

	got := StrictToolOrder([]llm.Message{orphan, call, first})
	assert.Equal(t, llm.User, got[0].Role)
	assert.Equal(t, "Tool result: earlier", got[0].Content)
	assert.Equal(t, []llm.Message{call, first, {Role: llm.ToolResult, ToolCallID: "c2", Content: stoppedResult}}, got[1:])
}
