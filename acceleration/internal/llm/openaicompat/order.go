package openaicompat

import "github.com/GetStream/Vision-Agents/acceleration/internal/llm"

// stoppedResult answers a call whose result never joined the history, because the caller
// talked over it and it was abandoned.
const stoppedResult = "(no result: the call was stopped before it answered)"

// strictToolOrder puts each tool call's results straight after it. What came between them,
// such as what the caller said while the tool ran, follows the results in its own order. A
// result whose call is not in the history (cut off by a history limit) becomes a note from
// the caller's side, and a call with no result is given one, so every call is answered
// exactly once.
func strictToolOrder(input []llm.Message) []llm.Message {
	results := map[string]int{}
	for i, message := range input {
		if message.Role == llm.ToolResult && message.ToolCallID != "" {
			if _, seen := results[message.ToolCallID]; !seen {
				results[message.ToolCallID] = i
			}
		}
	}
	used := map[int]bool{}
	out := make([]llm.Message, 0, len(input))
	for i, message := range input {
		switch {
		case message.Role == llm.ToolResult:
			if used[i] {
				continue
			}
			out = append(out, llm.Message{Role: llm.User, Content: "Tool result: " + messageText(message)})
		case message.Role == llm.Assistant && len(message.ToolCalls) > 0:
			out = append(out, message)
			for _, call := range message.ToolCalls {
				if at, ok := results[call.ID]; ok && at > i && !used[at] {
					used[at] = true
					out = append(out, input[at])
					continue
				}
				out = append(out, llm.Message{Role: llm.ToolResult, ToolCallID: call.ID, Content: stoppedResult})
			}
		default:
			out = append(out, message)
		}
	}
	return out
}
