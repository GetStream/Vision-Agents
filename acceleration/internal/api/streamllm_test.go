package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

func TestLLMSocketRejectsMalformedToolHistory(t *testing.T) {
	for _, message := range []string{
		`{"role":"tool","content":"result"}`,
		`{"role":"user","tool_call_id":"call-1","content":"result"}`,
		`{"role":"user","tool_calls":[{"id":"a","name":"read","arguments":"{}"}]}`,
		`{"role":"assistant","tool_calls":[{"id":"a","name":"read","arguments":"not json"}]}`,
		`{"role":"assistant","tool_calls":[{"name":"read","arguments":"{}"}]}`,
		`{"role":"assistant","tool_calls":[{"id":"a","arguments":"{}"}]}`,
	} {
		t.Run(message, func(t *testing.T) {
			var command respond
			require.NoError(t, json.Unmarshal([]byte(`{"messages":[`+message+`]}`), &command))
			_, err := command.params(options.LLM{})
			require.Error(t, err)
		})
	}
}

func TestLLMSocketToolOnlyHistoryAcceptsEmptyContent(t *testing.T) {
	for _, content := range []string{``, `,"content":null`, `,"content":""`} {
		t.Run(content, func(t *testing.T) {
			var command respond
			require.NoError(t, json.Unmarshal([]byte(`{"messages":[{"role":"assistant"`+content+`,"tool_calls":[{"id":"call-1","name":"read","arguments":"{}"}]}]}`), &command))
			params, err := command.params(options.LLM{})
			require.NoError(t, err)
			require.Equal(t, []llm.ToolCall{{ID: "call-1", Name: "read", Arguments: "{}"}}, params.Input[0].ToolCalls)
		})
	}
}

func TestLLMSocketCompletionPreservesReplayAndStopMetadata(t *testing.T) {
	encoded, ok := llmFrame(llm.ResponseCompleted{Response: llm.Response{
		ID: "r1", Status: llm.StatusIncomplete, IncompleteReason: llm.ReasonMaxOutputTokens,
		ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "read", Arguments: "{}", Signature: "opaque"}},
	}})
	require.True(t, ok)
	require.Equal(t, llm.ReasonMaxOutputTokens, encoded["incomplete_reason"])
	require.Equal(t, "opaque", encoded["tool_calls"].([]frame)[0]["signature"])
}
