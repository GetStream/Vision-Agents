package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

func TestLLMSocketToolHistory(t *testing.T) {
	var command respond
	err := json.Unmarshal([]byte(`{"type":"respond","id":"turn-2","messages":[
		{"role":"assistant","content":"Checking","tool_calls":[{"id":"call-1","name":"read","arguments":"{\"path\":\"report.txt\"}","signature":"opaque"}]},
		{"role":"tool","tool_call_id":"call-1","content":"report contents"}
	]}`), &command)
	require.NoError(t, err)
	params, err := command.params(options.LLM{})
	require.NoError(t, err)
	require.Equal(t, []llm.ToolCall{{ID: "call-1", Name: "read", Arguments: `{"path":"report.txt"}`, Signature: "opaque"}}, params.Input[0].ToolCalls)
	require.Equal(t, "Checking", params.Input[0].Content)
	require.Equal(t, "call-1", params.Input[1].ToolCallID)
	require.Equal(t, "report contents", params.Input[1].Content)
}

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
