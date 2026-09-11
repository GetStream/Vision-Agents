package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/stretchr/testify/require"
)

func TestToolArgumentsKeepTheirOutputItemIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(event map[string]any) {
			b, _ := json.Marshal(event)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event["type"], b)
		}
		// Output index zero is a text item. Added events identify calls inside item.id;
		// argument deltas carry that ID at the top level as item_id.
		for i, name := range []string{"first", "second"} {
			id := fmt.Sprint("fc", i)
			send(map[string]any{"type": "response.output_item.added", "output_index": i + 1, "item": map[string]any{"type": "function_call", "id": id, "call_id": "call" + id, "name": name}})
		}
		for i := range 2 {
			send(map[string]any{"type": "response.function_call_arguments.delta", "output_index": i + 1, "item_id": fmt.Sprint("fc", i), "delta": fmt.Sprintf(`{"number":%d}`, i)})
		}
		send(map[string]any{"type": "response.completed", "response": map[string]any{"id": "response", "status": "completed"}})
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL})
	require.NoError(t, err)
	defer provider.Close()
	stream, err := provider.Create(t.Context(), llm.ResponseParams{ID: "question", Input: []llm.Message{{Role: llm.User, Content: "two tools"}}})
	require.NoError(t, err)
	result, err := llm.Collect(stream)
	require.NoError(t, err)
	require.Len(t, result.ToolCalls, 2)
	require.Equal(t, "first", result.ToolCalls[0].Name)
	require.Equal(t, `{"number":0}`, result.ToolCalls[0].Arguments)
	require.Equal(t, "second", result.ToolCalls[1].Name)
	require.Equal(t, `{"number":1}`, result.ToolCalls[1].Arguments)
}
