//go:build integration

package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

func TestMetaSocketLiveToolReplay(t *testing.T) {
	if os.Getenv("META_API_KEY") == "" {
		t.Skip("META_API_KEY not set")
	}
	router, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{
		Provider: "meta", Model: "muse-spark-1.3", Languages: []string{"en"}, Realtime: true, Tier: routing.HighQuality,
	}}}, Registry: llmrouter.DefaultRegistry()})
	require.NoError(t, err)
	t.Cleanup(router.Close)
	server, err := NewServer(Options{Routers: map[routing.Modality]routing.Inspector{routing.LLM: router}, Streams: &Streams{LLM: router}})
	require.NoError(t, err)
	endpoint := httptest.NewServer(server.Handler())
	t.Cleanup(endpoint.Close)
	connection, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(endpoint.URL, "http")+"/v1/llm/stream", http.Header{CustomerHeader: []string{"athena-integration"}})
	require.NoError(t, err)
	t.Cleanup(func() { connection.Close() })
	require.NoError(t, connection.SetReadDeadline(time.Now().Add(90*time.Second)))
	require.NoError(t, connection.WriteJSON(frame{"type": "start", "target": "meta/muse-spark-1.3"}))
	var ready frame
	require.NoError(t, connection.ReadJSON(&ready))
	require.Equal(t, "started", ready["type"])
	require.Equal(t, true, ready["tool_history"])
	messages := []frame{{"role": "user", "content": "Use lookup_project to retrieve the status of athena. Do not guess."}}
	tools := []frame{{"name": "lookup_project", "description": "Get the exact status of a project", "parameters": frame{"type": "object", "properties": frame{"project": frame{"type": "string"}}, "required": []string{"project"}}}}
	ask := func(id string) frame {
		t.Helper()
		require.NoError(t, connection.WriteJSON(frame{"type": "respond", "id": id, "messages": messages, "tools": tools, "max_output_tokens": 1024, "reasoning_effort": "minimal"}))
		for {
			var event frame
			require.NoError(t, connection.ReadJSON(&event))
			require.NotEqual(t, "error", event["type"], "%v", event)
			if event["type"] == "complete" {
				require.Equal(t, "completed", event["status"])
				return event
			}
		}
	}
	first := ask("tool-turn")
	calls, ok := first["tool_calls"].([]any)
	require.True(t, ok)
	require.Len(t, calls, 1)
	call, ok := calls[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "lookup_project", call["name"])
	messages = append(messages, frame{"role": "assistant", "content": first["text"], "tool_calls": calls}, frame{"role": "tool", "tool_call_id": call["id"], "content": "The status is amber-742. Return this exact status to the user."})
	second := ask("result-turn")
	require.Contains(t, strings.ToLower(second["text"].(string)), "amber-742")
	t.Log("Muse tool call and correlated result replay passed through the real WebSocket handler")
}
