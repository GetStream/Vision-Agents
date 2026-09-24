package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestAnImageReachesClaude(t *testing.T) {
	sent := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		sent <- body
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a rose\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL, Model: "claude-opus-5-5"})
	require.NoError(t, err)
	defer provider.Close()

	stream, err := provider.Create(t.Context(), llm.ResponseParams{Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Text: "what flower"},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}}},
		},
	}}})
	require.NoError(t, err)
	response, err := llm.Collect(stream)
	require.NoError(t, err)

	require.Equal(t, "a rose", response.OutputText)
	body := <-sent
	require.Equal(t, "claude-opus-5-5", body["model"])
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	require.Equal(t, "image_url", content[1].(map[string]any)["type"])
}

func TestAReasoningEffortIsRefusedBeforeItIsSent(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL, Model: "claude-opus-5-5"})
	require.NoError(t, err)
	defer provider.Close()

	_, err = provider.Create(t.Context(), llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "think hard"}},
		Reasoning: llm.ReasoningParams{Effort: "high"},
	})

	require.ErrorContains(t, err, "does not reason")
	require.Zero(t, requests.Load())
}
