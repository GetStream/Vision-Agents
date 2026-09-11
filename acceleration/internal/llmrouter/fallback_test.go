package llmrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestRequestFailureFallsBackThroughAccelerate(t *testing.T) {
	var primaryStatus atomic.Int32
	primaryStatus.Store(503)
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(int(primaryStatus.Load()))
		fmt.Fprint(w, `{"error":{"message":"primary unavailable","type":"server_error"}}`)
	}))
	defer primary.Close()
	requests := make(chan map[string]any, 4)
	backup := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		requests <- body
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"fallback\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Opus answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n")
	}))
	defer backup.Close()
	registry := NewRegistry()
	registry.Register("openai", func(spec routing.Spec) (Provider, error) {
		return Started(openai.New(openai.Options{APIKey: "test", BaseURL: primary.URL, Model: spec.Model}))
	})
	registry.Register("anthropic", func(spec routing.Spec) (Provider, error) {
		return Started(anthropic.New(anthropic.Options{APIKey: "test", BaseURL: backup.URL, Model: spec.Model}))
	})
	router, err := New(Options{Registry: registry, Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{Provider: "openai", Model: "gpt-5.6-luna", Languages: []string{"en"}}, {Provider: "anthropic", Model: "claude-opus-5", Languages: []string{"en"}}}, Aliases: map[string]routing.Alias{"docs-support": {Prefer: "openai/gpt-5.6-luna", Only: []string{"openai/gpt-5.6-luna", "anthropic/claude-opus-5"}}}}})
	require.NoError(t, err)
	defer router.Close()
	session, err := router.Start(t.Context(), Request{CustomerID: "customer", Target: "docs-support", Tags: routing.Tags{"project": "docs", "user_id": "user", "organization_id": "org"}})
	require.NoError(t, err)
	defer session.Close()
	require.Equal(t, "openai", session.Provider())
	params := llm.ResponseParams{Instructions: "Be helpful", Input: []llm.Message{{Role: llm.User, Content: "Remember our earlier context"}}, Tools: []llm.Tool{{Name: "search_docs", Parameters: map[string]any{"type": "object"}}}}
	stream, err := session.Create(t.Context(), params)
	require.NoError(t, err)
	response, err := llm.Collect(stream)
	require.NoError(t, err)
	require.Equal(t, "Opus answer", response.OutputText)
	require.Equal(t, "anthropic", response.Provider)
	require.Equal(t, "claude-opus-5", response.Model)
	require.Equal(t, int64(12), response.Usage.InputTokens)
	request := <-requests
	require.Equal(t, "claude-opus-5", request["model"])
	require.Len(t, request["messages"], 2)
	require.Len(t, request["tools"], 1)
	require.Empty(t, session.children)
	// A caller cancellation and invalid request must not become a second billable request.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = session.Create(ctx, params)
	require.Error(t, err)
	primaryStatus.Store(400)
	_, err = session.Create(t.Context(), params)
	require.Error(t, err)
	require.Empty(t, requests)
}
