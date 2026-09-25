package xai

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

func grok(t *testing.T) (*httptest.Server, chan map[string]any) {
	sent := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		sent <- body
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server, sent
}

func TestGrokThinksAtLowUnlessAskedOtherwise(t *testing.T) {
	server, sent := grok(t)
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL})
	require.NoError(t, err)
	defer provider.Close()

	stream, err := provider.Create(t.Context(), llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Content: "hi"}}})
	require.NoError(t, err)
	response, err := llm.Collect(stream)
	require.NoError(t, err)

	require.Equal(t, "hi", response.OutputText)
	body := <-sent
	require.Equal(t, "grok-4.7", body["model"])
	require.Equal(t, "low", body["reasoning_effort"])
}

func TestANamedEffortIsSent(t *testing.T) {
	server, sent := grok(t)
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL})
	require.NoError(t, err)
	defer provider.Close()

	stream, err := provider.Create(t.Context(), llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "think hard"}},
		Reasoning: llm.ReasoningParams{Effort: "xhigh"},
	})
	require.NoError(t, err)
	_, err = llm.Collect(stream)
	require.NoError(t, err)

	require.Equal(t, "xhigh", (<-sent)["reasoning_effort"])
}

func TestNoneIsRefusedBeforeItIsSent(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
	}))
	defer server.Close()
	provider, err := New(Options{APIKey: "test", BaseURL: server.URL})
	require.NoError(t, err)
	defer provider.Close()

	_, err = provider.Create(t.Context(), llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "no thinking"}},
		Reasoning: llm.ReasoningParams{Effort: "none"},
	})

	require.ErrorContains(t, err, "is not one of")
	require.Zero(t, requests.Load())
}
