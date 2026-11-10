package hosttest

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// Noted is a conversation with a system message on both sides of the caller's turn, the
// shape the harness sends when it has something to tell the model about one reply.
func Noted() llm.ResponseParams {
	return llm.ResponseParams{Input: []llm.Message{
		{Role: llm.System, Content: "Summary so far."},
		{Role: llm.User, Content: "hi"},
		{Role: llm.System, Content: "Your colleague has come back."},
	}}
}

// SentRoles is the role of every message a provider sends upstream for params. build makes
// the provider against the endpoint it is given, which answers with one short reply.
func SentRoles(t *testing.T, build func(baseURL string) (*openaicompat.LLM, error), params llm.ResponseParams) []string {
	t.Helper()
	var roles []string
	messages, _ := Sent(t, build, params)["messages"].([]any)
	for _, message := range messages {
		role, _ := message.(map[string]any)["role"].(string)
		roles = append(roles, role)
	}
	return roles
}

// Sent is the body of the first request a provider sends upstream for params. build makes the
// provider against the endpoint it is given, which answers with one short reply.
func Sent(t *testing.T, build func(baseURL string) (*openaicompat.LLM, error), params llm.ResponseParams) map[string]any {
	t.Helper()
	var (
		mu   sync.Mutex
		sent map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		if sent == nil {
			sent = body
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
			`data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`+"\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	provider, err := build(server.URL)
	require.NoError(t, err)
	t.Cleanup(func() { provider.Close() })
	stream, err := provider.Create(context.Background(), params)
	require.NoError(t, err)
	for stream.Next() {
	}

	mu.Lock()
	defer mu.Unlock()
	return sent
}
