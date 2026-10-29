package agents

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func TestWaitingForCallsOnANumberWithAnAgentFailsAtOnce(t *testing.T) {
	var mu sync.Mutex
	var requests []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests = append(requests, r.Method+" "+r.URL.EscapedPath())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"e164": "+15125551234", "agent_config_id": "cfg_1"}`))
	}))
	t.Cleanup(server.Close)
	agent, err := New(Options{Name: "jean", LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: server.URL, CustomerID: "acme"},
	})})
	require.NoError(t, err)

	_, err = agent.WaitForCall(t.Context(), "+15125551234")
	require.EqualError(t, err, "agents: +15125551234 is answered by agent config cfg_1; remove the agent to answer it with a worker")
	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, []string{"POST /v1/phone/numbers/+15125551234/attach"}, requests)
}
