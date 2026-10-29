package agents

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
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

func TestACallAnsweredIsInTheAgentsTimezone(t *testing.T) {
	router := newWorked(t, nil)
	agent, err := New(Options{Name: "jean", Timezone: "Europe/Lisbon", LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: router.URL, CustomerID: "acme"},
	})})
	require.NoError(t, err)

	session, err := agent.Answer(t.Context(), InboundCall{SessionID: "session-a", CalledNumber: "+15125551234"})
	require.NoError(t, err)
	defer session.Close(context.WithoutCancel(t.Context()))

	require.Equal(t, "Europe/Lisbon", zoneOf(t, router))
}

func TestACallPlacedIsInTheAgentsTimezone(t *testing.T) {
	router := newWorked(t, nil)
	// The stand-in router opens sessions but places no calls, so placing one is answered in
	// front of it and everything else is passed through, the session's socket included.
	placed := "session-a"
	placing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/v1/phone/calls" {
			reply(w, http.StatusAccepted, acceleration.PlacedCall{SessionId: &placed, Status: "queued", VendorCallId: "CA1"})
			return
		}
		router.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(placing.Close)
	agent, err := New(Options{Name: "jean", Timezone: "Europe/Lisbon", LLM: stream.Accelerated(stream.Config{
		Backend: stream.Backend{URL: placing.URL, CustomerID: "acme"},
	})})
	require.NoError(t, err)

	session, err := agent.StartCall(t.Context(), "+15125551234", "+351210000000")
	require.NoError(t, err)
	defer session.Close(context.WithoutCancel(t.Context()))

	require.Equal(t, "Europe/Lisbon", zoneOf(t, router))
}
