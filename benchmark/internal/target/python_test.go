package target

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPythonKeepsWhatTheAgentMeasuredBeforeItsSessionCloses(t *testing.T) {
	closed := false
	mux := http.NewServeMux()
	mux.HandleFunc("POST /calls/c1/sessions", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"session_id": "s1"})
	})
	mux.HandleFunc("GET /calls/c1/sessions/s1/metrics", func(w http.ResponseWriter, r *http.Request) {
		if closed {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"metrics": map[string]any{
			"stt_latency_ms__avg": 120.5,
			"tts_latency_ms__avg": nil,
		}})
	})
	mux.HandleFunc("DELETE /calls/c1/sessions/s1", func(w http.ResponseWriter, r *http.Request) {
		closed = true
		w.WriteHeader(http.StatusAccepted)
	})
	mux.HandleFunc("GET /calls/c1/sessions/s1", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	agent := &Python{URL: srv.URL}
	stop, err := agent.StartCall(context.Background(), "c1", "default")
	if err != nil {
		t.Fatal(err)
	}
	stop()

	got := agent.AgentMetrics("c1")
	if len(got) != 1 || got["stt_latency_ms__avg"] != 120.5 {
		t.Fatalf("a metric the agent did not measure is not a zero: %+v", got)
	}
	if again := agent.AgentMetrics("c1"); len(again) != 0 {
		t.Fatalf("metrics are handed over once: %+v", again)
	}
}
