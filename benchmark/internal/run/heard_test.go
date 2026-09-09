package run

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchAgentHeardResolvesStreamCallID(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/calls", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Customer-Id") != "voicebench" {
			http.Error(w, "missing customer", http.StatusUnauthorized)
			return
		}
		_ = json.NewEncoder(w).Encode([]map[string]string{
			{"id": "sess-other", "call_id": "other"},
			{"id": "sess-1", "call_id": "vb-restaurant-golden-1"},
		})
	})
	mux.HandleFunc("GET /v1/agents/calls/sess-1/events", func(w http.ResponseWriter, r *http.Request) {
		said := "book a table for four this Saturday at 7:30"
		cough := "cough"
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"kind": "answer", "said": said, "at": "2026-09-07T16:00:00Z"},
			{"kind": "interrupt", "said": cough, "at": "2026-09-07T16:00:01Z"},
			{"kind": "ignore", "said": "the other table", "at": "2026-09-07T16:00:02Z"},
			{"kind": "flow", "said": "skip me"},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	events, err := fetchAgentHeard(srv.URL, "voicebench", "vb-restaurant-golden-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 3 {
		t.Fatalf("%+v", events)
	}
	if events[0].Said != "book a table for four this Saturday at 7:30" {
		t.Fatalf("%+v", events)
	}
	if events[1].Kind != "interrupt" || events[2].Kind != "ignore" {
		t.Fatalf("overlap rulings missing: %+v", events)
	}
}

// An ignore with no words is the ruling most worth reading under noise: it says the
// agent heard something and decided it was not for it. Filtering on the text hid it.
func TestFetchAgentHeardKeepsARulingWithNoWords(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/calls", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]string{{"id": "sess-1", "call_id": "vb-noise-1"}})
	})
	mux.HandleFunc("GET /v1/agents/calls/sess-1/events", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"kind": "ask", "said": "table for four"},
			{"kind": "ignore", "said": "  "},
			{"kind": "ignore"},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	events, err := fetchAgentHeard(srv.URL, "voicebench", "vb-noise-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 3 {
		t.Fatalf("wordless rulings were dropped: %+v", events)
	}
	if events[1].Said != "" || events[2].Said != "" {
		t.Fatalf("%+v", events)
	}
}
