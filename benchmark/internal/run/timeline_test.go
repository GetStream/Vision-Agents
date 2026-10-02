package run

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestCaptureRouterTimelineKeepsTheCallersTurns(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/calls", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]string{{"id": "sess-1", "call_id": "vb-restaurant-golden-1"}})
	})
	mux.HandleFunc("GET /v1/agents/calls/sess-1/timeline", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"turn_id": "greeting", "started_at": "2026-10-02T10:00:00Z", "roundtrip_ms": 900.0},
			{"turn_id": "turn-1", "started_at": "2026-10-02T10:00:05Z", "stt_latency_ms": 7.4, "cadence_ms": 350.2,
				"decision_ms": 440.6, "model_to_first_text_ms": 704.0, "text_to_tts_ms": 53.0, "tts_to_audio_ms": 685.0,
				"roundtrip_ms": 2232.8, "speech_end_to_audio_ms": 2240.2, "interrupted": true},
			{"turn_id": "tool-1", "started_at": "2026-10-02T10:00:09Z", "roundtrip_ms": 1200.0},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	t.Setenv("STREAM_ACCELERATION_URL", srv.URL)
	dir := t.TempDir()

	stages, err := captureRouterTimeline(Config{TargetName: "accelerated"}, "vb-restaurant-golden-1", dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 1 || stages[0].TurnID != "turn-1" {
		t.Fatalf("only the turn the caller started is a reply: %+v", stages)
	}
	got := stages[0]
	if got.STTMs != 7 || got.CadenceMs != 350 || got.DecisionMs != 441 || got.RoundtripMs != 2233 || got.SpeechToAudioMs != 2240 || !got.Interrupted {
		t.Fatalf("stages %+v", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "timeline.json")); err != nil {
		t.Fatalf("the router's timeline should be kept with the call: %v", err)
	}
}

func TestCaptureRouterTimelineIsOnlyForTargetsOnTheRouter(t *testing.T) {
	stages, err := captureRouterTimeline(Config{TargetName: "livekit"}, "vb-restaurant-golden-1", t.TempDir())
	if err != nil || stages != nil {
		t.Fatalf("a LiveKit call has no router timeline: %+v %v", stages, err)
	}
}
