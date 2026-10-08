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

// fakeRouter holds one call, sess-1, as the router does: its timeline and what it heard.
func fakeRouter(t *testing.T, callID string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/calls", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]string{{"id": "sess-1", "call_id": callID}})
	})
	mux.HandleFunc("GET /v1/agents/calls/sess-1/timeline", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"turn_id": "turn-1", "stt_latency_ms": 7.0, "roundtrip_ms": 1500.0},
		})
	})
	mux.HandleFunc("GET /v1/agents/calls/sess-1/events", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{{"kind": "answer", "said": "a table for four"}})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestRouterBaseIsTheRouterTheRunTalksTo(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  string
		cfg  Config
		want string
	}{
		{"acceleration target at its own URL", "", Config{TargetName: "acceleration", TargetURL: "http://127.0.0.1:9191/"}, "http://127.0.0.1:9191"},
		{"acceleration target spawned on the default", "", Config{TargetName: "acceleration"}, "http://127.0.0.1:8080"},
		{"the environment overrides the target URL", "http://router.example:7000", Config{TargetName: "acceleration", TargetURL: "http://127.0.0.1:9191"}, "http://router.example:7000"},
		{"accelerated target's URL is its Python agent", "", Config{TargetName: "accelerated", TargetURL: "http://127.0.0.1:8000"}, "http://127.0.0.1:8080"},
		{"accelerated target's router is named by the environment", "http://127.0.0.1:9292/", Config{TargetName: "accelerated", TargetURL: "http://127.0.0.1:8000"}, "http://127.0.0.1:9292"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("STREAM_ACCELERATION_URL", tc.env)
			if got := routerBase(tc.cfg); got != tc.want {
				t.Fatalf("router base %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCaptureRouterTimelineFollowsTheRunsTargetURL(t *testing.T) {
	srv := fakeRouter(t, "vb-restaurant-golden-1")
	t.Setenv("STREAM_ACCELERATION_URL", "")
	dir := t.TempDir()

	stages, err := captureRouterTimeline(Config{TargetName: "acceleration", TargetURL: srv.URL}, "vb-restaurant-golden-1", dir)
	if err != nil {
		t.Fatalf("a router on another port than the default should be reached: %v", err)
	}
	if len(stages) != 1 || stages[0].RoundtripMs != 1500 {
		t.Fatalf("stages %+v", stages)
	}
	if _, err := os.Stat(filepath.Join(dir, "timeline.json")); err != nil {
		t.Fatalf("the router's timeline should be kept with the call: %v", err)
	}
}

func TestCaptureAgentHeardFollowsTheRunsTargetURL(t *testing.T) {
	srv := fakeRouter(t, "vb-restaurant-golden-1")
	t.Setenv("STREAM_ACCELERATION_URL", "")
	dir := t.TempDir()

	if err := captureAgentHeard(Config{TargetName: "acceleration", TargetURL: srv.URL}, "vb-restaurant-golden-1", dir); err != nil {
		t.Fatalf("a router on another port than the default should be reached: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "heard.json")); err != nil {
		t.Fatalf("what the agent heard should be kept with the call: %v", err)
	}
}

func TestReplyStagesKeepsTheRepliesAfterATool(t *testing.T) {
	ptr := func(v float64) *float64 { return &v }
	stages := replyStages([]timelineEntry{
		{TurnID: "greeting", RoundtripMs: ptr(900)},
		{TurnID: "turn-1", SttLatencyMs: ptr(7), RoundtripMs: ptr(2200), ModelToFirstTextMs: ptr(700)},
		{TurnID: "tool-1", ModelToFirstTextMs: ptr(640.4), TextToTtsMs: ptr(51), TtsToAudioMs: ptr(2900.6)},
		{TurnID: "tool-2", ModelToFirstTextMs: ptr(500)},
	})
	if len(stages) != 2 || stages[0].TurnID != "turn-1" || stages[0].Tool || stages[1].TurnID != "tool-1" || !stages[1].Tool {
		t.Fatalf("the caller's turn and the reply that reached audio after a tool are kept, nothing else: %+v", stages)
	}
	got := stages[1]
	if got.ModelToTextMs != 640 || got.TextToTTSMs != 51 || got.TTSToAudioMs != 2901 || got.STTMs != 0 || got.RoundtripMs != 0 {
		t.Fatalf("a reply after a tool has the stages from the model on: %+v", got)
	}
}
