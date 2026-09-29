package api

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestModelCallTimingReachesSessionSocket(t *testing.T) {
	frame, ok := frameOf(agent.ModelCall{CallTiming: llm.CallTiming{
		OperationID: "flow-1", Purpose: "flow", TurnID: "turn-1",
		Provider: "stub", Model: "fast", TTFTMs: 80, DurationMs: 110, Success: true,
	}})
	if !ok || frame["type"] != "model_call" || frame["turn_id"] != "turn-1" ||
		frame["purpose"] != "flow" || frame["duration_ms"] != float64(110) {
		t.Fatalf("model timing frame = %#v, ok=%v", frame, ok)
	}
}

func TestConnectionTimingReachesSessionSocket(t *testing.T) {
	frame, ok := frameOf(agent.Connection{ConnectionTiming: agent.ConnectionTiming{
		Peer: "publisher", TotalMs: 720, FirstMediaMs: 730,
		Steps: []agent.ConnectionStep{{Name: "SFU join", Ms: 50, AtMs: 505}, {Name: "DTLS", Ms: 60, AtMs: 720}},
	}})
	steps, _ := frame["steps"].([]map[string]any)
	if !ok || frame["type"] != "connection" || frame["peer"] != "publisher" ||
		frame["total_ms"] != float64(720) || frame["first_media_ms"] != float64(730) ||
		len(steps) != 2 || steps[1]["name"] != "DTLS" || steps[1]["ms"] != float64(60) || steps[0]["at_ms"] != float64(505) {
		t.Fatalf("connection timing frame = %#v, ok=%v", frame, ok)
	}
}
