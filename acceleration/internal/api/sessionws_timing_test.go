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
