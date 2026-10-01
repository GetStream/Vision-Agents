package api

import (
	"encoding/json"
	"reflect"
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

func TestJoinTraceReachesSessionSocketUnchanged(t *testing.T) {
	trace := json.RawMessage(`{"critical_path":["sfu.ws.dial","sfu.join"],"critical_ms":150,"spans":[]}`)
	frame, ok := frameOf(agent.Connection{JoinTrace: agent.JoinTrace{Trace: trace, Flow: "fast", CriticalMs: 150}})
	if !ok || frame["type"] != "connection" {
		t.Fatalf("connection frame = %#v, ok=%v", frame, ok)
	}
	encoded, err := json.Marshal(frame)
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"type":"connection","flow":"fast","trace":`+string(trace)+`}`), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("connection frame = %s, want the flow and the trace as it came", encoded)
	}
}
