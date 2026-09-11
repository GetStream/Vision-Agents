package stream

import "testing"

func TestResearchProgress(t *testing.T) {
	e := eventOf(Frame{"type": "research_progress", "phase": "answered", "tool_call_id": "call", "elapsed_ms": float64(1200), "verified_citations": float64(2)})
	if e.Research == nil || e.Research.VerifiedCitations != 2 || e.Research.ElapsedMS != 1200 || e.Research.ToolCallID != "call" {
		t.Fatal(e)
	}
}
