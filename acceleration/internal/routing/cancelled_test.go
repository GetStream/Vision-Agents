package routing

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestACancelledRequestSaysNothingAboutItsProvider(t *testing.T) {
	if measuresProvider(store.Request{ErrorCode: ErrorCancelled}) {
		t.Fatal("a request its caller cancelled was counted for the provider")
	}
	if !measuresProvider(store.Request{ErrorCode: "create_failed"}) {
		t.Fatal("a request the provider failed was left out of its health")
	}
	if !measuresProvider(store.Request{Success: true}) {
		t.Fatal("a successful request was left out of its provider's health")
	}
}

func TestACancelledRequestStillAddsWhatItSpentToTheLiveCounters(t *testing.T) {
	latency := 120.0
	cancelled := store.Request{
		Modality: "llm", CustomerID: "acme", Provider: "alpha", Model: "quick",
		LatencyMs: &latency, InputTokens: 10, OutputTokens: 4, CostMicros: 18,
		ErrorCode: ErrorCancelled,
	}
	usage := liveUsage(cancelled)
	if !usage.Cancelled {
		t.Fatal("a request its caller cancelled was handed to the live counters as one that counts")
	}
	if usage.InputTokens != 10 || usage.OutputTokens != 4 || usage.CostMicros != 18 {
		t.Fatalf("what a cancelled request generated was left out of the live spend: %+v", usage)
	}

	served := cancelled
	served.ErrorCode, served.Success = "", true
	if liveUsage(served).Cancelled {
		t.Fatal("a request that was served was handed to the live counters as cancelled")
	}
	failed := cancelled
	failed.ErrorCode = "provider_error"
	if liveUsage(failed).Cancelled {
		t.Fatal("a request the provider failed was handed to the live counters as cancelled")
	}
}
