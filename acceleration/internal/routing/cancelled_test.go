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
