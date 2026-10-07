package llmrouter

import (
	"context"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestARequestItsCallerCancelledIsNotAProviderFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	if got := createErrorCode(ctx); got != "create_failed" {
		t.Fatalf("a request that failed on its own was recorded as %q", got)
	}
	cancel()
	if got := createErrorCode(ctx); got != routing.ErrorCancelled {
		t.Fatalf("a request its caller cancelled was recorded as %q", got)
	}
}
