package streamrtc

import (
	"context"
	"testing"
)

func TestJoinRejectsEmptyConfig(t *testing.T) {
	_, err := Join(context.Background(), Options{})
	if err == nil {
		t.Fatal("expected error")
	}
}
