package routing

import (
	"io"
	"log/slog"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

func TestAStatRecordedAfterCloseIsDroppedNotAPanic(t *testing.T) {
	recorder := newRecorder(STT, nil, &live.Client{}, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	recorder.Close()

	recorder.Record(ProviderConfig{}, Stat{})

	if got := recorder.dropped.Load(); got != 1 {
		t.Fatalf("a stat recorded after Close was counted as dropped %d times, want 1", got)
	}
}
