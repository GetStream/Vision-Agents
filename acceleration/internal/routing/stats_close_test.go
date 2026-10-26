package routing

import (
	"io"
	"log/slog"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

func TestAStatRecordedAfterCloseIsLostLoudlyNotAPanic(t *testing.T) {
	recorder := newRecorder(STT, nil, &live.Client{}, nil, nil, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	recorder.Close()
	before := testutil.ToFloat64(usageLost.WithLabelValues(lostShutdown))

	recorder.Record(ProviderConfig{}, Stat{})

	if got := testutil.ToFloat64(usageLost.WithLabelValues(lostShutdown)) - before; got != 1 {
		t.Fatalf("a stat recorded after Close was counted as lost %v times, want 1", got)
	}
}
