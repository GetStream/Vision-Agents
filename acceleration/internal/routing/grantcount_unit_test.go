package routing

import (
	"bytes"
	"log/slog"
	"testing"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestAFullGrantCounterDropsTheBatchAndSaysSoAtError(t *testing.T) {
	var logged bytes.Buffer
	c := &grantCounter{logger: slog.New(slog.NewTextHandler(&logged, nil)), queue: make(chan countJob, 1)}
	dropped := testutil.ToFloat64(grantCountsDropped)
	use := func() []store.GrantUse { return []store.GrantUse{{GrantID: 1, UsageID: uuid.New(), Micros: 5}} }

	c.offer(use(), nil)
	c.offer(use(), nil)

	require.Len(t, c.queue, 1, "the first waits to be counted")
	require.Equal(t, 1.0, testutil.ToFloat64(grantCountsDropped)-dropped, "the second has no room")
	require.Contains(t, logged.String(), "level=ERROR")
	require.Contains(t, logged.String(), "which the reconcile counts")
}
