package main

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestAReconciliationChecksTheDayBeforeYesterday(t *testing.T) {
	for _, now := range []time.Time{time.Date(2026, 10, 23, 0, 30, 0, 0, time.UTC), time.Date(2026, 10, 23, 23, 59, 0, 0, time.UTC)} {
		day, err := sandboxDayOf("", now)
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC), day, "Daytona's records lag up to 48 hours")
	}

	day, err := sandboxDayOf("2026-10-01", time.Now())
	require.NoError(t, err)
	require.Equal(t, time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC), day)
	_, err = sandboxDayOf("2026-10-01T00:00:00Z", time.Now())
	require.ErrorContains(t, err, "is not a YYYY-MM-DD day")
}

func TestASandboxIsApartFromDaytonasRecordsBeyond5PercentOr60Seconds(t *testing.T) {
	daytonas := tally{cpu: 3600, memory: 3600, disk: 10_800}
	require.False(t, apart(daytonas, daytonas))
	require.False(t, apart(daytonas, tally{cpu: 3420, memory: 3780, disk: 10_260}), "5% either way")
	require.True(t, apart(daytonas, tally{cpu: 3419, memory: 3600, disk: 10_800}), "on any meter")
	require.True(t, apart(daytonas, tally{cpu: 3600, memory: 3600, disk: 11_341}))

	small := tally{cpu: 100, memory: 100, disk: 300}
	require.False(t, apart(small, tally{cpu: 40, memory: 160, disk: 240}), "60 units is the least")
	require.True(t, apart(small, tally{cpu: 39, memory: 100, disk: 300}))
	require.True(t, apart(small, tally{}), "a sandbox nothing was billed for")
}

func TestTheSessionsSandboxesAreLabelledWithTheDeploymentAndShard(t *testing.T) {
	t.Setenv("SENTRY_ENVIRONMENT", "production")
	settings := config.Defaults()
	settings.Shard = "shard-b"

	labels := newSandboxes(settings, nil, nil, nil).Labels(routing.Owner{CustomerID: "app-1",
		Attribution: routing.Attribution{SessionID: "session-1"}})

	require.Equal(t, map[string]string{metered.LabelEnv: "production", metered.LabelRouter: "shard-b", metered.LabelApp: "app-1",
		metered.LabelSession: "session-1"}, labels, "an organization the session does not know is left off")
	t.Setenv("SENTRY_ENVIRONMENT", "")
	settings.Shard = ""
	labels = newSandboxes(settings, nil, nil, nil).Labels(routing.Owner{})
	require.Equal(t, map[string]string{metered.LabelEnv: "development", metered.LabelRouter: routing.Unsharded}, labels)
}
