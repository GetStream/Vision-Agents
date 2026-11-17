package main

import (
	"bytes"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

func TestAUsageTokenWithoutTheServiceWarnsAndShipsNothing(t *testing.T) {
	var logs bytes.Buffer
	settings := config.Defaults()
	settings.Usage.Token = "the-ingest-token"

	client, err := newUsageClient(settings, nil, slog.New(slog.NewTextHandler(&logs, nil)))

	require.NoError(t, err)
	require.Nil(t, client)
	require.Contains(t, logs.String(), "level=WARN")
	require.Contains(t, logs.String(), "usage token is set without the usage service's URL")
	require.NotContains(t, logs.String(), "the-ingest-token")
}

func TestAStorageRunRecordsTheLastHourToHaveEnded(t *testing.T) {
	ended := time.Date(2026, 10, 23, 13, 0, 0, 0, time.UTC)
	for _, now := range []time.Time{ended.Add(time.Hour + 17*time.Minute), ended.Add(time.Hour + 59*time.Minute)} {
		hour, err := storageHourOf("", now)
		require.NoError(t, err)
		require.Equal(t, ended, hour, "the run at %s and its retries within the hour bill the same hour", now)
	}

	hour, err := storageHourOf("2026-10-23T09:30:00+02:00", time.Now())
	require.NoError(t, err)
	require.Equal(t, time.Date(2026, 10, 23, 7, 0, 0, 0, time.UTC), hour, "--hour names the hour it falls in")

	_, err = storageHourOf("yesterday", time.Now())
	require.ErrorContains(t, err, "RFC 3339")
}
