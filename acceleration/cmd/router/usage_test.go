package main

import (
	"bytes"
	"log/slog"
	"testing"

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
