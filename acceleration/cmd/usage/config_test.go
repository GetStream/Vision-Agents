package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func environment(variables ...string) func() []string {
	return func() []string { return variables }
}

func TestTheServiceReadsItsSettingsFromTheEnvironment(t *testing.T) {
	loaded, err := load(environment(
		"USAGE_POSTGRES_DSN=postgres://usage", "USAGE_INGEST_TOKEN=secret", "USAGE_REDIS_URL=redis://valkey:6379",
		"USAGE_METRICS_ADDR=:9090", "USAGE_MANAGE_KEY=not-read-yet", "ROUTER_ADDR=:1",
	))

	require.NoError(t, err)
	require.Equal(t, settings{
		Addr: ":8070", MetricsAddr: ":9090", PostgresDSN: "postgres://usage", IngestToken: "secret", RedisURL: "redis://valkey:6379",
	}, loaded)
}

func TestTheServiceRefusesToStartWithoutItsDatabase(t *testing.T) {
	_, err := load(environment("USAGE_INGEST_TOKEN=secret"))

	require.ErrorContains(t, err, "USAGE_POSTGRES_DSN is required")
}

func TestTheServiceRefusesToStartWithoutAnIngestToken(t *testing.T) {
	_, err := load(environment("USAGE_POSTGRES_DSN=postgres://usage"))

	require.ErrorContains(t, err, "USAGE_INGEST_TOKEN is required")
}
