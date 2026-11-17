package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func environment(variables ...string) func() []string {
	return func() []string { return variables }
}

func TestTheServiceReadsItsSettingsFromTheEnvironment(t *testing.T) {
	loaded, err := load(environment(
		"USAGE_POSTGRES_DSN=postgres://usage", "USAGE_INGEST_TOKEN=secret", "USAGE_REDIS_URL=redis://valkey:6379",
		"USAGE_METRICS_ADDR=:9090", "USAGE_MANAGE_KEY=manage", "USAGE_TIERS_FILE=/etc/usage/tiers.json", "ROUTER_ADDR=:1",
	))

	require.NoError(t, err)
	require.Equal(t, settings{
		Addr: ":8070", MetricsAddr: ":9090", PostgresDSN: "postgres://usage", IngestToken: "secret", RedisURL: "redis://valkey:6379",
		ManageKey: "manage", TiersFile: "/etc/usage/tiers.json",
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

func TestTheManageKeyIsNeverTheIngestToken(t *testing.T) {
	_, err := load(environment("USAGE_POSTGRES_DSN=postgres://usage", "USAGE_INGEST_TOKEN=secret", "USAGE_MANAGE_KEY=secret"))

	require.ErrorContains(t, err, "USAGE_MANAGE_KEY cannot be USAGE_INGEST_TOKEN")
}

func TestTiersCheckAcceptsTheDevelopmentTierFile(t *testing.T) {
	var out bytes.Buffer

	err := tiers(context.Background(), []string{"check", "--file", "../../tiers/dev.json"}, environment(), &out)

	require.NoError(t, err)
	require.Equal(t, "../../tiers/dev.json: 2 tiers: default, tier-a\n", out.String())
}

func TestTiersCheckNamesTheFileAndTheTierOfItsFirstError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiers.json")
	require.NoError(t, os.WriteFile(path, []byte(`{"tiers": [{"name": "tier-a", "fail_mode": "sometimes"}]}`), 0o600))

	err := tiers(context.Background(), []string{"check", "--file", path}, environment(), &bytes.Buffer{})

	require.ErrorContains(t, err, path)
	require.ErrorContains(t, err, `tier "tier-a"`)
	require.ErrorContains(t, err, `fail_mode "sometimes" is not closed or open`)
}

func TestTiersSyncOnlyRunsDry(t *testing.T) {
	err := tiers(context.Background(), []string{"sync"}, environment("USAGE_POSTGRES_DSN=postgres://usage"), &bytes.Buffer{})

	require.ErrorContains(t, err, "takes --dry-run")
}
