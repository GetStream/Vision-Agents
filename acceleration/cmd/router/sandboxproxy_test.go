package main

import (
	"log/slog"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

// TestARouterWithoutAProxyAddressServesNoProxy is the control: unset, as on every router
// before the proxy, there is nothing to listen with.
func TestARouterWithoutAProxyAddressServesNoProxy(t *testing.T) {
	proxy, err := newSandboxProxy(config.Defaults().CodeSandbox, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.Nil(t, proxy)
}

// TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart: the grants live in Redis and the
// tokens come from the connections resolver.
func TestAProxyAddressWithoutRedisOrConnectorsDoesNotStart(t *testing.T) {
	settings := config.Defaults().CodeSandbox
	settings.ProxyAddr = ":8091"

	_, err := newSandboxProxy(settings, nil, nil, nil, slog.New(slog.DiscardHandler))

	require.ErrorContains(t, err, "redis.addr")
	_, err = newSandboxProxy(settings, nil, &resolver.Resolver{}, nil, slog.New(slog.DiscardHandler))
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(redis.Close)
	_, err = newSandboxProxy(settings, redis, nil, nil, slog.New(slog.DiscardHandler))
	require.ErrorContains(t, err, "connectors.enabled")
}
