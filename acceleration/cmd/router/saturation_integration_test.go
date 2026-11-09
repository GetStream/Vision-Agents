//go:build integration

package main

import (
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// scrape is what the router's /metrics serves, or nothing while it does not.
func scrape(address string) string {
	client := http.Client{Timeout: time.Second}
	res, err := client.Get("http://" + address + "/metrics")
	if err != nil {
		return ""
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil || res.StatusCode != http.StatusOK {
		return ""
	}
	return string(body)
}

// saturationMetrics is every metric family of the pools' and Redis connections' in the
// registry /metrics serves.
func saturationMetrics() []string {
	families, _ := prometheus.DefaultGatherer.Gather()
	var found []string
	for _, family := range families {
		if strings.HasPrefix(family.GetName(), "router_db_pool_") || strings.HasPrefix(family.GetName(), "router_redis_command_") {
			found = append(found, family.GetName())
		}
	}
	return found
}

func (s *RouterStartSuite) TestARouterServesItsPostgresPoolAndRedisRoundTripsOnMetricsUntilItStops() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	settings := s.settings
	settings.Redis.Addr = address
	settings.MetricsAddr = freeAddress(s.T())
	stop := s.start(settings)

	var served string
	// The health refresh makes the router's first round trip to Redis as it starts.
	s.Eventually(func() bool {
		served = scrape(settings.MetricsAddr)
		return strings.Contains(served, `router_redis_command_seconds_count{client="shared"}`)
	}, 10*time.Second, 50*time.Millisecond, "the router served no Redis round trips")
	s.Contains(served, `router_db_pool_max_open_connections{pool="shared",shard=""} 40`)
	s.Contains(served, `router_redis_command_errors_total{client="shared"} 0`)

	s.Require().NoError(stop())
	s.Empty(saturationMetrics(), "a stopped router's pools are still served")
}

// The control: a router with neither Postgres nor Redis serves none of these, as on base.
func (s *RouterStartSuite) TestARouterWithNeitherPostgresNorRedisServesNoSaturationMetrics() {
	settings := s.settings
	settings.Postgres.DSN = ""
	settings.Redis.Addr = ""
	// Spend limits are kept in Postgres, and a router without it runs with them off.
	settings.Limits.Mode = config.LimitsOff
	settings.MetricsAddr = freeAddress(s.T())
	stop := s.start(settings)

	var served string
	s.Eventually(func() bool {
		served = scrape(settings.MetricsAddr)
		return strings.Contains(served, "# TYPE ")
	}, 10*time.Second, 50*time.Millisecond, "the router served no metrics")
	s.NotContains(served, "router_db_pool_")
	s.NotContains(served, "router_redis_command_")
	s.Require().NoError(stop())
}

// A second router in one process, as an overlap test runs, serves the first one's pools, and
// its stop leaves them served until the first stops.
func (s *RouterStartSuite) TestASecondRoutersStopLeavesTheFirstOnesPoolsServed() {
	stopFirst := serveSaturation(s.db, nil, slog.New(slog.DiscardHandler))
	defer stopFirst()
	stopSecond := serveSaturation(s.db, nil, slog.New(slog.DiscardHandler))

	stopSecond()
	s.NotEmpty(saturationMetrics(), "the second router's stop took the first one's pools out")
	stopFirst()
	s.Empty(saturationMetrics())
}
