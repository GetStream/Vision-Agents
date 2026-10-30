//go:build integration

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// ServeSuite runs the whole service, as the binary does, on a database of its own.
type ServeSuite struct {
	suite.Suite
	dsn string
}

func TestServeSuite(t *testing.T) {
	suite.Run(t, new(ServeSuite))
}

func (s *ServeSuite) SetupSuite() {
	s.dsn = usagetest.Database(s.T(), "usage_cmd")
}

func (s *ServeSuite) SetupTest() {
	opened, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer opened.Close() //nolint:errcheck // a pool the test is done with
	usagetest.Empty(s.T(), opened)
}

// start serves the service with settings until the test ends, once it answers.
func (s *ServeSuite) start(settings settings) string {
	ctx, stop := context.WithCancel(context.Background())
	stopped := make(chan error, 1)
	go func() { stopped <- serve(ctx, settings, slog.New(slog.NewTextHandler(io.Discard, nil))) }()
	s.T().Cleanup(func() {
		stop()
		<-stopped
	})
	base := "http://" + settings.Addr
	s.Require().Eventually(func() bool {
		response, err := http.Get(base + "/healthz")
		if err != nil {
			return false
		}
		_ = response.Body.Close()
		return response.StatusCode == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond)
	return base
}

func (s *ServeSuite) TestTheServiceSeedsItsTierFileAndServesTheManagementAPI() {
	base := s.start(settings{
		Addr: s.freeAddress(), PostgresDSN: s.dsn, IngestToken: "token", ManageKey: "manage", TiersFile: "../../tiers/dev.json",
	})

	request, err := http.NewRequest(http.MethodGet, base+"/v1/manage/tiers", nil)
	s.Require().NoError(err)
	request.Header.Set("X-Manage-Key", "manage")
	request.Header.Set("X-Stream-Actor-Id", "operator")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body

	s.Require().Equal(http.StatusOK, response.StatusCode)
	var list struct {
		Tiers []usage.Tier `json:"tiers"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&list))
	file, err := ledger.ReadTierFile("../../tiers/dev.json")
	s.Require().NoError(err)
	s.Equal(file.Tiers, list.Tiers)
}

func (s *ServeSuite) TestWithoutATierFileTheBuiltInDefaultIsSeeded() {
	base := s.start(settings{Addr: s.freeAddress(), PostgresDSN: s.dsn, IngestToken: "token"})

	response, err := http.Get(base + "/v1/manage/tiers")
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusNotFound, response.StatusCode, "no manage key, no management API")
	opened, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer opened.Close() //nolint:errcheck // a pool the test is done with
	tiers, err := opened.Tiers(context.Background())
	s.Require().NoError(err)
	s.Equal([]usage.Tier{usage.BuiltInDefaultTier()}, tiers)
}

func (s *ServeSuite) TestABadTierFileStopsTheStart() {
	path := filepath.Join(s.T().TempDir(), "tiers.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"tiers": [{"name": "tier-a", "fail_mode": "open"}, {"name": "tier-a", "fail_mode": "open"}]}`), 0o600))

	err := serve(context.Background(), settings{Addr: s.freeAddress(), PostgresDSN: s.dsn, IngestToken: "token", TiersFile: path},
		slog.New(slog.NewTextHandler(io.Discard, nil)))

	s.Require().ErrorContains(err, path)
	s.Require().ErrorContains(err, `tier "tier-a" is named twice`)
}

func (s *ServeSuite) TestTiersSyncDryRunPrintsWhatTheNextStartAdds() {
	environ := environment("USAGE_POSTGRES_DSN="+s.dsn, "USAGE_TIERS_FILE=../../tiers/dev.json")
	var before, after bytes.Buffer

	s.Require().NoError(tiers(context.Background(), []string{"sync", "--dry-run"}, environ, &before))
	s.start(settings{Addr: s.freeAddress(), PostgresDSN: s.dsn, IngestToken: "token", TiersFile: "../../tiers/dev.json"})
	s.Require().NoError(tiers(context.Background(), []string{"sync", "--dry-run"}, environ, &after))

	s.Equal("the next start adds: default, tier-a\n", before.String())
	s.Equal("the next start adds no tier\n", after.String())
}

func (s *ServeSuite) TestTiersSyncDryRunOnADatabaseNotMigratedYetPlansEveryTier() {
	unmigrated := usagetest.EmptyDatabase(s.T(), "usage_cmd_unmigrated")
	var out bytes.Buffer

	err := tiers(context.Background(), []string{"sync", "--dry-run", "--file", "../../tiers/dev.json"},
		environment("USAGE_POSTGRES_DSN="+unmigrated), &out)

	s.Require().NoError(err)
	s.Equal("the database has no tiers table yet: the next start migrates it, then adds: default, tier-a\n", out.String())
}

// freeAddress is a local address nothing listens on.
func (s *ServeSuite) freeAddress() string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	defer listener.Close() //nolint:errcheck // only its port was wanted
	return listener.Addr().String()
}

func (s *ServeSuite) TestTheServiceRecordsServesMetricsAndDrainsOnSignal() {
	settings := settings{
		Addr: s.freeAddress(), MetricsAddr: s.freeAddress(), PostgresDSN: s.dsn,
		IngestToken: "token", RedisURL: usagetest.RedisURL(s.T()),
	}
	ctx, stop := context.WithCancel(context.Background())
	stopped := make(chan error, 1)
	go func() { stopped <- serve(ctx, settings, slog.New(slog.NewTextHandler(io.Discard, nil))) }()
	base := "http://" + settings.Addr
	s.Eventually(func() bool {
		response, err := http.Get(base + "/healthz")
		if err != nil {
			return false
		}
		_ = response.Body.Close()
		return response.StatusCode == http.StatusOK
	}, 10*time.Second, 50*time.Millisecond)

	client, err := usage.NewClient(usage.ClientOptions{URL: base, Token: "token"})
	s.Require().NoError(err)
	organization := usagetest.Organization()
	defer usagetest.Forget(s.T(), usagetest.Redis(s.T()), organization)
	ack, err := client.Ship(context.Background(), []usage.Event{usagetest.Event(organization)})
	s.Require().NoError(err)
	s.Equal(1, ack.Accepted)

	response, err := http.Get("http://" + settings.MetricsAddr + "/metrics")
	s.Require().NoError(err)
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(response.Body.Close())
	s.Require().NoError(err)
	s.True(bytes.Contains(body, []byte("usage_events_written_total")), "the service's metrics are served")
	s.True(bytes.Contains(body, []byte("usage_windows_upserted_total")), "and the ledger's")

	_, port, err := net.SplitHostPort(settings.Addr)
	s.Require().NoError(err)
	s.NoError(healthcheck(":"+port), "the container's health check reaches it")

	stop()
	select {
	case err := <-stopped:
		s.NoError(err)
	case <-time.After(drainTimeout + 5*time.Second):
		s.Fail("the service did not stop after its drain")
	}
	_, err = http.Get(base + "/healthz")
	s.Error(err, "nothing listens once it has stopped")
	s.Error(healthcheck(settings.Addr))
}

// TestEveryListenerClosesAnIdleConnection starts the service with its API and metrics
// listeners and both idle timeouts shortened, and sends one request on a keep-alive connection
// to each. The server must then close the connection. A server that loses its IdleTimeout, or
// has it overwritten in serve after its literal, keeps the connection open and the read times
// out.
func (s *ServeSuite) TestEveryListenerClosesAnIdleConnection() {
	previous, previousMetrics := idleTimeout, metricsIdleTimeout
	// 300 ms is far under the 20 s read deadline below, so a closed connection shows quickly and
	// a kept-open one fails the test.
	idleTimeout, metricsIdleTimeout = 300*time.Millisecond, 300*time.Millisecond
	s.T().Cleanup(func() { idleTimeout, metricsIdleTimeout = previous, previousMetrics })
	// 20000-32767 is below the ephemeral ranges of Linux (32768-60999) and macOS (49152-65535),
	// so the service's own Postgres connections do not take a port before serve listens on it.
	// The process id spreads parallel packages over the range (as cmd/router/idletimeout_test.go).
	apiAddr, metricsAddr := s.freeAddressPair(20000 + (os.Getpid()%2000)*6)
	settings := settings{Addr: apiAddr, MetricsAddr: metricsAddr, PostgresDSN: s.dsn, IngestToken: "token"}
	s.start(settings)

	listeners := []struct{ name, addr, path string }{
		{"api", settings.Addr, "/healthz"},
		{"metrics", settings.MetricsAddr, "/metrics"},
	}
	for _, listener := range listeners {
		s.Run(listener.name, func() {
			conn := s.dialWhenListening(listener.addr)
			defer conn.Close() //nolint:errcheck // the test is ending
			reader := bufio.NewReader(conn)

			_, err := fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: x\r\n\r\n", listener.path)
			s.Require().NoError(err)
			response, err := http.ReadResponse(reader, nil)
			s.Require().NoError(err)
			_, err = io.Copy(io.Discard, response.Body)
			s.Require().NoError(err)
			s.Require().NoError(response.Body.Close())
			s.Require().Equal(http.StatusOK, response.StatusCode)

			// 20 s is generous for a starved CI runner (the router's test saw a stalled VM overrun
			// 5 s) and stays under readTimeout, 30 s: a server whose IdleTimeout is zero falls back
			// to its ReadTimeout (net/http Server.IdleTimeout doc), so the API server would close
			// the connection then and pass a longer wait.
			s.Require().NoError(conn.SetReadDeadline(time.Now().Add(20 * time.Second)))
			_, err = reader.ReadByte()
			s.True(errors.Is(err, io.EOF), "the server held an idle connection open: %v", err)
		})
	}
}

// TestFreeAddressPairDiffers holds the first candidate port, which pushes the API address on to
// the next port. The metrics address must still differ from it: scanning from the same start
// would have both addresses land on that next port.
func (s *ServeSuite) TestFreeAddressPairDiffers() {
	port := s.freePort(20000 + (os.Getpid()%2000)*6)
	held, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	s.Require().NoError(err)
	defer held.Close() //nolint:errcheck // the test is ending
	apiAddr, metricsAddr := s.freeAddressPair(port)
	s.NotEqual(apiAddr, metricsAddr)
}

// freeAddressPair is two different local addresses that nothing listens on, found by scanning
// from port up to 32767 and wrapping to 20000. The second scan starts after the first address.
func (s *ServeSuite) freeAddressPair(port int) (string, string) {
	apiPort := s.freePort(port)
	return fmt.Sprintf("127.0.0.1:%d", apiPort), fmt.Sprintf("127.0.0.1:%d", s.freePort(apiPort+1))
}

func (s *ServeSuite) freePort(from int) int {
	for i := 0; i < 32768-20000; i++ {
		port := 20000 + (from-20000+i)%(32768-20000)
		opened, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			continue
		}
		s.Require().NoError(opened.Close())
		return port
	}
	s.FailNow("no free port in 20000-32767")
	return 0
}

// dialWhenListening connects to addr once something listens there: serve starts the metrics
// listener in a goroutine of its own, so it can still be starting when the API answers.
func (s *ServeSuite) dialWhenListening(addr string) net.Conn {
	// 10 s matches the healthz wait in start.
	deadline := time.Now().Add(10 * time.Second)
	for {
		conn, err := net.DialTimeout("tcp", addr, time.Second)
		if err == nil {
			return conn
		}
		s.Require().True(time.Now().Before(deadline), "nothing listened on %s: %v", addr, err)
		time.Sleep(50 * time.Millisecond)
	}
}
