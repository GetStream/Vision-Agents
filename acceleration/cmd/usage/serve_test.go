//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
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
