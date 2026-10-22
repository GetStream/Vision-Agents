//go:build integration

package main

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
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
