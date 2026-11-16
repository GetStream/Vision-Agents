//go:build integration

package main

import (
	"context"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MeteringAlertsSuite runs the router's metering alerts as its start does, against the
// process's Sentry hub, with a client that keeps each event and drops it.
type MeteringAlertsSuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store

	mu   sync.Mutex
	sent []*sentry.Event
}

func TestMeteringAlertsSuite(t *testing.T) {
	suite.Run(t, new(MeteringAlertsSuite))
}

func (s *MeteringAlertsSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	settings := config.Defaults()
	settings.Postgres.DSN = testenv.Database(dsn, "metering")
	db, err := openStore(s.ctx, settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.db = db

	client, err := sentry.NewClient(sentry.ClientOptions{BeforeSend: func(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.sent = append(s.sent, event)
		return nil
	}})
	s.Require().NoError(err)
	hub := sentry.CurrentHub()
	previous := hub.Client()
	hub.BindClient(client)
	s.T().Cleanup(func() { hub.BindClient(previous) })
}

func (s *MeteringAlertsSuite) TestRowsRecordedWhileTheWriterDrainsStillAlert() {
	model := "drained-" + uuid.NewString()
	config := routing.ProviderConfig{Provider: "vendor", Model: model, Price: routing.Price{PerMillionInputTokens: 2}}
	prices, err := routing.Prices(routing.Config{routing.LLM: {Providers: []routing.ProviderConfig{config}}}, nil)
	s.Require().NoError(err)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	shutdown, stopping := context.WithCancel(s.ctx)
	stop := startMeteringAlerts(shutdown, "router/test", "test", logger)

	stopping()
	recorder := routing.NewRecorder(routing.LLM, s.db, nil, prices, logger)
	recorder.Record(config, routing.Stat{Owner: routing.Owner{CustomerID: uuid.NewString()}, Success: true})
	recorder.Close()
	stop()

	s.mu.Lock()
	defer s.mu.Unlock()
	var fingerprints []string
	for _, event := range s.sent {
		if event.Tags["item"] == "llm/vendor/"+model {
			fingerprints = append(fingerprints, event.Fingerprint...)
			s.Equal("test", event.Tags["shard"])
			s.Equal("router/test", event.Tags["source"])
		}
	}
	s.Equal([]string{"usage_missing:llm/vendor/" + model}, fingerprints,
		"the row recorded after shutdown began is checked once more when the alerts stop")
}
