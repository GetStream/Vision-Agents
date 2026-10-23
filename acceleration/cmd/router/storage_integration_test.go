//go:build integration

package main

import (
	"bytes"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// storageHour is the hour the storage tests record.
var storageHour = time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

// runStorage runs the hourly storage job once against a turbopuffer, as the CronJob does: a
// writer of its own, drained before it returns.
func (s *UsageCommandsSuite) runStorage(server *turbopuffertest.Server, at time.Time, logger *slog.Logger) (string, error) {
	backend, err := turbopuffer.New(turbopuffer.Options{
		APIKey: "test-key", BaseURL: server.URL, Cost: knowledgeCost(s.settings.Knowledge), Logger: logger,
	})
	s.Require().NoError(err)
	recorder, stop, err := storageRecorder(s.ctx, s.settings, s.store, logger)
	s.Require().NoError(err)
	defer stop()
	var out bytes.Buffer
	err = recordStorage(s.ctx, metered.NewKnowledge(backend, recorder), at, logger, &out)
	return out.String(), err
}

func (s *UsageCommandsSuite) TestStorageIsRecordedOncePerNamespacePerHour() {
	server := turbopuffertest.Start(s.T())
	server.Holds("app-1__docs", 2_000_000_000)
	server.Holds("app-2__handbook", 1_000_000)
	server.Holds("acceleration-test-scratch", 5)
	logger := slog.New(slog.DiscardHandler)

	first, err := s.runStorage(server, storageHour.Add(17*time.Minute), logger)
	s.Require().NoError(err)
	_, err = s.runStorage(server, storageHour.Add(41*time.Minute), logger)
	s.Require().NoError(err, "a rerun within the hour")

	s.Contains(first, "2026-10-23T14:00:00Z: 2 knowledge bases recorded, 0 missed, 0 gone since listed, 1 not a customer's")
	rows := s.requests()
	s.Require().Len(rows, 2, "one row per knowledge base, however often the hour is sampled")
	byNamespace := map[string]store.Request{}
	for _, row := range rows {
		byNamespace[row.CustomerID] = row
		s.Equal("knowledge", row.Modality)
		s.Equal("turbopuffer", row.Provider)
		s.Equal(metered.KnowledgeStorage, row.Model)
		s.Equal(routing.JobOrigin("knowledge_storage"), row.APIKey, "the storage job's own origin, for the namespace's app")
		s.True(row.ToShip, "shipped by the job's writer, or the routers' sweeper")
		s.True(row.Success)
	}
	docs := byNamespace["app-1"]
	s.Equal(routing.UsageID("knowledge_storage", "app-1__docs", "2026-10-23T14:00:00Z"), docs.UsageID)
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 2_000_000_000}, docs.Quantities())
	s.EqualValues(904, docs.CostMicros, "2 GB for an hour at $0.33 a GB-month")
	s.Equal("904.110", docs.Attributes[routing.AttributeProviderCostMicros], "turbopuffer's storage rate, from config")
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 1_000_000}, byNamespace["app-2"].Quantities())

	_, err = s.runStorage(server, storageHour.Add(time.Hour), logger)
	s.Require().NoError(err)
	s.Len(s.requests(), 4, "the next hour is billed again")
}

func (s *UsageCommandsSuite) TestAKnowledgeBaseThatCannotBeReadIsAMissedHour() {
	server := turbopuffertest.Start(s.T())
	server.Holds("app-1__docs", 2_000_000_000)
	server.Holds("app-2__handbook", 1_000_000)
	server.Unreadable("app-2__handbook")
	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))
	// The alert is logged through the default logger as it is sent.
	defaultLogger := slog.Default()
	slog.SetDefault(logger)
	defer slog.SetDefault(defaultLogger)

	out, err := s.runStorage(server, storageHour, logger)

	s.ErrorContains(err, "1 of 2 knowledge bases could not be read", "the run fails, so the CronJob's retry fills the hour in")
	s.Contains(out, "1 knowledge bases recorded, 1 missed")
	s.Contains(logs.String(), `level=ERROR msg="could not read a knowledge base's size, so this hour of it is not billed" namespace=app-2__handbook`)
	s.Contains(logs.String(), `msg="usage alert" alert=storage_missed level=warn`)
	s.Len(s.requests(), 2, "the row of the base that was read, and the failed read, which bills nothing")

	_, err = s.runStorage(server, storageHour, slog.New(slog.DiscardHandler))
	s.Require().Error(err, "still unreadable")
	billed := 0
	for _, row := range s.requests() {
		if row.Success {
			billed++
		}
	}
	s.Equal(1, billed, "the base that was read is not billed twice by the retry")
}
