//go:build integration

package main

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// storageHour is the hour the storage tests record.
var storageHour = time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

// deploymentApps are the apps this deployment has seen, as a request naming each with its
// organization leaves them.
func (s *UsageCommandsSuite) deploymentApps(ids ...string) {
	for _, id := range ids {
		_, err := s.store.DB().NewInsert().
			Model(&store.AppOrganization{AppID: id, OrganizationID: "org-" + id, SeenAt: time.Now().UTC()}).Exec(s.ctx)
		s.Require().NoError(err)
	}
}

// runStorage runs the hourly storage job once against a turbopuffer, as the CronJob does: a
// writer of its own, drained before it returns.
func (s *UsageCommandsSuite) runStorage(server *turbopuffertest.Server, hour time.Time, logger *slog.Logger) (string, error) {
	backend, err := turbopuffer.New(turbopuffer.Options{
		APIKey: "test-key", BaseURL: server.URL, Cost: knowledgeCost(s.settings.Knowledge), Logger: logger,
	})
	s.Require().NoError(err)
	recorder, stop, err := jobRecorder(s.ctx, s.settings, s.store, routing.Knowledge, logger)
	s.Require().NoError(err)
	defer stop()
	var out bytes.Buffer
	err = recordStorage(s.ctx, metered.NewKnowledge(backend, recorder, nil), s.store, hour, logger, &out)
	return out.String(), err
}

func (s *UsageCommandsSuite) TestStorageIsRecordedOncePerNamespacePerHour() {
	s.deploymentApps("app-1", "app-2")
	server := turbopuffertest.Start(s.T())
	server.Holds("app-1__docs", 2_000_000_000)
	server.Holds("app-1__empty", 0)
	server.Holds("app-2__handbook", 1_000_000)
	server.Holds("app-9__theirs", 5_000)
	server.Holds("acceleration-test-scratch", 5)
	logger := slog.New(slog.DiscardHandler)

	first, err := s.runStorage(server, storageHour, logger)
	s.Require().NoError(err)
	_, err = s.runStorage(server, storageHour, logger)
	s.Require().NoError(err, "a rerun for the hour")

	s.Contains(first, "2026-10-23T14:00:00Z: 2 knowledge bases recorded, 1 empty, 0 missed, 0 gone since listed, 2 not this deployment's")
	rows := s.requests()
	s.Require().Len(rows, 2, "one row per knowledge base of this deployment's, however often the hour is sampled")
	byCustomer := map[string]store.Request{}
	for _, row := range rows {
		byCustomer[row.CustomerID] = row
		s.Equal("knowledge", row.Modality)
		s.Equal("turbopuffer", row.Provider)
		s.Equal(metered.KnowledgeStorage, row.Model)
		s.Equal(routing.JobOrigin("knowledge_storage"), row.APIKey, "the storage job's own origin, for the namespace's app")
		s.Equal(storageHour, row.StartedAt.UTC(), "dated to the hour it bills")
		s.True(row.ToShip, "shipped by the job's writer, or the routers' sweeper")
		s.True(row.Success)
	}
	docs := byCustomer["app-1"]
	s.Equal(routing.UsageID("knowledge_storage", "app-1__docs", "2026-10-23T14:00:00Z"), docs.UsageID)
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 2_000_000_000}, docs.Quantities())
	base, _ := s.quoted(routing.Knowledge, "turbopuffer", metered.KnowledgeStorage, nil,
		pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 2_000_000_000})
	s.Equal(base, docs.BaseCostMicros, "2 GB for an hour at turbopuffer's GB-month rate")
	s.Equal("904.110", docs.Attributes[routing.AttributeProviderCostMicros], "turbopuffer's storage rate, from config")
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 1_000_000}, byCustomer["app-2"].Quantities())

	_, err = s.runStorage(server, storageHour.Add(time.Hour), logger)
	s.Require().NoError(err)
	s.Len(s.requests(), 4, "the next hour is billed again")
}

// keptKeys are the keys customers keep, by customer, provider and region.
type keptKeys map[string]string

func (k keptKeys) CustomerKey(_ context.Context, customerID, provider, region string) (string, bool, error) {
	key, ok := k[customerID+"/"+provider+"/"+region]
	return key, ok, nil
}

func (s *UsageCommandsSuite) TestABaseInACustomersOwnAccountIsSampledOnTheirKeyAndOneWhoseKeyIsGoneIsNotMissed() {
	s.deploymentApps("app-1", "app-2")
	ours, theirs := turbopuffertest.Start(s.T()), turbopuffertest.Start(s.T())
	theirs.Holds(knowledge.Scoped("app-1", "docs"), 1_000_000)
	inTheirs := store.KnowledgePlace{Region: pricing.RegionUS, Account: store.KnowledgeAccountCustomer}
	for app, namespace := range map[string]string{"app-1": "docs", "app-2": "handbook"} {
		_, err := s.store.ClaimKnowledgePlace(s.ctx, app, namespace, inTheirs)
		s.Require().NoError(err)
	}
	logger := slog.New(slog.DiscardHandler)
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: ours.URL, Logger: logger})
	s.Require().NoError(err)
	recorder, stop, err := jobRecorder(s.ctx, s.settings, s.store, routing.Knowledge, logger)
	s.Require().NoError(err)
	base := metered.NewRegionalKnowledge(map[string]knowledge.Backend{pricing.RegionUS: backend}, recorder, nil, s.store)
	base.WithCustomerAccounts(keptKeys{"app-1/turbopuffer/us": "tpuf_customer_0123456789abcdef"}, func(_, key string) (knowledge.Backend, error) {
		return turbopuffer.New(turbopuffer.Options{APIKey: key, BaseURL: theirs.URL, Logger: logger})
	})

	var out bytes.Buffer
	err = recordStorage(s.ctx, base, s.store, storageHour, logger, &out)
	stop()

	s.Require().NoError(err, "a base whose key is gone is nobody's to bill, not a missed hour")
	s.Contains(out.String(), "1 knowledge bases recorded, 0 empty, 0 missed, 0 gone since listed, 0 not this deployment's, "+
		"1 in a customer's own account whose key is gone")
	rows := s.requests()
	s.Require().Len(rows, 1)
	s.Equal("app-1", rows[0].CustomerID)
	s.True(strings.HasSuffix(rows[0].SKU, "-us-byok"), rows[0].SKU)
	s.Zero(rows[0].BaseCostMicros, "turbopuffer bills the customer")
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 1_000_000}, rows[0].Quantities())
}

func (s *UsageCommandsSuite) TestAKnowledgeBaseThatCannotBeReadIsAMissedHour() {
	s.deploymentApps("app-1", "app-2")
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

	s.ErrorContains(err, "1 of 2 knowledge bases could not be read for 2026-10-23T14:00:00Z; run router usage storage --hour 2026-10-23T14:00:00Z",
		"the run fails, so the CronJob's retry fills the hour in")
	s.Contains(out, "1 knowledge bases recorded, 0 empty, 1 missed")
	s.Contains(logs.String(), `level=ERROR msg="could not read a knowledge base's size, so this hour of it is not billed" namespace=app-2__handbook`)
	s.Contains(logs.String(), `msg="usage alert" alert=storage_missed level=warn`)
	s.Contains(logs.String(), "item=2026-10-23T14:00:00Z", "the alert names the hour to run again")
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
