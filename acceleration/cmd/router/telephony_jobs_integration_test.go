//go:build integration

package main

import (
	"bytes"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// telephonyRows is every phone row of one model, oldest first, once the writer has drained.
func (s *UsageCommandsSuite) telephonyRows(model string) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
		Where("modality = ?", routing.Phone).Where("model = ?", model).Order("started_at").Scan(s.ctx))
	return rows
}

// runNumbers runs the rental job once, as the CronJob does: a writer of its own, drained
// before it returns.
func (s *UsageCommandsSuite) runNumbers(from, now time.Time) (string, error) {
	logger := slog.New(slog.DiscardHandler)
	recorder, stop, err := jobRecorder(s.ctx, s.settings, s.store, routing.Phone, logger)
	s.Require().NoError(err)
	defer stop()
	var out bytes.Buffer
	err = billRentals(s.ctx, s.store, recorder, from, now, logger, &out)
	return out.String(), err
}

func (s *UsageCommandsSuite) TestTheRentalJobBillsEachMonthAfterThePurchaseOnce() {
	bought := time.Date(2026, 9, 14, 10, 0, 0, 0, time.UTC)
	priced := store.PhoneNumber{E164: "+15125550101", Vendor: "telnyx", Country: "US", CustomerID: "app-1",
		MonthlyCostMicros: 1_000_000, PurchasedAt: bought}
	unpriced := store.PhoneNumber{E164: "+15125550102", Vendor: "bandwidth", Country: "US", CustomerID: "app-1",
		PurchasedAt: bought}
	own := store.PhoneNumber{E164: "+15125550103", Vendor: phone.SIPTrunkVendor, Country: "US", CustomerID: "app-1",
		PurchasedAt: bought}
	for _, number := range []*store.PhoneNumber{&priced, &unpriced, &own} {
		s.Require().NoError(s.store.RecordNumber(s.ctx, number))
	}
	now := time.Date(2026, 10, 14, 11, 0, 0, 0, time.UTC)

	first, err := s.runNumbers(bought, now)
	s.Require().NoError(err)
	again, err := s.runNumbers(bought, now)
	s.Require().NoError(err)

	s.Contains(first, "2 months of numbers billed, 1 with no price")
	s.Contains(again, "0 months of numbers billed", "a rerun bills nothing twice")
	months := s.telephonyRows(phone.ModelNumber)
	s.Require().Len(months, 2, "October's, not September's, which the purchase bills, nor the customer's own trunk's")
	for _, month := range months {
		s.Equal(phone.OriginNumberRental, month.APIKey)
		s.Equal(time.Date(2026, 10, 14, 10, 0, 0, 0, time.UTC), month.StartedAt.UTC())
		s.True(month.Billable)
	}
	costs := []int64{months[0].CostMicros, months[1].CostMicros}
	s.ElementsMatch([]int64{1_200_000, 0}, costs, "the vendor's 1,000,000 resold at the factor; a number with no price is billable work with no price")
}

func (s *UsageCommandsSuite) TestTheRentalJobBillsNothingThatBeganBeforeFrom() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: "+15125550104", Vendor: "telnyx", Country: "US", CustomerID: "app-1",
		MonthlyCostMicros: 1_000_000, PurchasedAt: time.Date(2026, 8, 14, 10, 0, 0, 0, time.UTC),
	}))

	out, err := s.runNumbers(time.Date(2026, 10, 20, 0, 0, 0, 0, time.UTC), time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC))

	s.Require().NoError(err)
	s.Contains(out, "0 months of numbers billed")
	s.Empty(s.telephonyRows(phone.ModelNumber), "October's month began before the job was switched on")
}

func (s *UsageCommandsSuite) TestTheTelephonyJobsNeedFrom() {
	var out bytes.Buffer
	s.ErrorContains(runUsage([]string{"numbers"}, s.settings, &out), "--from is required")
	s.ErrorContains(runUsage([]string{"dlc"}, s.settings, &out), "--from is required")
	s.ErrorContains(runUsage([]string{"dlc", "--from", "yesterday"}, s.settings, &out), "not an RFC 3339 time")
}

func (s *UsageCommandsSuite) TestTheDLCJobBillsACampaignsMonthsFromItsRegistration() {
	useCase := store.UseCase{CustomerID: "app-1", Name: "support", Status: dlc.Approved, UseCaseType: "low_volume"}
	s.Require().NoError(s.store.CreateUseCase(s.ctx, &useCase))
	useCase.Vendor, useCase.VendorCampaignID = "telnyx", "campaign-1"
	s.Require().NoError(s.store.MoveUseCase(s.ctx, &useCase, dlc.Approved,
		&store.ReviewLog{Actor: dlc.ActorVendor, ActorName: "telnyx", Notes: "Registered as campaign campaign-1."}))
	now := func() time.Time { return time.Now().UTC() }
	logger := slog.New(slog.DiscardHandler)
	recorder, stop, err := jobRecorder(s.ctx, s.settings, s.store, routing.Phone, logger)
	s.Require().NoError(err)
	var out bytes.Buffer

	s.Require().NoError(billCampaigns(s.ctx, s.store, recorder, time.Now().AddDate(-1, 0, 0), now, logger, &out))
	stop()

	s.Contains(out.String(), "1 months of 10DLC campaigns billed", "this month only: last month was before it was registered")
	months := s.telephonyRows(dlc.ModelCampaignMonth)
	s.Require().Len(months, 1)
	base, cost := s.quoted(routing.Phone, "telnyx", dlc.ModelCampaignMonth, map[string]string{dlc.AttributeUseCaseType: "LOW_VOLUME"},
		pricing.Quantities{pricing.Requests: 1})
	s.Equal(base, months[0].BaseCostMicros, "a low-volume campaign's month, whatever case its type was saved in")
	s.Equal(cost, months[0].CostMicros, "resold at the factor")
	s.Equal(dlc.OriginFees, months[0].APIKey)
}
