package phone

import (
	"context"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RentalLookback is how far back the rental job looks for a month that began and was not
// billed: a router that was down for a day still bills the numbers whose month began then.
const RentalLookback = 72 * time.Hour

// RentalEvery is how often the rental job runs. A month is billed once whatever the count.
const RentalEvery = 6 * time.Hour

// OriginNumberRental is the origin of the rows billing a held number's months after its first.
var OriginNumberRental = routing.JobOrigin("number_rental")

// billMonth bills one month of a number to owner, at the price it was bought at. The month is
// named by its usage id, so billing it again bills nothing. A number that costs nothing, as one
// on the customer's own trunk, writes no row.
func (s *Service) billMonth(number store.PhoneNumber, owner routing.Owner, month time.Time) {
	if number.MonthlyCostMicros <= 0 {
		return
	}
	meter := numberMonth(number, month)
	meter.Recorder, meter.Owner = s.recorder, owner
	if err := routing.Settle(meter, pricing.Quantities{}); err != nil {
		s.logger.Error("could not bill a month of a number", "number", number.E164, "error", err)
	}
}

// BillRentals bills each month of a held number that began within RentalLookback and has no
// row yet, to the number's app, and reports how many it billed. The month a number was bought
// in is billed when it is bought; this bills the ones after it.
func (s *Service) BillRentals(ctx context.Context) (int, error) {
	if s.store == nil || s.recorder == nil {
		return 0, nil
	}
	now := s.now().UTC()
	since := now.Add(-RentalLookback)
	held, err := s.store.NumbersHeldSince(ctx, since)
	if err != nil {
		return 0, err
	}
	type due struct {
		number store.PhoneNumber
		month  time.Time
	}
	var months []due
	var ids []uuid.UUID
	for _, number := range held {
		for _, month := range MonthsBegun(number.PurchasedAt, since, now) {
			if number.ReleasedAt != nil && !month.Before(*number.ReleasedAt) {
				continue
			}
			months = append(months, due{number, month})
			ids = append(ids, numberMonth(number, month).UsageID)
		}
	}
	recorded, err := s.store.RecordedUsage(ctx, ids)
	if err != nil {
		return 0, err
	}
	billed := 0
	for _, each := range months {
		if recorded[numberMonth(each.number, each.month).UsageID] {
			continue
		}
		s.billMonth(each.number, routing.Owner{
			CustomerID:  each.number.CustomerID,
			Attribution: routing.Attribution{APIKey: OriginNumberRental},
			Tags:        each.number.Tags,
		}, each.month)
		billed++
	}
	return billed, nil
}

// RunRentals bills rentals at once and then every interval until ctx ends.
func (s *Service) RunRentals(ctx context.Context, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		if billed, err := s.BillRentals(ctx); err != nil {
			s.logger.Error("could not bill the months of held numbers", "error", err)
		} else if billed > 0 {
			s.logger.Info("billed the months of held numbers", "months", billed)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// MonthsBegun is when each month of a purchase began, in (after, until]: the purchase's day
// and time in each month after it, or that month's last day when it is shorter, as a number
// bought on January 31st is billed again on February 28th. The month of the purchase itself
// is the first.
func MonthsBegun(purchased, after, until time.Time) []time.Time {
	purchased = purchased.UTC()
	var begun []time.Time
	for k := 0; ; k++ {
		month := monthAfter(purchased, k)
		if month.After(until) {
			return begun
		}
		if month.After(after) {
			begun = append(begun, month)
		}
	}
}

// monthAfter is the start of the k-th month after purchased, clamped to that month's end.
func monthAfter(purchased time.Time, k int) time.Time {
	first := time.Date(purchased.Year(), purchased.Month()+time.Month(k), 1, 0, 0, 0, 0, time.UTC)
	last := first.AddDate(0, 1, -1).Day()
	return time.Date(first.Year(), first.Month(), min(purchased.Day(), last),
		purchased.Hour(), purchased.Minute(), purchased.Second(), purchased.Nanosecond(), time.UTC)
}
