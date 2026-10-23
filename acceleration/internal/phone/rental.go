package phone

import (
	"context"
	"log/slog"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RentalLookback is how far back a rental run looks for a month that began and has no row: a
// job that did not run for weeks still bills the months that began meanwhile.
const RentalLookback = 35 * 24 * time.Hour

// OriginNumberRental is the origin of the rows billing a held number's months after its first.
var OriginNumberRental = routing.JobOrigin("number_rental")

// quote is what the vendor offers the purchase's number at, read off a search for that number,
// or 0 when the vendor cannot search for one number or names no price.
func (s *Service) quote(ctx context.Context, provider Provider, purchase Purchase) int64 {
	national := National(purchase.E164)
	if national == "" {
		return 0
	}
	search := Search{Country: purchase.Country, Prefix: national}
	if search.Country == "" {
		search.Country = Country(purchase.E164)
	}
	if len(search.Unsupported(provider)) > 0 {
		search = Search{Country: search.Country, Contains: national}
		if len(search.Unsupported(provider)) > 0 {
			return 0
		}
	}
	offers, err := provider.SearchNumbers(ctx, search)
	if err != nil {
		s.logger.Warn("could not read what a number is offered at before buying it", "vendor", purchase.Vendor, "error", err)
		return 0
	}
	for _, offer := range offers {
		if offer.E164 == purchase.E164 {
			return offer.MonthlyCostMicros
		}
	}
	return 0
}

// billMonth bills one month of a number to owner, at the price it was bought at. The month is
// named by its usage id, so billing it again bills nothing. A number on the customer's own
// trunk costs nothing and writes no row; one whose price is not known is billable work with no
// price, which raises the alarm, rather than a month given away.
func billMonth(recorder *routing.Recorder, logger *slog.Logger, number store.PhoneNumber, owner routing.Owner,
	month time.Time) {
	if number.Vendor == SIPTrunkVendor {
		return
	}
	meter := numberMonth(number, month)
	meter.Recorder, meter.Owner = recorder, owner
	if err := routing.Settle(meter, pricing.Quantities{}); err != nil {
		logger.Error("could not bill a month of a number", "number", number.E164, "error", err)
	}
}

// billMonth bills a month of a number the service bought.
func (s *Service) billMonth(number store.PhoneNumber, owner routing.Owner, month time.Time) {
	billMonth(s.recorder, s.logger, number, owner, month)
}

// Rentals bills the months of held numbers after the one each was bought in, which the purchase
// billed. It is run as a job, `router usage numbers`.
type Rentals struct {
	Store    *store.Store
	Recorder *routing.Recorder
	Logger   *slog.Logger
}

// Rented is what one rental run billed.
type Rented struct {
	// Months is how many months it billed, of which Unpriced at no price the vendor named.
	Months   int
	Unpriced []string
}

// Bill bills each month of a held number that began at or after from, within RentalLookback of
// now, and has no row yet, to the number's app under OriginNumberRental. A month that began
// before from is never billed, so a first run does not bill what began before it.
func (r Rentals) Bill(ctx context.Context, from, now time.Time) (Rented, error) {
	since := now.UTC().Add(-RentalLookback)
	if from.After(since) {
		since = from.UTC()
	}
	held, err := r.Store.NumbersHeldSince(ctx, since)
	if err != nil {
		return Rented{}, err
	}
	type due struct {
		number store.PhoneNumber
		month  time.Time
	}
	var months []due
	var ids []uuid.UUID
	for _, number := range held {
		if number.Vendor == SIPTrunkVendor {
			continue
		}
		for _, month := range MonthsBegun(number.PurchasedAt, since.Add(-time.Nanosecond), now) {
			// The month it was bought in is the purchase's to bill.
			if !month.After(number.PurchasedAt.UTC()) {
				continue
			}
			if number.ReleasedAt != nil && !month.Before(*number.ReleasedAt) {
				continue
			}
			months = append(months, due{number, month})
			ids = append(ids, numberMonth(number, month).UsageID)
		}
	}
	recorded, err := r.Store.RecordedUsage(ctx, ids)
	if err != nil {
		return Rented{}, err
	}
	var rented Rented
	for i, each := range months {
		if recorded[ids[i]] {
			continue
		}
		billMonth(r.Recorder, r.Logger, each.number, routing.Owner{
			CustomerID:  each.number.CustomerID,
			Attribution: routing.Attribution{APIKey: OriginNumberRental},
			Tags:        each.number.Tags,
		}, each.month)
		rented.Months++
		if each.number.MonthlyCostMicros <= 0 {
			rented.Unpriced = append(rented.Unpriced, each.number.Vendor+" "+each.number.E164)
		}
	}
	return rented, nil
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
