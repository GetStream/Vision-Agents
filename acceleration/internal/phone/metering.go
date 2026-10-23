package phone

import (
	"context"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The models telephony rows are priced as, under the vendor that did the work.
const (
	// ModelDial is placing a call or a transfer: free, since a vendor bills a call by its
	// minutes, and kept for the latency and the failures it records.
	ModelDial = "dial"
	// ModelCallMinutes is the minutes of one leg of a call, by country and direction.
	ModelCallMinutes = "call_minutes"
	// ModelNumber is a month of a number, at the price the vendor quoted when it was bought.
	ModelNumber = "number"
)

// The attributes telephony rows are priced and explained by.
const (
	AttributeCountry   = "country"
	AttributeDirection = "direction"
	// AttributeDLCSandbox is "true" on the row of a text or a call of an app the 10DLC sandbox
	// holds, which is priced as usual and given away.
	AttributeDLCSandbox = "dlc_sandbox"
)

// Used is nothing: a call is billed by the minutes of its legs, not by being placed.
func (Dialed) Used() pricing.Quantities { return pricing.Quantities{} }

// Used is nothing: a number is billed by the months it is held, at the price it was bought at.
func (Number) Used() pricing.Quantities { return pricing.Quantities{} }

// vendorFailed is the error code of a vendor that refused or did not answer.
func vendorFailed(error) string { return "vendor_failed" }

// dialMeter records placing a call or a transfer through a vendor.
func (s *Service) dialMeter(vendor string, owner routing.Owner) routing.Meter {
	return routing.Meter{
		Recorder: s.recorder,
		Config:   routing.ProviderConfig{Provider: vendor, Model: ModelDial},
		Owner:    owner,
		Code:     vendorFailed,
	}
}

// numberMonth is the usage id of one month of a number: the same month of the same purchase
// is billed once, however many times it is recorded.
func numberMonth(number store.PhoneNumber, month time.Time) routing.Meter {
	return routing.Meter{
		Config: routing.ProviderConfig{Provider: number.Vendor, Model: ModelNumber},
		UsageID: routing.UsageID("number", number.Vendor, number.E164,
			strconv.FormatInt(number.PurchasedAt.UTC().UnixMicro(), 10), month.UTC().Format("2006-01")),
		CostMicros: number.MonthlyCostMicros,
		BilledAt:   month,
		Attributes: map[string]string{AttributeCountry: number.Country},
		Unmeasured: true,
	}
}

// sandboxed reports whether the 10DLC sandbox holds the app, so its texts and calls are
// given away. One that cannot be read is billed, and said so.
func (s *Service) sandboxed(ctx context.Context, customerID string) bool {
	held, err := s.gate.Sandboxed(ctx, customerID)
	if err != nil {
		s.logger.Warn("could not tell whether the 10DLC sandbox holds an app; its minutes are billed",
			"customer", customerID, "error", err)
	}
	return held
}
