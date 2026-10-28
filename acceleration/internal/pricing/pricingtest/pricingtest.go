// Package pricingtest works out what a test expects work to cost from the price list it is
// billed with, so that a price change in vendor.json leaves the test passing.
package pricingtest

import (
	"maps"
	"math"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// Micros is what used costs at the prices list gives item, in micro-dollars. It is worked out
// from the rates' own numbers rather than by the list, so a fault in how the list prices still
// fails the test: a meter with no price costs nothing, each other is rounded up to its increment
// and then its minimum, and the sum is rounded half up once, as a row is. Left out are tiers, so
// used must stay below them, and a row's exception of a meter at zero from its minimum.
func Micros(list *pricing.List, item pricing.Item, used pricing.Quantities) int64 {
	// An item the list lacks costs nothing here, and so fails against the row the list priced.
	entry, _ := list.Entry(item)
	var micros float64
	for _, meter := range slices.Sorted(maps.Keys(used)) {
		rate, ok := pricing.RateOf(entry.Prices, meter)
		if !ok {
			continue
		}
		// NewList has refused a rate per nothing, so Per is above zero.
		billed := used[meter]
		if rate.Increment > 0 && billed%rate.Increment != 0 {
			billed += rate.Increment - billed%rate.Increment
		}
		billed = max(billed, rate.Minimum)
		micros += float64(billed) * rate.USD * 1e6 / float64(rate.Per)
	}
	return int64(math.Floor(micros + 0.5))
}
