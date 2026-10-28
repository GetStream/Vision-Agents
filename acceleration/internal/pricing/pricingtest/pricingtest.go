// Package pricingtest works out what a test expects work to cost from the price list it is
// billed with, so that a price change in vendor.json leaves the test passing.
package pricingtest

import (
	"maps"
	"math"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// Micros is what used costs at the prices list gives item, in micro-dollars. It is worked out
// from the rates' own numbers rather than by the list, so a fault in how the list prices still
// fails the test: a meter with no price costs nothing, each other is rounded up to its increment
// and its minimum, and the sum is rounded half up once, as a row is. Tiers are left out, so used
// must stay below them.
func Micros(t testing.TB, list *pricing.List, item pricing.Item, used pricing.Quantities) int64 {
	t.Helper()
	entry, ok := list.Entry(item)
	require.True(t, ok, "the list has no price for %s/%s/%s %v", item.Modality, item.Provider, item.Model, item.Attributes)
	var micros float64
	for _, meter := range slices.Sorted(maps.Keys(used)) {
		rate, ok := pricing.RateOf(entry.Prices, meter)
		if !ok {
			continue
		}
		require.Positive(t, rate.Per, "%s/%s prices %s per nothing", item.Provider, item.Model, meter)
		billed := used[meter]
		if rate.Increment > 0 && billed%rate.Increment != 0 {
			billed += rate.Increment - billed%rate.Increment
		}
		if used[meter] > 0 {
			billed = max(billed, rate.Minimum)
		}
		micros += float64(billed) * rate.USD * 1e6 / float64(rate.Per)
	}
	return int64(math.Floor(micros + 0.5))
}
