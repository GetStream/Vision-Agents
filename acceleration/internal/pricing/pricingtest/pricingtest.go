// Package pricingtest works out what a test expects work to cost from the price list it is
// billed with, so that a price change in vendor.json leaves the test passing.
package pricingtest

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// Micros is what used costs at the price list gives item, in micro-dollars: the list's own
// quote, so the test agrees with production to the micro-dollar. The pricing package's tests
// check how a quote is worked out. An item the list has no entry for fails the test, so work
// recorded under the wrong item never passes as costing nothing.
func Micros(t testing.TB, list *pricing.List, item pricing.Item, used pricing.Quantities) int64 {
	t.Helper()
	if _, ok := list.Entry(item); !ok {
		t.Fatalf("the price list has no entry for %+v", item)
	}
	return list.Quote(item, used).CostMicros
}
