// Package pricingtest works out what a test expects work to cost from the price list it is
// billed with, so that a price change in vendor.json leaves the test passing, and holds the
// price file the suites lay over vendor.json for the regions and BYOK entries it lacks.
package pricingtest

import (
	_ "embed"
	"log/slog"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
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

// pricesJSON is a deployment's price file, as ROUTER_PRICES_FILE names one. vendor.json is all
// us; this file prices the LLM, Knowledge and Sandbox items in the eu too, each at rates of its
// own, dearer than its us entry's, and holds BYOK entries priced outright: Sonnet's in the us
// and Knowledge's in the eu.
//
//go:embed prices.json
var pricesJSON []byte

// FileName is what the fixtures' price file is called in a change's source and in an error.
const FileName = "pricingtest/prices.json"

// File is the fixtures' price file, read as a router reads a deployment's own: every entry of
// an llm, knowledge or sandbox item it gives no BYOK entry has a BYOK twin, and every entry with
// no price of its own is priced.
func File(t testing.TB) prices.File {
	t.Helper()
	file, err := prices.Parse(FileName, pricesJSON)
	if err != nil {
		t.Fatalf("pricingtest: %v", err)
	}
	return file
}

// Entries is File laid over vendor.json, as a router's start lays a deployment's file over it:
// what a suite seeds its database or builds its list with.
func Entries(t testing.TB) []pricing.Entry {
	t.Helper()
	return prices.Merge(File(t), prices.Vendor())
}

// List is the price list of Entries, naming version on every quote.
func List(t testing.TB, version string) *pricing.List {
	t.Helper()
	list, err := pricing.NewList(version, Entries(t), slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatalf("pricingtest: %v", err)
	}
	return list
}

// LLM is Claude Haiku in region, with the customer's own key or without: priced in the us and
// the eu, each with its BYOK twin.
func LLM(region string, byok bool) pricing.Item {
	return pricing.Item{Modality: "llm", Provider: "anthropic", Model: "claude-haiku-5-5", Region: region, BYOK: byok}
}

// OwnBYOKLLM is Claude Sonnet with the customer's own key in the us, whose BYOK entry File
// prices outright rather than at pricing.BYOKFeeBps of its rates.
func OwnBYOKLLM() pricing.Item {
	return pricing.Item{Modality: "llm", Provider: "anthropic", Model: "claude-sonnet-5-5", Region: pricing.RegionUS, BYOK: true}
}

// Knowledge is a write to turbopuffer in region: priced in the us and the eu. Its eu BYOK entry
// is priced outright in File; its us one is vendor.json's twin.
func Knowledge(region string, byok bool) pricing.Item {
	return pricing.Item{Modality: "knowledge", Provider: "turbopuffer", Model: "write", Region: region, BYOK: byok}
}

// Sandbox is a running Linux container on Daytona in region: priced in the us and the eu, each
// with its BYOK twin.
func Sandbox(region string, byok bool) pricing.Item {
	return pricing.Item{Modality: "sandbox", Provider: "daytona", Model: "running", Region: region, BYOK: byok,
		Attributes: map[string]string{"class": "container"}}
}
