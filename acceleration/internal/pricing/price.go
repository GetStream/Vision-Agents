package pricing

import (
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// PriceFactorBps is what work is sold at, in basis points of what it costs at the vendor's
// rates, wherever no price is written down: an entry written without a price of its own (a
// price list's first load from the price files, and the items a release adds that the database
// has never priced) is priced at it once, when it is written, and an amount a vendor quotes
// outright is resold at it. No price already written ever moves with it.
const PriceFactorBps = 12_000

// MaxUSDPerUnit is the most one unit of any meter may cost: a token, a character, a second, an
// image or a request. The dearest unit any vendor lists costs cents; a rate past this is a
// typo, and one that is not a number could price nothing.
const MaxUSDPerUnit = 1_000

// priceDigits is how many significant digits a price worked out from a rate keeps: it reads
// as a person would write it (1.375, never 1.3750000000000002), and twelve digits is far below
// a millionth of a dollar on any usage a row records.
const priceDigits = 12

// MaxSKULength is the longest SKU an entry may carry, in bytes.
const MaxSKULength = 255

// skuPattern is what a SKU is made of: runs of a-z and 0-9 joined by single hyphens.
var skuPattern = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// ValidSKU reports whether sku is one an entry may carry.
func ValidSKU(sku string) bool {
	return len(sku) <= MaxSKULength && skuPattern.MatchString(sku)
}

var notInSKU = regexp.MustCompile(`[^a-z0-9]+`)

// BaseSKU is the SKU an item is first given: its modality, provider, model and each attribute
// of when and its value in attribute order, lowercased, with every run of characters outside
// a-z and 0-9 one hyphen. Two items can share it; the one priced second takes NumberedSKU.
func BaseSKU(modality, provider, model string, when map[string]string) string {
	parts := []string{modality, provider, model}
	for _, attribute := range slices.Sorted(maps.Keys(when)) {
		parts = append(parts, attribute, when[attribute])
	}
	sku := strings.Trim(notInSKU.ReplaceAllString(strings.ToLower(strings.Join(parts, "-")), "-"), "-")
	if sku == "" {
		return "sku"
	}
	return sku
}

// NumberedSKU is the n-th SKU an item whose BaseSKU is taken is offered, from n = 2.
func NumberedSKU(base string, n int) string {
	return base + "-" + strconv.Itoa(n)
}

// skuOf is the entry's BaseSKU.
func skuOf(entry Entry) string {
	return BaseSKU(entry.Modality, entry.Provider, entry.Model, entry.When)
}

// Priced is the entry with a price when it is written without one: every rate and tier rate
// times PriceFactorBps, kept to twelve significant digits. An entry with a price of its own,
// and a free one, are as they were.
func Priced(entry Entry) Entry {
	if entry.Free || len(entry.Prices) > 0 || len(entry.PriceTiers) > 0 {
		return entry
	}
	entry.Prices = factored(entry.Rates)
	entry.PriceTiers = nil
	for _, tier := range entry.Tiers {
		entry.PriceTiers = append(entry.PriceTiers, Tier{AboveInputTokens: tier.AboveInputTokens, Rates: factored(tier.Rates)})
	}
	return entry
}

// PricedAll is Priced of every entry.
func PricedAll(entries []Entry) []Entry {
	priced := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		priced = append(priced, Priced(entry))
	}
	return priced
}

// AtCost is the entry charged at its vendor's rates, whatever price it had: how a customer's
// own model is shown, at what its host bills. A free entry is as it was.
func AtCost(entry Entry) Entry {
	if !entry.Free {
		entry.Prices, entry.PriceTiers = entry.Rates, entry.Tiers
	}
	return entry
}

// AllAtCost is AtCost of every entry.
func AllAtCost(entries []Entry) []Entry {
	priced := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		priced = append(priced, AtCost(entry))
	}
	return priced
}

// factored is every rate times PriceFactorBps, to priceDigits significant digits.
func factored(rates map[Meter]Rate) map[Meter]Rate {
	if rates == nil {
		return nil
	}
	scaled := make(map[Meter]Rate, len(rates))
	for meter, rate := range rates {
		rate.USD = significant(rate.USD * PriceFactorBps / 10_000)
		scaled[meter] = rate
	}
	return scaled
}

// significant is usd to priceDigits significant digits.
func significant(usd float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(usd, 'g', priceDigits, 64), 64)
	return rounded
}

// WithLocal marks the items a deployment runs itself: each of their entries is local, and an
// item with none gets one with no rate, so its work is free rather than unpriced.
func WithLocal(entries []Entry, local []Item) []Entry {
	all := make([]Entry, len(entries))
	copy(all, entries)
	for _, item := range local {
		found := false
		for i := range all {
			if all[i].Modality == item.Modality && all[i].Provider == item.Provider && all[i].Model == item.Model {
				all[i].Local, found = true, true
			}
		}
		if !found {
			all = append(all, Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model, Local: true})
		}
	}
	return all
}
