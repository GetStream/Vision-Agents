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

// BYOKFeeBps is what work done with the customer's own key is sold at, in basis points of what
// it costs at the vendor's rates, wherever no price is written down: the vendor bills the
// customer for the work, and this is the fee for routing it. A BYOK entry written without a
// price is priced at it once, when it is written, as PriceFactorBps prices every other one.
const BYOKFeeBps = 2_000

// BYOKModalities are the modalities a customer may bring their own key for. Every entry of
// theirs that is not BYOK has a BYOK twin (WithBYOK).
var BYOKModalities = []string{"llm", "knowledge", "sandbox"}

// The regions an entry may name. Every entry and every item names one.
const (
	RegionUS = "us"
	RegionEU = "eu"
)

// DefaultRegion is the region a provider instance runs in when its config names none. It is
// never a fallback when an item is priced: an item of a region no entry names is unpriced.
const DefaultRegion = RegionUS

// regionPattern is what a region is made of: a letter, then up to 31 of a-z, 0-9 and hyphens.
var regionPattern = regexp.MustCompile(`^[a-z][a-z0-9-]{0,31}$`)

// ValidRegion reports whether region is one an entry or a provider may name.
func ValidRegion(region string) bool {
	return regionPattern.MatchString(region)
}

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

// BaseSKU is the SKU an item is first given: its modality, provider, model, each attribute of
// when and its value in attribute order, its region, and byok for a BYOK item, lowercased, with
// every run of characters outside a-z and 0-9 one hyphen. Two items can share it; the one priced
// second takes NumberedSKU.
func BaseSKU(modality, provider, model, region string, byok bool, when map[string]string) string {
	parts := []string{modality, provider, model}
	for _, attribute := range slices.Sorted(maps.Keys(when)) {
		parts = append(parts, attribute, when[attribute])
	}
	parts = append(parts, region)
	if byok {
		parts = append(parts, "byok")
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
	return BaseSKU(entry.Modality, entry.Provider, entry.Model, entry.Region, entry.BYOK, entry.When)
}

// Priced is the entry with a price when it is written without one: every rate and tier rate
// times PriceFactorBps, or BYOKFeeBps for a BYOK entry, kept to twelve significant digits. An
// entry with a price of its own, and a free one, are as they were.
func Priced(entry Entry) Entry {
	if entry.Free || len(entry.Prices) > 0 || len(entry.PriceTiers) > 0 {
		return entry
	}
	bps := float64(PriceFactorBps)
	if entry.BYOK {
		bps = BYOKFeeBps
	}
	entry.Prices = factored(entry.Rates, bps)
	entry.PriceTiers = nil
	for _, tier := range entry.Tiers {
		entry.PriceTiers = append(entry.PriceTiers, Tier{AboveInputTokens: tier.AboveInputTokens, Rates: factored(tier.Rates, bps)})
	}
	return entry
}

// WithBYOK is the entries, each of a modality in BYOKModalities followed by its BYOK twin: the
// same item, region and When, the same rates, tiers and free flag, and no price or SKU, so it
// is priced at BYOKFeeBps and given its own SKU when it is written. An item and When the
// entries already have a BYOK entry for keeps that one.
func WithBYOK(entries []Entry) []Entry {
	written := map[string]bool{}
	for _, entry := range entries {
		if entry.BYOK {
			written[nameWithWhen(entry)] = true
		}
	}
	all := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		all = append(all, entry)
		if entry.BYOK || !slices.Contains(BYOKModalities, entry.Modality) {
			continue
		}
		twin := Entry{
			Modality: entry.Modality,
			Provider: entry.Provider,
			Model:    entry.Model,
			Region:   entry.Region,
			BYOK:     true,
			When:     maps.Clone(entry.When),
			Free:     entry.Free,
			Rates:    maps.Clone(entry.Rates),
			Tiers:    clonedTiers(entry.Tiers),
		}
		if name := nameWithWhen(twin); !written[name] {
			written[name] = true
			all = append(all, twin)
		}
	}
	return all
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

// factored is every rate times bps basis points, to priceDigits significant digits.
func factored(rates map[Meter]Rate, bps float64) map[Meter]Rate {
	if rates == nil {
		return nil
	}
	scaled := make(map[Meter]Rate, len(rates))
	for meter, rate := range rates {
		rate.USD = significant(rate.USD * bps / 10_000)
		scaled[meter] = rate
	}
	return scaled
}

// significant is usd to priceDigits significant digits.
func significant(usd float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(usd, 'g', priceDigits, 64), 64)
	return rounded
}

// WithLocal marks the items a deployment runs itself: each of their entries is local, in every
// region and BYOK or not, and an item with none in its own region gets one there with no rate,
// so its work is free rather than unpriced.
func WithLocal(entries []Entry, local []Item) []Entry {
	all := make([]Entry, len(entries))
	copy(all, entries)
	for _, item := range local {
		found := false
		for i := range all {
			if all[i].Modality == item.Modality && all[i].Provider == item.Provider && all[i].Model == item.Model {
				all[i].Local = true
				found = found || (all[i].Region == item.Region && all[i].BYOK == item.BYOK)
			}
		}
		if !found {
			all = append(all, Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model,
				Region: item.Region, BYOK: item.BYOK, Local: true})
		}
	}
	return all
}
