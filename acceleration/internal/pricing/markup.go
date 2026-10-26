package pricing

import "strconv"

// MaxMarkupBps is the largest markup a list or an entry may carry: ten times the vendor's
// rate, past which a markup is a typo rather than a price.
const MaxMarkupBps = 100_000

// MaxUSDPerUnit is the most one unit of any meter may cost: a token, a character, a second, an
// image or a request. The dearest unit any vendor lists costs cents; a rate past this is a
// typo, and one that is not a number could price nothing.
const MaxUSDPerUnit = 1_000

// chargedDigits is how many significant digits a charged rate keeps: a rate a markup
// multiplied reads as a person would write it (1.375, never 1.3750000000000002), and twelve
// digits is far below a millionth of a dollar on any usage a row records.
const chargedDigits = 12

// ValidMarkup reports whether a markup in basis points is one a list or an entry may carry.
func ValidMarkup(bps int64) bool {
	return bps >= 0 && bps <= MaxMarkupBps
}

// Charged is the entry as customers are charged under a list with that markup: every rate
// and tier rate times (10000 + markup) / 10000, the entry's own markup when it has one, kept
// to twelve significant digits. A quote never prices from it: Quote adds the markup to the
// unrounded cost at the vendor's rates, which agrees with these rates to far below a micro.
func Charged(entry Entry, listMarkupBps int64) Entry {
	markup := markupOf(entry, listMarkupBps)
	scale := func(rates map[Meter]Rate) map[Meter]Rate {
		if rates == nil {
			return nil
		}
		scaled := make(map[Meter]Rate, len(rates))
		for meter, rate := range rates {
			rate.USD = significant(marked(rate.USD, markup))
			scaled[meter] = rate
		}
		return scaled
	}
	charged := entry
	charged.Rates = scale(entry.Rates)
	charged.Tiers = nil
	for _, tier := range entry.Tiers {
		charged.Tiers = append(charged.Tiers, Tier{AboveInputTokens: tier.AboveInputTokens, Rates: scale(tier.Rates)})
	}
	return charged
}

// significant is usd to chargedDigits significant digits.
func significant(usd float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(usd, 'g', chargedDigits, 64), 64)
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
