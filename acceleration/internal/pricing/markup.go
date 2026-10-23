package pricing

// MaxMarkupBps is the largest markup a list or an entry may carry: ten times the vendor's
// rate, past which a markup is a typo rather than a price.
const MaxMarkupBps = 100_000

// ValidMarkup reports whether a markup in basis points is one a list or an entry may carry.
func ValidMarkup(bps int64) bool {
	return bps >= 0 && bps <= MaxMarkupBps
}

// Charged is the entry as customers are charged under a list with that markup: every rate
// and tier rate times (10000 + markup) / 10000, the entry's own markup when it has one.
func Charged(entry Entry, listMarkupBps int64) Entry {
	markup := listMarkupBps
	if entry.MarkupBps != nil {
		markup = *entry.MarkupBps
	}
	factor := float64(10_000+markup) / 10_000
	scale := func(rates map[Meter]Rate) map[Meter]Rate {
		if rates == nil {
			return nil
		}
		scaled := make(map[Meter]Rate, len(rates))
		for meter, rate := range rates {
			rate.USD *= factor
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
