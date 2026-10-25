package routing

import (
	"log/slog"
	"maps"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// Prices is the price list for work on the config's providers when nobody passed one: the
// rates the config sets itself, then the vendor list embedded in the binary, with the
// providers it marks local, with no markup. A router passes the list it was given instead.
func Prices(config Config, logger *slog.Logger) (*pricing.List, error) {
	var own []pricing.Entry
	for _, entry := range PriceEntries(config) {
		if entry.Free || len(entry.Rates) > 0 || len(entry.Tiers) > 0 {
			entry.Local = false
			entry.Version = "config"
			own = append(own, entry)
		}
	}
	entries := prices.Merge(prices.File{Entries: own}, prices.Vendor())
	return pricing.NewList(prices.EmbeddedVersion(""), prices.WithLocal(entries, LocalItems(config)), 0, logger)
}

// PriceEntries is the price list the config's price: blocks declare, at the vendors' rates:
// one entry per provider in every modality, and one more for each service tier a provider
// names.
func PriceEntries(config Config) []pricing.Entry {
	var entries []pricing.Entry
	for _, modality := range slices.Sorted(maps.Keys(config)) {
		for _, provider := range config[modality].Providers {
			entry := entryOf(modality, provider)
			entries = append(entries, entry)
			tiers := provider.Price.ServiceTiers
			for _, tier := range slices.Sorted(maps.Keys(tiers)) {
				entries = append(entries, atServiceTier(entry, tier, tiers[tier]))
			}
		}
	}
	return entries
}

// entryOf is a provider's default entry. A rate of zero is left out, so it falls back to
// the rate it would fall back to rather than pricing at nothing.
func entryOf(modality Modality, config ProviderConfig) pricing.Entry {
	price := config.Price
	rates := map[pricing.Meter]pricing.Rate{}
	rate := func(meter pricing.Meter, usd float64, per int64) {
		if usd != 0 {
			rates[meter] = pricing.Rate{USD: usd, Per: per}
		}
	}
	rate(pricing.Characters, price.PerMillionChars, 1_000_000)
	rate(pricing.AudioMs, price.PerAudioHour, 3_600_000)
	rate(pricing.InputTokens, price.PerMillionInputTokens, 1_000_000)
	rate(pricing.CachedInputTokens, price.PerMillionCachedInputTokens, 1_000_000)
	rate(pricing.CacheWriteInputTokens, price.PerMillionCacheWriteInputTokens, 1_000_000)
	rate(pricing.OutputTokens, price.PerMillionOutputTokens, 1_000_000)
	rate(pricing.Requests, price.PerThousandRequests, 1_000)
	rate(pricing.Images, price.PerImage, 1)
	rate(pricing.Pixels, price.PerMegapixel, 1_000_000)
	maps.Copy(rates, price.Rates)

	entry := pricing.Entry{
		Modality: string(modality),
		Provider: config.Provider,
		Model:    config.Model,
		Free:     price.Free,
		Local:    config.Local,
	}
	if len(rates) > 0 {
		entry.Rates = rates
	}
	for _, tier := range price.ContextTiers {
		entry.Tiers = append(entry.Tiers, pricing.Tier{
			AboveInputTokens: tier.AboveInputTokens,
			Rates:            atContextTier(rates, tier),
		})
	}
	return entry
}

// atContextTier is the rates multiplied as the tier says: input and output by their own
// factors, anything else as it is.
func atContextTier(rates map[pricing.Meter]pricing.Rate, tier ContextTier) map[pricing.Meter]pricing.Rate {
	scaled := make(map[pricing.Meter]pricing.Rate, len(rates))
	for meter, rate := range rates {
		switch {
		case tier.Input != 0 && slices.Contains(pricing.InputMeters, meter):
			rate.USD *= tier.Input
		case tier.Output != 0 && slices.Contains(pricing.OutputMeters, meter):
			rate.USD *= tier.Output
		}
		scaled[meter] = rate
	}
	return scaled
}

// atServiceTier is the entry for work the vendor served at that tier: every rate, its
// context tiers' included, times the factor.
func atServiceTier(entry pricing.Entry, tier string, factor float64) pricing.Entry {
	scale := func(rates map[pricing.Meter]pricing.Rate) map[pricing.Meter]pricing.Rate {
		if rates == nil {
			return nil
		}
		scaled := make(map[pricing.Meter]pricing.Rate, len(rates))
		for meter, rate := range rates {
			rate.USD *= factor
			scaled[meter] = rate
		}
		return scaled
	}
	tiered := entry
	tiered.When = map[string]string{"service_tier": tier}
	tiered.Rates = scale(entry.Rates)
	tiered.Tiers = nil
	for _, long := range entry.Tiers {
		tiered.Tiers = append(tiered.Tiers, pricing.Tier{AboveInputTokens: long.AboveInputTokens, Rates: scale(long.Rates)})
	}
	return tiered
}

// ItemOf is what work on a provider of a modality is priced as.
func ItemOf(modality Modality, config ProviderConfig, attributes map[string]string) pricing.Item {
	return pricing.Item{Modality: string(modality), Provider: config.Provider, Model: config.Model, Attributes: attributes}
}

// Items is one item, with no attributes, per provider in every modality: what a deployment
// checks for work it would bill with no price.
func Items(config Config) []pricing.Item {
	var items []pricing.Item
	for _, modality := range slices.Sorted(maps.Keys(config)) {
		for _, provider := range config[modality].Providers {
			items = append(items, ItemOf(modality, provider, nil))
		}
	}
	return items
}

// LocalItems is the items whose provider this deployment runs itself.
func LocalItems(config Config) []pricing.Item {
	var items []pricing.Item
	for _, modality := range slices.Sorted(maps.Keys(config)) {
		for _, provider := range config[modality].Providers {
			if provider.Local {
				items = append(items, ItemOf(modality, provider, nil))
			}
		}
	}
	return items
}

// quote prices what one unit of work used. A customer's own model carries its price in its
// config: its name is the customer's, so no list shared by every customer can hold it. Its
// cost is what the customer's host bills them, so it is shown but counts toward no spend,
// and a model with no rate is free rather than unpriced.
func quote(pricer pricing.Pricer, modality Modality, config ProviderConfig, attributes map[string]string, used pricing.Quantities) pricing.Quote {
	if config.Provider == CustomProvider {
		own := entryOf(modality, config).Quote(used)
		own.Billable, own.Unpriced = false, false
		return own
	}
	return pricer.Quote(ItemOf(modality, config, attributes), used)
}
