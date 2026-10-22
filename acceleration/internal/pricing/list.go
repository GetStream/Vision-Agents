package pricing

import (
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"time"
)

// key is the item an entry prices, without the attributes that choose between its entries.
type key struct {
	modality, provider, model string
}

func keyOf(item Item) key { return key{item.Modality, item.Provider, item.Model} }

// List is a price list: one version of the entries every item is priced by.
type List struct {
	version string
	entries map[key][]Entry
	logger  *slog.Logger
	// warned is when each unpriced item was last logged, so a busy one is not logged on
	// every row. It is logged again once unpricedEvery has passed, by now.
	warned        sync.Map
	unpricedEvery time.Duration
	now           func() time.Time
}

// NewList checks the entries and returns the list they make. version names it on every
// quote.
func NewList(version string, entries []Entry, logger *slog.Logger) (*List, error) {
	if logger == nil {
		logger = slog.Default()
	}
	list := &List{version: version, entries: map[key][]Entry{}, logger: logger, unpricedEvery: 10 * time.Minute, now: time.Now}
	for _, entry := range entries {
		if err := validate(entry); err != nil {
			return nil, err
		}
		item := key{entry.Modality, entry.Provider, entry.Model}
		for _, other := range list.entries[item] {
			if ambiguous(entry.When, other.When) {
				return nil, fmt.Errorf("pricing: %s has two entries an item with %v would match equally", name(entry), entry.When)
			}
		}
		list.entries[item] = append(list.entries[item], entry)
	}
	return list, nil
}

func validate(entry Entry) error {
	if entry.Modality == "" || entry.Provider == "" || entry.Model == "" {
		return errors.New("pricing: every entry needs a modality, a provider and a model")
	}
	if err := validateRates(entry, entry.Rates); err != nil {
		return err
	}
	var above int64
	for _, tier := range entry.Tiers {
		if tier.AboveInputTokens <= above {
			return fmt.Errorf("pricing: %s has a tier above %d input tokens, which is not above the one before it",
				name(entry), tier.AboveInputTokens)
		}
		above = tier.AboveInputTokens
		if err := validateRates(entry, tier.Rates); err != nil {
			return err
		}
		for meter := range entry.Rates {
			if _, ok := tier.Rates[meter]; !ok {
				return fmt.Errorf("pricing: %s has a tier above %d input tokens with no %s rate", name(entry), tier.AboveInputTokens, meter)
			}
		}
	}
	return nil
}

func validateRates(entry Entry, rates map[Meter]Rate) error {
	for _, meter := range slices.Sorted(maps.Keys(rates)) {
		rate := rates[meter]
		switch {
		case !slices.Contains(meters, meter):
			return fmt.Errorf("pricing: %s rates %q, which is not a meter", name(entry), meter)
		case rate.Per <= 0:
			return fmt.Errorf("pricing: %s rates %s per %d units; a rate needs a unit", name(entry), meter, rate.Per)
		case rate.USD < 0 || rate.Increment < 0 || rate.Minimum < 0:
			return fmt.Errorf("pricing: %s rates %s below zero", name(entry), meter)
		}
	}
	return nil
}

// ambiguous reports whether some item could match both conditions with as many pairs.
func ambiguous(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for attribute, value := range a {
		if other, ok := b[attribute]; ok && other != value {
			return false
		}
	}
	return true
}

func name(entry Entry) string {
	return entry.Modality + "/" + entry.Provider + "/" + entry.Model
}

// match is the item's entry whose conditions all hold in its attributes, with the most of
// them.
func (l *List) match(item Item) (Entry, bool) {
	var found Entry
	ok := false
	for _, entry := range l.entries[keyOf(item)] {
		if !holds(entry.When, item.Attributes) {
			continue
		}
		if !ok || len(entry.When) > len(found.When) {
			found, ok = entry, true
		}
	}
	return found, ok
}

func holds(when, attributes map[string]string) bool {
	for attribute, value := range when {
		if attributes[attribute] != value {
			return false
		}
	}
	return true
}

// Quote prices used at the item's entry. Billable work with no rate costs nothing and is
// logged, at most once per item every ten minutes.
func (l *List) Quote(item Item, used Quantities) Quote {
	entry, ok := l.match(item)
	if !ok {
		entry = Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model}
	}
	quote := entry.Quote(used)
	quote.Version = l.version
	if quote.Unpriced {
		l.warn(item)
	}
	return quote
}

// warn logs an unpriced item unless it was logged within unpricedEvery. Rows are priced
// from many goroutines, so the time is swapped in rather than read and then written.
func (l *List) warn(item Item) {
	now := l.now()
	k := keyOf(item)
	for {
		last, seen := l.warned.LoadOrStore(k, now)
		if !seen {
			break
		}
		if now.Sub(last.(time.Time)) < l.unpricedEvery {
			return
		}
		if l.warned.CompareAndSwap(k, last, now) {
			break
		}
	}
	l.logger.Error("billable work has no price", "modality", item.Modality, "provider", item.Provider, "model", item.Model)
}

// Billable reports whether work on the item counts toward spend.
func (l *List) Billable(item Item) bool {
	entry, ok := l.match(item)
	return !ok || entry.Billable()
}

// Entry is a copy of the item's entry: the list's own rates are read by concurrent quotes.
func (l *List) Entry(item Item) (Entry, bool) {
	entry, ok := l.match(item)
	if !ok {
		return Entry{}, false
	}
	entry.When = maps.Clone(entry.When)
	entry.Rates = maps.Clone(entry.Rates)
	entry.Tiers = slices.Clone(entry.Tiers)
	for i := range entry.Tiers {
		entry.Tiers[i].Rates = maps.Clone(entry.Tiers[i].Rates)
	}
	return entry, true
}

// Unpriced is the items that are billable and have no rate: what a deployment should
// price, or mark free or local.
func (l *List) Unpriced(items []Item) []Item {
	var unpriced []Item
	for _, item := range items {
		entry, ok := l.match(item)
		if !ok || (!entry.priced() && entry.Billable()) {
			unpriced = append(unpriced, item)
		}
	}
	return unpriced
}
