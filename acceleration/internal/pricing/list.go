package pricing

import (
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
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
	// markup is the list's markup in basis points, added to every entry with none of its own.
	markup int64
	logger *slog.Logger
	// unpriced is when each unpriced item was last logged. Lists read one after another
	// share it (Active), so a list read again does not log the same item again.
	unpriced *unpricedLog
}

// unpricedLog is when each unpriced item was last logged, so a busy one is not logged on
// every row. It is logged again once every has passed, by now.
type unpricedLog struct {
	warned sync.Map
	every  time.Duration
	now    func() time.Time
}

func newUnpricedLog() *unpricedLog {
	return &unpricedLog{every: 10 * time.Minute, now: time.Now}
}

// NewList checks the entries and returns the list they make. version names it on every
// quote, and markupBps is added to every entry with no markup of its own: 0 sells at the
// vendors' rates.
func NewList(version string, entries []Entry, markupBps int64, logger *slog.Logger) (*List, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if !ValidMarkup(markupBps) {
		return nil, fmt.Errorf("pricing: a list markup of %d basis points, outside 0 to %d", markupBps, MaxMarkupBps)
	}
	list := &List{version: version, entries: map[key][]Entry{}, markup: markupBps, logger: logger, unpriced: newUnpricedLog()}
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
	if entry.MarkupBps != nil && !ValidMarkup(*entry.MarkupBps) {
		return fmt.Errorf("pricing: %s has a markup of %d basis points, outside 0 to %d", name(entry), *entry.MarkupBps, MaxMarkupBps)
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
		case math.IsNaN(rate.USD) || math.IsInf(rate.USD, 0) || rate.USD/float64(rate.Per) > MaxUSDPerUnit:
			return fmt.Errorf("pricing: %s rates %s at more than $%d a unit", name(entry), meter, MaxUSDPerUnit)
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
	quote := entry.quote(used, markupOf(entry, l.markup))
	quote.Version = l.versionOf(entry, ok)
	if quote.Unpriced {
		l.warn(item)
	}
	return quote
}

// Quoted resells an amount the vendor quoted outright for the item, such as a number's
// monthly rental, which no rate prices: the amount plus the markup the item's entry is sold
// at, the list's when it has none of its own or no entry.
func (l *List) Quoted(item Item, micros int64) Quote {
	entry, _ := l.match(item)
	markup := markupOf(entry, l.markup)
	cost := roundHalfUp(marked(float64(micros), markup))
	return Quote{
		CostMicros:     cost,
		BaseCostMicros: micros,
		MarkupBps:      markup,
		Version:        QuotedVersion,
		Billable:       micros > 0,
	}
}

// MarkupBps is the list's markup in basis points.
func (l *List) MarkupBps() int64 {
	return l.markup
}

// versionOf is what a quote at the entry names as its price: the change that wrote the
// entry, else the list's own version. Work no entry matched, and a local entry with no rate,
// were priced by nothing, so they name nothing.
func (l *List) versionOf(entry Entry, matched bool) string {
	switch {
	case !matched, entry.Local && !entry.priced():
		return ""
	case entry.Version != "":
		return entry.Version
	default:
		return l.version
	}
}

// warn logs an unpriced item unless it was logged within the log's interval. Rows are priced
// from many goroutines, so the time is swapped in rather than read and then written.
func (l *List) warn(item Item) {
	log := l.unpriced
	now := log.now()
	k := keyOf(item)
	for {
		last, seen := log.warned.LoadOrStore(k, now)
		if !seen {
			break
		}
		if now.Sub(last.(time.Time)) < log.every {
			return
		}
		if log.warned.CompareAndSwap(k, last, now) {
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
	if entry.MarkupBps != nil {
		markup := *entry.MarkupBps
		entry.MarkupBps = &markup
	}
	return entry, true
}

// Charged is the item's entry at the rates customers are charged under this list's markup,
// which is what a customer is shown: never the vendor's rates.
func (l *List) Charged(item Item) (Entry, bool) {
	entry, ok := l.Entry(item)
	if !ok {
		return Entry{}, false
	}
	return Charged(entry, l.markup), true
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
