package pricing

import (
	"cmp"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"
	"time"
)

// key is the item an entry prices, without the attributes that choose between its entries.
type key struct {
	modality, provider, model, region string
	byok                              bool
}

func keyOf(item Item) key { return key{item.Modality, item.Provider, item.Model, item.Region, item.BYOK} }

func keyOfEntry(entry Entry) key {
	return key{entry.Modality, entry.Provider, entry.Model, entry.Region, entry.BYOK}
}

// List is a price list: one version of the entries every item is priced by.
type List struct {
	version string
	entries map[key][]Entry
	logger  *slog.Logger
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
// quote. An entry with no SKU, as a price file may leave it, is given its BaseSKU, numbered
// past any SKU another entry of the list has.
func NewList(version string, entries []Entry, logger *slog.Logger) (*List, error) {
	if logger == nil {
		logger = slog.Default()
	}
	list := &List{version: version, entries: map[key][]Entry{}, logger: logger, unpriced: newUnpricedLog()}
	skus := map[string]string{}
	for _, entry := range entries {
		if err := Validate(entry); err != nil {
			return nil, err
		}
		item := keyOfEntry(entry)
		for _, other := range list.entries[item] {
			if ambiguous(entry.When, other.When) {
				return nil, fmt.Errorf("pricing: %s has two entries an item with %v would match equally", name(entry), entry.When)
			}
		}
		if entry.SKU != "" {
			if other, ok := skus[entry.SKU]; ok {
				return nil, fmt.Errorf("pricing: %s and %s both have the SKU %s", other, nameWithWhen(entry), entry.SKU)
			}
			skus[entry.SKU] = nameWithWhen(entry)
		}
		list.entries[item] = append(list.entries[item], entry)
	}
	for _, item := range slices.SortedFunc(maps.Keys(list.entries), compareKeys) {
		for i, entry := range list.entries[item] {
			if entry.SKU != "" {
				continue
			}
			base := skuOf(entry)
			sku := base
			for n := 2; skus[sku] != ""; n++ {
				sku = NumberedSKU(base, n)
			}
			skus[sku] = nameWithWhen(entry)
			list.entries[item][i].SKU = sku
		}
	}
	return list, nil
}

func compareKeys(a, b key) int {
	return cmp.Or(strings.Compare(a.modality, b.modality), strings.Compare(a.provider, b.provider), strings.Compare(a.model, b.model),
		strings.Compare(a.region, b.region), compareBYOK(a.byok, b.byok))
}

// compareBYOK puts an entry that is not BYOK before its BYOK twin.
func compareBYOK(a, b bool) int {
	switch {
	case a == b:
		return 0
	case b:
		return -1
	}
	return 1
}

// Errors Validate returns for an entry whose price is not in the shape of its vendor cost.
var (
	// ErrPriceMeters is a price whose meters are not the vendor rates' meters.
	ErrPriceMeters = errors.New("pricing: the prices do not have the meters the rates have")
	// ErrPriceTiers is a price whose tiers are not the vendor rates' tiers.
	ErrPriceTiers = errors.New("pricing: the price tiers are not the tiers the rates have")
	// ErrRegion is an entry that names no region, or one that is not a region.
	ErrRegion = errors.New("pricing: every entry names its region, such as us")
)

// Validate checks one entry: an item in a region, an SKU of the right form, rates and prices
// within bounds, and a price with the meters of the vendor rates and a price tier for each of
// their tiers, at the same threshold and with the same meters. A free entry carries no price.
func Validate(entry Entry) error {
	if entry.Modality == "" || entry.Provider == "" || entry.Model == "" {
		return errors.New("pricing: every entry needs a modality, a provider and a model")
	}
	switch {
	case entry.Region == "":
		return fmt.Errorf("%w: %s names none", ErrRegion, name(entry))
	case !ValidRegion(entry.Region):
		return fmt.Errorf("%w: %s names %q, and a region is a letter and up to 31 of a-z, 0-9 and hyphens",
			ErrRegion, name(entry), entry.Region)
	}
	if entry.SKU != "" && !ValidSKU(entry.SKU) {
		return fmt.Errorf("pricing: %s has the SKU %q: a SKU is 1 to %d characters of a-z and 0-9 in runs joined by single hyphens",
			name(entry), entry.SKU, MaxSKULength)
	}
	for _, rates := range [][]Tier{{{Rates: entry.Rates}}, entry.Tiers, {{Rates: entry.Prices}}, entry.PriceTiers} {
		for _, tier := range rates {
			if err := validateRates(entry, tier.Rates); err != nil {
				return err
			}
		}
	}
	if err := validateTiers(entry, entry.Rates, entry.Tiers); err != nil {
		return err
	}
	if err := validateTiers(entry, entry.Prices, entry.PriceTiers); err != nil {
		return err
	}
	if entry.Free {
		if len(entry.Prices) > 0 || len(entry.PriceTiers) > 0 {
			return fmt.Errorf("pricing: %s is free and has a price: a free entry carries none", name(entry))
		}
		return nil
	}
	if !sameMeters(entry.Rates, entry.Prices) {
		return fmt.Errorf("%w: %s rates %s and prices %s", ErrPriceMeters, name(entry), meterNames(entry.Rates), meterNames(entry.Prices))
	}
	if len(entry.Tiers) != len(entry.PriceTiers) {
		return fmt.Errorf("%w: %s has %d tiers of rates and %d of prices", ErrPriceTiers, name(entry), len(entry.Tiers), len(entry.PriceTiers))
	}
	for i, tier := range entry.Tiers {
		price := entry.PriceTiers[i]
		if price.AboveInputTokens != tier.AboveInputTokens {
			return fmt.Errorf("%w: %s has a tier of rates above %d input tokens and its price tier is above %d",
				ErrPriceTiers, name(entry), tier.AboveInputTokens, price.AboveInputTokens)
		}
		if !sameMeters(tier.Rates, price.Rates) {
			return fmt.Errorf("%w: %s above %d input tokens rates %s and prices %s",
				ErrPriceTiers, name(entry), tier.AboveInputTokens, meterNames(tier.Rates), meterNames(price.Rates))
		}
	}
	return nil
}

// validateTiers checks that each tier is above the one before it and rates every meter the
// entry does.
func validateTiers(entry Entry, rates map[Meter]Rate, tiers []Tier) error {
	var above int64
	for _, tier := range tiers {
		if tier.AboveInputTokens <= above {
			return fmt.Errorf("pricing: %s has a tier above %d input tokens, which is not above the one before it",
				name(entry), tier.AboveInputTokens)
		}
		above = tier.AboveInputTokens
		for meter := range rates {
			if _, ok := tier.Rates[meter]; !ok {
				return fmt.Errorf("pricing: %s has a tier above %d input tokens with no %s rate", name(entry), tier.AboveInputTokens, meter)
			}
		}
	}
	return nil
}

func sameMeters(a, b map[Meter]Rate) bool {
	if len(a) != len(b) {
		return false
	}
	for meter := range a {
		if _, ok := b[meter]; !ok {
			return false
		}
	}
	return true
}

func meterNames(rates map[Meter]Rate) string {
	if len(rates) == 0 {
		return "nothing"
	}
	names := make([]string, 0, len(rates))
	for _, meter := range slices.Sorted(maps.Keys(rates)) {
		names = append(names, string(meter))
	}
	return strings.Join(names, ", ")
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

// name is the item an entry prices: modality/provider/model, then its region and byok for a
// BYOK entry, as in llm/openai/m (us, byok).
func name(entry Entry) string {
	var qualifiers []string
	if entry.Region != "" {
		qualifiers = append(qualifiers, entry.Region)
	}
	if entry.BYOK {
		qualifiers = append(qualifiers, "byok")
	}
	written := entry.Modality + "/" + entry.Provider + "/" + entry.Model
	if len(qualifiers) > 0 {
		written += " (" + strings.Join(qualifiers, ", ") + ")"
	}
	return written
}

// nameWithWhen is the item an entry prices, its When included.
func nameWithWhen(entry Entry) string {
	if len(entry.When) == 0 {
		return name(entry)
	}
	pairs := make([]string, 0, len(entry.When))
	for _, attribute := range slices.Sorted(maps.Keys(entry.When)) {
		pairs = append(pairs, attribute+"="+entry.When[attribute])
	}
	return name(entry) + " when " + strings.Join(pairs, ",")
}

// match is the item's entry whose conditions all hold in its attributes, with the most of
// them, among the entries of its own region and BYOK state.
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
		entry = Entry{Modality: item.Modality, Provider: item.Provider, Model: item.Model, Region: item.Region, BYOK: item.BYOK}
	}
	quote := entry.quote(used)
	quote.Version = l.versionOf(entry, ok)
	if quote.Unpriced {
		l.warn(item)
	}
	return quote
}

// Quoted resells an amount the vendor quoted outright for the item, such as a number's
// monthly rental, which no price prices: the amount times PriceFactorBps, under the SKU of the
// item's entry when it has one. A BYOK item's vendor bills the customer the amount, so it is
// charged BYOKFeeBps of it and costs the operator nothing.
func (l *List) Quoted(item Item, micros int64) Quote {
	entry, _ := l.match(item)
	bps, base := float64(PriceFactorBps), micros
	if item.BYOK {
		bps, base = BYOKFeeBps, 0
	}
	return Quote{
		CostMicros:     roundHalfUp(float64(micros) * bps / 10_000),
		BaseCostMicros: base,
		SKU:            entry.SKU,
		Version:        QuotedVersion,
		Billable:       micros > 0,
	}
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
	l.logger.Error("billable work has no price", "modality", item.Modality, "provider", item.Provider, "model", item.Model,
		"region", item.Region, "byok", item.BYOK)
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
	return cloned(entry), true
}

// cloned is a copy of the entry that shares nothing with it.
func cloned(entry Entry) Entry {
	entry.When = maps.Clone(entry.When)
	entry.Rates = maps.Clone(entry.Rates)
	entry.Tiers = clonedTiers(entry.Tiers)
	entry.Prices = maps.Clone(entry.Prices)
	entry.PriceTiers = clonedTiers(entry.PriceTiers)
	return entry
}

func clonedTiers(tiers []Tier) []Tier {
	tiers = slices.Clone(tiers)
	for i := range tiers {
		tiers[i].Rates = maps.Clone(tiers[i].Rates)
	}
	return tiers
}

// Entries is a copy of every entry of the list, by modality, provider, model, region, the entry
// that is not BYOK before its twin, and then the fewest conditions.
func (l *List) Entries() []Entry {
	var entries []Entry
	for _, item := range l.entries {
		for _, entry := range item {
			entries = append(entries, cloned(entry))
		}
	}
	slices.SortFunc(entries, func(a, b Entry) int {
		return cmp.Or(compareKeys(keyOfEntry(a), keyOfEntry(b)), cmp.Compare(len(a.When), len(b.When)),
			strings.Compare(fmt.Sprint(a.When), fmt.Sprint(b.When)))
	})
	return entries
}

// Unpriced is the items that are billable and have no price: what a deployment should
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
