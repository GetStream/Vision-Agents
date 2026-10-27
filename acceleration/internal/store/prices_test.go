//go:build integration

package store

import (
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

func perMillionTokens(usd float64) pricing.Rate { return pricing.Rate{USD: usd, Per: 1_000_000} }

// priced is an LLM entry at input usd per million tokens, its vendor's rate and no price, as a
// price file without a price of its own has it.
func priced(model string, usd float64) pricing.Entry {
	return pricing.Entry{Modality: "llm", Provider: "openai", Model: model,
		Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(usd)}}
}

// load is what a router with these entries in its files loads, under release v1.
func load(entries ...pricing.Entry) PriceLoad {
	return PriceLoad{Entries: entries, Source: "own-prices.json, vendor.json", Release: "v1"}
}

// write stores a change at effectiveAt with its entries, as the management API writes one.
func (s *StoreSuite) write(name string, effectiveAt time.Time, entries ...PriceEntry) {
	change := PriceChange{Name: name, Kind: PriceChangeAPI, Reason: "a change", Source: "a test",
		EffectiveAt: effectiveAt, CreatedBy: "operator"}
	_, err := s.store.DB().NewInsert().Model(&change).Exec(s.ctx)
	s.Require().NoError(err)
	for i := range entries {
		entries[i].Change = name
	}
	if len(entries) > 0 {
		_, err = s.store.DB().NewInsert().Model(&entries).Exec(s.ctx)
		s.Require().NoError(err)
	}
}

// entryRow is the row a change would write for the entry, priced at its vendor rates under
// its BaseSKU when it has neither.
func entryRow(entry pricing.Entry) PriceEntry {
	if entry.Prices == nil && entry.PriceTiers == nil {
		entry = pricing.AtCost(entry)
	}
	if entry.SKU == "" {
		entry.SKU = pricing.BaseSKU(entry.Modality, entry.Provider, entry.Model, entry.When)
	}
	return priceEntryOf("", entry)
}

// activeList is the price list in force now, built as a router would build it.
func (s *StoreSuite) activeList() (*pricing.List, []PriceEntry) {
	active, err := s.store.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	entries := make([]pricing.Entry, 0, len(active))
	for _, row := range active {
		entries = append(entries, row.Entry())
	}
	list, err := pricing.NewList("database", entries, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	return list, active
}

func (s *StoreSuite) inputRate(list *pricing.List, model string, attributes map[string]string) float64 {
	entry, ok := list.Entry(pricing.Item{Modality: "llm", Provider: "openai", Model: model, Attributes: attributes})
	s.Require().True(ok, "%s has no entry", model)
	return entry.Rates[pricing.InputTokens].USD
}

func (s *StoreSuite) inputPrice(list *pricing.List, model string, attributes map[string]string) float64 {
	entry, ok := list.Entry(pricing.Item{Modality: "llm", Provider: "openai", Model: model, Attributes: attributes})
	s.Require().True(ok, "%s has no entry", model)
	return entry.Prices[pricing.InputTokens].USD
}

// names is each entry's prices.Name and SKU.
func names(entries []pricing.Entry) []string {
	named := make([]string, 0, len(entries))
	for _, entry := range entries {
		named = append(named, entry.Modality+"/"+entry.Provider+"/"+entry.Model+" "+entry.SKU)
	}
	return named
}

func (s *StoreSuite) TestTheFirstSyncSeedsEverything() {
	free := pricing.Entry{Modality: "phone", Provider: "telnyx", Model: "call", Free: true}
	files := load(priced("a", 1), priced("b", 2), free)

	change, added, wrote, err := s.store.SyncPrices(s.ctx, files)

	s.Require().NoError(err)
	s.True(wrote)
	s.Equal(PriceChangeSeed, change.Kind)
	s.Regexp(`^\d{14}_seed$`, change.Name)
	s.Equal("The first price list", change.Reason)
	s.Equal("own-prices.json, vendor.json, router v1", change.Source)
	s.Equal("router v1", change.CreatedBy)
	s.Equal([]string{"llm/openai/a llm-openai-a", "llm/openai/b llm-openai-b", "phone/telnyx/call phone-telnyx-call"}, names(added))
	list, active := s.activeList()
	s.Len(active, 3)
	s.InDelta(2.0, s.inputRate(list, "b", nil), 1e-12, "the vendor's rate is kept")
	s.InDelta(2.4, s.inputPrice(list, "b", nil), 1e-12, "priced at the rate times the factor")
	s.False(list.Billable(pricing.Item{Modality: "phone", Provider: "telnyx", Model: "call"}))
	quote := list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: "a"}, pricing.Quantities{pricing.InputTokens: 1_000_000})
	s.Equal(change.Name, quote.Version, "every entry names the change that wrote it")
	s.EqualValues(1_200_000, quote.CostMicros)
	s.EqualValues(1_000_000, quote.BaseCostMicros)
	s.Equal("llm-openai-a", quote.SKU)
	var markups int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM price_entries WHERE markup_bps IS NOT NULL").Scan(&markups))
	s.Zero(markups, "no markup is written")
}

// TestASeedKeepsAFilesOwnPriceAndSKU: an entry the file prices is written at that price, under
// the SKU the file gives it.
func (s *StoreSuite) TestASeedKeepsAFilesOwnPriceAndSKU() {
	own := priced("a", 1)
	own.SKU = "house-a"
	own.Prices = map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1.1)}

	_, _, _, err := s.store.SyncPrices(s.ctx, load(own))
	s.Require().NoError(err)

	list, active := s.activeList()
	s.Require().Len(active, 1)
	s.Equal("house-a", active[0].SKU)
	s.InDelta(1.1, s.inputPrice(list, "a", nil), 1e-12)
}

func (s *StoreSuite) TestALaterSyncOnlyAddsNewItems() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)

	next := load(priced("a", 9), priced("b", 2))
	next.Release = "v2"
	change, added, wrote, err := s.store.SyncPrices(s.ctx, next)

	s.Require().NoError(err)
	s.True(wrote)
	s.Equal(PriceChangeRelease, change.Kind)
	s.Regexp(`^\d{14}_release_v2$`, change.Name)
	s.Equal("Items new in v2", change.Reason)
	s.Equal([]string{"llm/openai/b llm-openai-b"}, names(added))
	list, _ := s.activeList()
	s.InDelta(1.0, s.inputRate(list, "a", nil), 1e-12, "an item the database has keeps its rate")
	s.InDelta(1.2, s.inputPrice(list, "a", nil), 1e-12, "and its price")
	s.InDelta(2.0, s.inputRate(list, "b", nil), 1e-12)
	s.InDelta(2.4, s.inputPrice(list, "b", nil), 1e-12, "an item new in a release is priced at the factor")
}

func (s *StoreSuite) TestASyncWithNothingNewWritesNothing() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)

	_, added, wrote, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))

	s.Require().NoError(err)
	s.False(wrote)
	s.Empty(added)
	changes, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{})
	s.Require().NoError(err)
	s.Len(changes, 1)
}

func (s *StoreSuite) TestAServiceTierNewInAReleaseIsAddedBesideTheDefault() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)
	flex := priced("a", 0.5)
	flex.When = map[string]string{"service_tier": "flex"}

	_, added, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1), flex))

	s.Require().NoError(err)
	s.Require().Len(added, 1)
	s.Equal(flex.When, added[0].When)
	s.Equal("llm-openai-a-service-tier-flex", added[0].SKU)
}

func (s *StoreSuite) TestASyncNeverTouchesAWithdrawnItem() {
	seed, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)
	withdrawn := PriceEntry{Modality: "llm", Provider: "openai", Model: "a", SKU: "llm-openai-a", Withdrawn: true}
	s.write("20990101000000_withdraw_a", seed.EffectiveAt, withdrawn)

	_, added, wrote, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))

	s.Require().NoError(err)
	s.False(wrote)
	s.Empty(added)
	_, active := s.activeList()
	s.Empty(active)
}

func (s *StoreSuite) TestAnItemOnlyACancelledChangePricedIsAddedAgain() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)
	s.write("20990101000000_add_b", time.Now().Add(time.Hour), entryRow(priced("b", 5)))
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'operator' WHERE name = '20990101000000_add_b'")
	s.Require().NoError(err)

	_, added, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1), priced("b", 2)))

	s.Require().NoError(err)
	s.Equal([]string{"llm/openai/b llm-openai-b"}, names(added), "a cancelled change's SKU holds nothing")
}

func (s *StoreSuite) TestTwoSyncsAtOnceWriteOnce() {
	other, err := Open(s.dsn)
	s.Require().NoError(err)
	defer other.Close()
	files := load(priced("a", 1), priced("b", 2))

	var wg sync.WaitGroup
	wrote := make([]bool, 8)
	failed := make([]error, 8)
	for i := range wrote {
		router := s.store
		if i%2 == 1 {
			router = other
		}
		wg.Go(func() { _, _, wrote[i], failed[i] = router.SyncPrices(s.ctx, files) })
	}
	wg.Wait()

	written := 0
	for i := range wrote {
		s.Require().NoError(failed[i])
		if wrote[i] {
			written++
		}
	}
	s.Equal(1, written)
	changes, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{})
	s.Require().NoError(err)
	s.Len(changes, 1)
	_, active := s.activeList()
	s.Len(active, 2)
}

func (s *StoreSuite) TestASyncIsAudited() {
	files := load(priced("a", 1))

	change, _, _, err := s.store.SyncPrices(s.ctx, files)
	s.Require().NoError(err)

	// audit_log outlives a test, and a change made in the same second by another test has
	// this one's name, so the newest row is this sync's.
	var entry AuditEntry
	s.Require().NoError(s.store.DB().NewSelect().Model(&entry).
		Where("resource_type = ? AND resource_id = ?", AuditResourcePriceList, change.Name).
		Order("created_at DESC").Limit(1).Scan(s.ctx))
	s.Empty(entry.CustomerID)
	s.Equal(AuditCreated, entry.Action)
	s.Equal(AuditSourceCLI, entry.Source)
	s.Equal("router v1", entry.ActorID)
	s.Equal([]AuditChange{{Field: "entries", After: []any{"llm/openai/a"}}}, entry.Changes)
}

func (s *StoreSuite) TestAReleaseNameIsSanitized() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)
	next := load(priced("b", 1))
	next.Release = "v1.4.0-RC/2"

	change, _, _, err := s.store.SyncPrices(s.ctx, next)

	s.Require().NoError(err)
	s.Regexp(`^\d{14}_release_v1_4_0_rc_2$`, change.Name)
	s.Equal("Items new in v1.4.0-RC/2", change.Reason)
}

func (s *StoreSuite) TestPricesToLoadWritesNothing() {
	seed, added, err := s.store.PricesToLoad(s.ctx, []pricing.Entry{priced("a", 1)})

	s.Require().NoError(err)
	s.True(seed)
	s.Len(added, 1)
	changes, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{})
	s.Require().NoError(err)
	s.Empty(changes)
}

func (s *StoreSuite) TestAPriceEntryCannotBeChanged() {
	_, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1)))
	s.Require().NoError(err)

	for _, update := range []string{"UPDATE price_entries SET free = true",
		`UPDATE price_entries SET prices = '{"input_tokens": {"usd": 9, "per": 1000000}}'`,
		"UPDATE price_entries SET price_tiers = '[]'", "UPDATE price_entries SET sku = 'another'"} {
		_, err := s.store.DB().ExecContext(s.ctx, update)
		s.ErrorContains(err, "price_entries rows are never changed or deleted", update)
	}
	_, deleted := s.store.DB().ExecContext(s.ctx, "DELETE FROM price_entries")
	s.ErrorContains(deleted, "price_entries rows are never changed or deleted")
}

// TestAnEntryWithoutASKUOrAPriceIsRefused is a router built before prices were stored
// writing an entry: it names neither, so the table refuses it.
func (s *StoreSuite) TestAnEntryWithoutASKUOrAPriceIsRefused() {
	s.write("20990101000000_old_router", time.Now().Add(time.Hour))

	_, unnamed := s.store.DB().ExecContext(s.ctx, `INSERT INTO price_entries (change, modality, provider, model, free, effective_at)
VALUES ('20990101000000_old_router', 'llm', 'openai', 'a', true, now())`)
	_, unpriced := s.store.DB().ExecContext(s.ctx, `INSERT INTO price_entries (change, sku, modality, provider, model, rates, effective_at)
VALUES ('20990101000000_old_router', 'llm-openai-a', 'llm', 'openai', 'a', '{"input_tokens": {"usd": 1, "per": 1000000}}', now())`)

	s.ErrorContains(unnamed, "price_entries_sku")
	s.ErrorContains(unpriced, "price_entries_priced")
}

func (s *StoreSuite) TestAnEntryTakesItsChangesTime() {
	tomorrow := time.Now().Add(24 * time.Hour).UTC().Truncate(time.Microsecond)
	s.write("20990101000000_tomorrow", tomorrow)

	_, err := s.store.DB().ExecContext(s.ctx, `INSERT INTO price_entries (change, sku, modality, provider, model, effective_at)
VALUES ('20990101000000_tomorrow', 'llm-openai-a', 'llm', 'openai', 'a', now() - interval '1 year')`)
	s.Require().NoError(err)

	var effectiveAt time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT effective_at FROM price_entries").Scan(&effectiveAt))
	s.True(tomorrow.Equal(effectiveAt), "%s is not %s", effectiveAt, tomorrow)
}

func (s *StoreSuite) TestAChangeCanBeCancelledOnlyBeforeItTakesEffect() {
	s.write("20990101000000_later", time.Now().Add(time.Hour))
	s.write("20990101000000_now", time.Now().Add(-time.Second))

	_, later := s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'operator' WHERE name = '20990101000000_later'")
	_, now := s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'operator' WHERE name = '20990101000000_now'")

	s.NoError(later)
	s.ErrorContains(now, "a price change can only be cancelled, once, before it takes effect")
}

func (s *StoreSuite) TestAChangeIsCancelledOnce() {
	s.write("20990101000000_later", time.Now().Add(time.Hour))
	cancel := "UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'operator' WHERE name = '20990101000000_later'"
	_, err := s.store.DB().ExecContext(s.ctx, cancel)
	s.Require().NoError(err)

	_, err = s.store.DB().ExecContext(s.ctx, cancel)

	s.ErrorContains(err, "a price change can only be cancelled, once")
}

func (s *StoreSuite) TestNothingElseAboutAChangeCanBeUpdated() {
	s.write("20990101000000_later", time.Now().Add(time.Hour))

	_, reason := s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET reason = 'rewritten' WHERE name = '20990101000000_later'")
	_, both := s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now(), markup_bps = 1000 WHERE name = '20990101000000_later'")

	s.ErrorContains(reason, "a price change can only be cancelled")
	s.ErrorContains(both, "a price change can only be cancelled")
}

func (s *StoreSuite) TestAPriceChangeCannotBeDeleted() {
	s.write("20990101000000_later", time.Now().Add(time.Hour))

	_, err := s.store.DB().ExecContext(s.ctx, "DELETE FROM price_changes")

	s.ErrorContains(err, "price_changes rows are never deleted")
}

func (s *StoreSuite) TestTheNewestEntryForAnItemWins() {
	s.write("20260101000000_first", time.Now().Add(-2*time.Hour), entryRow(priced("a", 1)))
	s.write("20260101000001_second", time.Now().Add(-time.Hour), entryRow(priced("a", 2)))

	list, active := s.activeList()

	s.Len(active, 1)
	s.InDelta(2.0, s.inputRate(list, "a", nil), 1e-12)
}

func (s *StoreSuite) TestTwoChangesAtOneTimeGoToTheLaterName() {
	at := time.Now().Add(-time.Hour)
	s.write("20260101000000_b", at, entryRow(priced("a", 2)))
	s.write("20260101000000_a", at, entryRow(priced("a", 1)))

	list, _ := s.activeList()

	s.InDelta(2.0, s.inputRate(list, "a", nil), 1e-12)
}

func (s *StoreSuite) TestAWithdrawnModelHasNoEntry() {
	s.write("20260101000000_first", time.Now().Add(-2*time.Hour), entryRow(priced("a", 1)), entryRow(priced("b", 1)))
	withdrawn := entryRow(priced("a", 1))
	withdrawn.Withdrawn = true
	s.write("20260101000001_withdraw", time.Now().Add(-time.Hour), withdrawn)

	_, active := s.activeList()

	s.Require().Len(active, 1)
	s.Equal("b", active[0].Model)
}

func (s *StoreSuite) TestAChangeForTomorrowIsNotInForceToday() {
	s.write("20260101000000_today", time.Now().Add(-time.Hour), entryRow(priced("a", 1)))
	s.write("20990101000000_tomorrow", time.Now().Add(24*time.Hour), entryRow(priced("a", 2)))

	list, _ := s.activeList()
	later, err := s.store.ActivePriceEntries(s.ctx, time.Now().Add(25*time.Hour))
	s.Require().NoError(err)

	s.InDelta(1.0, s.inputRate(list, "a", nil), 1e-12)
	s.Require().Len(later, 1)
	s.Equal("20990101000000_tomorrow", later[0].Change)
}

func (s *StoreSuite) TestACancelledChangeIsNeverInForce() {
	s.write("20260101000000_today", time.Now().Add(-time.Hour), entryRow(priced("a", 1)))
	s.write("20990101000000_later", time.Now().Add(time.Hour), entryRow(priced("a", 2)))
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'operator' WHERE name = '20990101000000_later'")
	s.Require().NoError(err)

	entries, err := s.store.ActivePriceEntries(s.ctx, time.Now().Add(2*time.Hour))
	s.Require().NoError(err)
	upcoming, err := s.store.UpcomingPriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)

	s.Require().Len(entries, 1)
	s.Equal("20260101000000_today", entries[0].Change)
	s.Empty(upcoming)
}

func (s *StoreSuite) TestTheEntryForACountryIsKeptBesideTheDefault() {
	sms := func(usd float64, when map[string]string) PriceEntry {
		return entryRow(pricing.Entry{Modality: "phone", Provider: "telnyx", Model: "sms", When: when,
			Rates: map[pricing.Meter]pricing.Rate{pricing.MessageSegments: {USD: usd, Per: 1}}})
	}
	s.write("20260101000000_first", time.Now().Add(-2*time.Hour), sms(0.004, nil), sms(0.02, map[string]string{"country": "CA"}))
	s.write("20260101000001_default", time.Now().Add(-time.Hour), sms(0.005, nil))

	list, active := s.activeList()

	s.Len(active, 2)
	item := pricing.Item{Modality: "phone", Provider: "telnyx", Model: "sms"}
	s.EqualValues(5_000, list.Quote(item, pricing.Quantities{pricing.MessageSegments: 1}).CostMicros)
	item.Attributes = map[string]string{"country": "CA"}
	s.EqualValues(20_000, list.Quote(item, pricing.Quantities{pricing.MessageSegments: 1}).CostMicros)
}

func (s *StoreSuite) TestAServiceTierEntryIsKeptBesideTheDefault() {
	flex := priced("a", 0.5)
	flex.When = map[string]string{"service_tier": "flex"}
	s.write("20260101000000_first", time.Now().Add(-2*time.Hour), entryRow(priced("a", 1)), entryRow(flex))
	withdrawn := entryRow(priced("a", 1))
	withdrawn.Withdrawn = true
	s.write("20260101000001_withdraw_default", time.Now().Add(-time.Hour), withdrawn)

	list, active := s.activeList()

	s.Require().Len(active, 1, "withdrawing the default leaves the flex entry pricing")
	s.InDelta(0.5, s.inputRate(list, "a", map[string]string{"service_tier": "flex"}), 1e-12)
}

func (s *StoreSuite) TestTiersRoundTrip() {
	entry := pricing.Entry{Modality: "llm", Provider: "openai", Model: "long",
		Rates: map[pricing.Meter]pricing.Rate{
			pricing.InputTokens:  perMillionTokens(2),
			pricing.OutputTokens: perMillionTokens(8),
			pricing.CallSeconds:  {USD: 0.007, Per: 60, Increment: 60, Minimum: 60},
		},
		Tiers: []pricing.Tier{{AboveInputTokens: 272_000, Rates: map[pricing.Meter]pricing.Rate{
			pricing.InputTokens: perMillionTokens(4), pricing.OutputTokens: perMillionTokens(12),
			pricing.CallSeconds: {USD: 0.007, Per: 60, Increment: 60, Minimum: 60},
		}}},
	}
	_, _, _, err := s.store.SyncPrices(s.ctx, load(entry))
	s.Require().NoError(err)

	list, active := s.activeList()

	s.Require().Len(active, 1)
	read := active[0].Entry()
	s.Equal(active[0].Change, read.Version)
	read.Version = ""
	written := pricing.Priced(entry)
	written.SKU = "llm-openai-long"
	s.Equal(written, read)
	s.Equal(0.0084, read.Prices[pricing.CallSeconds].USD, "0.007 × 1.2, kept to twelve digits")
	long := list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: "long"}, pricing.Quantities{pricing.InputTokens: 300_000})
	s.EqualValues(1_440_000, long.CostMicros, "a long prompt is charged at its price tier")
	s.EqualValues(1_200_000, long.BaseCostMicros, "and costed at its tier of rates")
}

func (s *StoreSuite) TestTheNextChangeIsKnown() {
	soon := time.Now().Add(time.Hour).UTC().Truncate(time.Microsecond)
	s.write("20260101000000_past", time.Now().Add(-time.Hour))
	s.write("20990101000000_soon", soon)
	s.write("20990101000001_later", soon.Add(time.Hour))

	next, ok, err := s.store.NextPriceChangeAt(s.ctx, time.Now())
	s.Require().NoError(err)
	_, none, err := s.store.NextPriceChangeAt(s.ctx, soon.Add(2*time.Hour))
	s.Require().NoError(err)

	s.True(ok)
	s.True(soon.Equal(next), "%s is not %s", next, soon)
	s.False(none)
}

func (s *StoreSuite) TestUpcomingEntriesCarryTheirChangesReason() {
	s.write("20260101000000_past", time.Now().Add(-time.Hour), entryRow(priced("a", 1)))
	s.write("20990101000000_soon", time.Now().Add(time.Hour), entryRow(priced("a", 2)))

	upcoming, err := s.store.UpcomingPriceEntries(s.ctx, time.Now())

	s.Require().NoError(err)
	s.Require().Len(upcoming, 1)
	s.Equal("20990101000000_soon", upcoming[0].Change)
	s.Equal("a change", upcoming[0].Reason)
}

func (s *StoreSuite) TestAPriceChangeCanBeReadBack() {
	change, _, _, err := s.store.SyncPrices(s.ctx, load(priced("a", 1), priced("b", 2)))
	s.Require().NoError(err)

	read, entries, err := s.store.PriceChange(s.ctx, change.Name)
	_, _, missing := s.store.PriceChange(s.ctx, "20990101000000_nobody")

	s.Require().NoError(err)
	s.Equal(change.Name, read.Name)
	s.Len(entries, 2)
	s.ErrorIs(missing, ErrNoPriceChange)
}

func (s *StoreSuite) TestPriceChangesPageNewestFirst() {
	s.write("20260101000000_a", time.Now().Add(-time.Hour))
	s.write("20260101000001_b", time.Now().Add(-time.Hour))
	s.write("20260101000002_c", time.Now().Add(-time.Hour))

	first, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{Limit: 2})
	s.Require().NoError(err)
	s.Require().Len(first, 3, "one more than the limit says there is another page")
	last := first[1]
	second, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{Limit: 2, After: &PriceChangePosition{CreatedAt: last.CreatedAt, Name: last.Name}})
	s.Require().NoError(err)
	seeds, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{Kind: PriceChangeSeed})
	s.Require().NoError(err)

	s.Equal([]string{"20260101000002_c", "20260101000001_b"}, []string{first[0].Name, first[1].Name})
	s.Require().Len(second, 1)
	s.Equal("20260101000000_a", second[0].Name)
	s.Empty(seeds)
}
