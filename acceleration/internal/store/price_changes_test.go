//go:build integration

package store

import (
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// seeded loads a first price list, as a router's first start does.
func (s *StoreSuite) seeded(entries ...pricing.Entry) PriceChange {
	change, _, _, err := s.store.SyncPrices(s.ctx, load(entries...))
	s.Require().NoError(err)
	return change
}

// apiChange is a change through the management API to the entries, under key, each without a
// price of its own priced at its vendor rates.
func apiChange(name, key string, entries ...pricing.Entry) PriceChangeWrite {
	writes := make([]PriceEntryWrite, 0, len(entries))
	for _, entry := range entries {
		if entry.Prices == nil {
			entry = pricing.AtCost(entry)
		}
		writes = append(writes, PriceEntryWrite{Entry: entry})
	}
	return PriceChangeWrite{Name: name, Reason: "The vendor halved its rate.", Source: "https://vendor.example/pricing, 2026-10-23",
		Entries: writes, IdempotencyKey: key, ActorID: "staff:ada", ActorName: "Ada"}
}

// withdrawal is a change's write that withdraws model.
func withdrawal(model string) PriceEntryWrite {
	return PriceEntryWrite{Entry: pricing.Entry{Modality: "llm", Provider: "openai", Model: model}, Withdrawn: true}
}

// accept is a check that takes every change.
func accept(PriceListAt, []PriceEntryWrite) error { return nil }

// auditOf is the audit rows of a change, oldest first.
func (s *StoreSuite) auditOf(name string) []AuditEntry {
	var rows []AuditEntry
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("resource_id = ?", name).Order("created_at").Scan(s.ctx))
	return rows
}

func (s *StoreSuite) TestAPriceChangeIsRefusedBeforeTheSeed() {
	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("early", "key-early", priced("a", 1)), accept)

	s.ErrorIs(err, ErrPriceListNotSeeded)
	changes, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{})
	s.Require().NoError(err)
	s.Empty(changes)
}

func (s *StoreSuite) TestAPriceChangeWritesItsEntriesAndOneAuditRow() {
	s.seeded(priced("a", 1))

	change, rows, replayed, err := s.store.CreatePriceChange(s.ctx, apiChange("halve_a", "key-halve", priced("a", 0.5)), accept)

	s.Require().NoError(err)
	s.False(replayed)
	s.Regexp(`^\d{14}_halve_a$`, change.Name)
	s.Equal(PriceChangeAPI, change.Kind)
	s.Equal("staff:ada", change.CreatedBy)
	s.Equal("key-halve", change.IdempotencyKey)
	s.Require().Len(rows, 1)
	s.Equal(change.EffectiveAt, rows[0].EffectiveAt, "the entry takes effect with its change")
	list, _ := s.activeList()
	s.InDelta(0.5, s.inputRate(list, "a", nil), 1e-12)
	audit := s.auditOf(change.Name)
	s.Require().Len(audit, 1)
	s.Equal(AuditCreated, audit[0].Action)
	s.Equal(AuditSourceManage, audit[0].Source)
	s.Equal(AuditResourcePriceList, audit[0].ResourceType)
	s.Equal("staff:ada", audit[0].ActorID)
	s.Equal("Ada", audit[0].ActorName)
}

func (s *StoreSuite) TestAPriceChangeResubmittedUnderItsKeyIsWrittenOnce() {
	s.seeded(priced("a", 1))
	write := apiChange("resubmitted", "key-once", priced("a", 0.5))
	first, _, _, err := s.store.CreatePriceChange(s.ctx, write, accept)
	s.Require().NoError(err)

	again, rows, replayed, err := s.store.CreatePriceChange(s.ctx, write, func(PriceListAt, []PriceEntryWrite) error {
		return errors.New("a resubmit is not checked again")
	})

	s.Require().NoError(err)
	s.True(replayed)
	s.Equal(first.Name, again.Name)
	s.Len(rows, 1)
	s.Len(s.auditOf(first.Name), 1)
}

func (s *StoreSuite) TestAPriceChangeKeyHeldByAnotherBodyIsRefused() {
	s.seeded(priced("a", 1))
	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("reused", "key-reused", priced("a", 0.5)), accept)
	s.Require().NoError(err)

	_, _, _, err = s.store.CreatePriceChange(s.ctx, apiChange("reused", "key-reused", priced("a", 0.4)), accept)

	s.ErrorIs(err, ErrPriceChangeKeyReused)
}

func (s *StoreSuite) TestAPriceChangeTheCheckRefusesWritesNothing() {
	s.seeded(priced("a", 1))
	refused := errors.New("refused")

	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("checked", "key-check", priced("a", 0.5)), func(before PriceListAt, writes []PriceEntryWrite) error {
		s.Len(before.Entries, 1, "the check is handed the list the change lands on")
		s.Equal("llm-openai-a", writes[0].SKU, "and the entries with their SKUs")
		return refused
	})

	s.ErrorIs(err, refused)
	changes, err := s.store.PriceChanges(s.ctx, PriceChangeFilter{})
	s.Require().NoError(err)
	s.Len(changes, 1, "only the seed")
}

func (s *StoreSuite) TestAPriceChangeTakesEffectNoEarlierThanNow() {
	s.seeded(priced("a", 1))
	tenMinutesAgo, twoMinutesAgo := time.Now().Add(-10*time.Minute), time.Now().Add(-2*time.Minute)
	past := apiChange("past", "key-past", priced("a", 0.5))
	past.EffectiveAt = &tenMinutesAgo
	recent := apiChange("recent", "key-recent", priced("a", 0.5))
	recent.EffectiveAt = &twoMinutesAgo

	_, _, _, err := s.store.CreatePriceChange(s.ctx, past, accept)
	s.ErrorIs(err, ErrPriceChangeInPast)
	change, _, _, err := s.store.CreatePriceChange(s.ctx, recent, accept)
	s.Require().NoError(err)

	s.Equal(change.CreatedAt, change.EffectiveAt, "two minutes ago is taken as now")
	_, _, replayed, err := s.store.CreatePriceChange(s.ctx, recent, accept)
	s.Require().NoError(err)
	s.True(replayed, "the time it asked for still matches the change it took")
}

func (s *StoreSuite) TestAScheduledPriceChangeIsCancelledOnce() {
	s.seeded(priced("a", 1))
	tomorrow := time.Now().Add(24 * time.Hour)
	write := apiChange("tomorrow", "key-tomorrow", priced("a", 0.5))
	write.EffectiveAt = &tomorrow
	change, _, _, err := s.store.CreatePriceChange(s.ctx, write, accept)
	s.Require().NoError(err)

	cancelled, rows, done, err := s.store.CancelPriceChange(s.ctx, change.Name, "The vendor withdrew it.", "staff:grace", "Grace")
	s.Require().NoError(err)
	s.True(done)
	s.Require().NotNil(cancelled.CancelledAt)
	s.Equal("staff:grace", cancelled.CancelledBy)
	s.Len(rows, 1)

	again, _, done, err := s.store.CancelPriceChange(s.ctx, change.Name, "Again.", "staff:alan", "Alan")
	s.Require().NoError(err)
	s.False(done, "a cancel sent again writes nothing")
	s.Equal("staff:grace", again.CancelledBy)
	s.Equal("The vendor withdrew it.", again.CancelReason)
	audit := s.auditOf(change.Name)
	s.Require().Len(audit, 2)
	s.Equal(AuditCancelled, audit[1].Action)
	upcoming, err := s.store.UpcomingPriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Empty(upcoming)
}

func (s *StoreSuite) TestAPriceChangeInForceCannotBeCancelled() {
	seed := s.seeded(priced("a", 1))
	change, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("now", "key-now", priced("a", 0.5)), accept)
	s.Require().NoError(err)

	for _, name := range []string{seed.Name, change.Name} {
		_, _, _, err := s.store.CancelPriceChange(s.ctx, name, "Too late.", "staff:ada", "Ada")
		s.ErrorIs(err, ErrPriceChangeInForce, name)
	}
	_, _, _, err = s.store.CancelPriceChange(s.ctx, "20261023000000_nobody", "No such change.", "staff:ada", "Ada")
	s.ErrorIs(err, ErrNoPriceChange)
}

func (s *StoreSuite) TestAWriteThatWaitsOutThePriceListsLockIsBusy() {
	s.seeded(priced("a", 1))
	tomorrow := time.Now().Add(24 * time.Hour)
	scheduled := apiChange("busy_cancel", "key-busy-cancel", priced("a", 0.5))
	scheduled.EffectiveAt = &tomorrow
	change, _, _, err := s.store.CreatePriceChange(s.ctx, scheduled, accept)
	s.Require().NoError(err)
	held, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // the lock goes with the transaction
	_, err = held.ExecContext(s.ctx, "SELECT pg_advisory_xact_lock(hashtextextended('router.prices', 0))")
	s.Require().NoError(err)

	_, _, _, err = s.store.CreatePriceChange(s.ctx, apiChange("busy", "key-busy", priced("a", 0.5)), accept)
	s.ErrorIs(err, ErrPriceListBusy)
	_, _, _, err = s.store.CancelPriceChange(s.ctx, change.Name, "Busy.", "staff:ada", "Ada")
	s.ErrorIs(err, ErrPriceListBusy)
}

func (s *StoreSuite) TestThePriceSourceAnswersTheListInForceAndTheNextChange() {
	seed := s.seeded(priced("a", 1))
	tomorrow := time.Now().Add(24 * time.Hour).UTC().Truncate(time.Microsecond)
	write := apiChange("tomorrow", "key-source", priced("a", 0.5))
	write.EffectiveAt = &tomorrow
	change, _, _, err := s.store.CreatePriceChange(s.ctx, write, accept)
	s.Require().NoError(err)
	source := s.store.PriceSource()

	entries, next, err := source.Active(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Equal(seed.Name, entries[0].Version)
	s.True(next.Equal(tomorrow))
	upcoming, err := source.Upcoming(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Require().Len(upcoming, 1)
	s.Equal(change.Name, upcoming[0].Version)
	s.Equal(write.Reason, upcoming[0].Reason)
	s.Equal(entries[0].SKU, upcoming[0].SKU, "the item keeps its SKU in the change scheduled for it")

	entries, next, err = source.Active(s.ctx, tomorrow)
	s.Require().NoError(err)
	s.Equal(change.Name, entries[0].Version)
	s.InDelta(0.5, entries[0].Prices[pricing.InputTokens].USD, 1e-12)
	s.True(next.IsZero())
}

// TestAnItemKeepsItsSKUAcrossItsPriceChanges: the seed names a, and every change of a's price
// after it, now or scheduled, names it the same.
func (s *StoreSuite) TestAnItemKeepsItsSKUAcrossItsPriceChanges() {
	s.seeded(priced("a", 1))
	tomorrow := time.Now().Add(24 * time.Hour)
	later := apiChange("later", "key-later", priced("a", 0.4))
	later.EffectiveAt = &tomorrow

	_, now, _, err := s.store.CreatePriceChange(s.ctx, apiChange("now", "key-sku-now", priced("a", 0.5)), accept)
	s.Require().NoError(err)
	_, scheduled, _, err := s.store.CreatePriceChange(s.ctx, later, accept)
	s.Require().NoError(err)

	s.Equal("llm-openai-a", now[0].SKU)
	s.Equal("llm-openai-a", scheduled[0].SKU)
	list, _ := s.activeList()
	s.Equal("llm-openai-a", list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: "a"}, pricing.Quantities{pricing.InputTokens: 1}).SKU)
}

// TestARenamedModelKeepsItsSKU: the vendor renamed a to a-2026. One change withdraws a and
// prices a-2026 under a's SKU, which every row of the new name then records.
func (s *StoreSuite) TestARenamedModelKeepsItsSKU() {
	s.seeded(priced("a", 1))
	renamed := priced("a-2026", 1)
	renamed.SKU = "llm-openai-a"
	write := apiChange("rename_a", "key-rename", renamed)
	write.Entries = append(write.Entries, withdrawal("a"))

	_, rows, _, err := s.store.CreatePriceChange(s.ctx, write, accept)

	s.Require().NoError(err)
	s.Require().Len(rows, 2)
	s.Equal("llm-openai-a", rows[0].SKU)
	s.Equal("llm-openai-a", rows[1].SKU, "the withdrawal names the SKU it stops pricing")
	s.True(rows[1].Withdrawn)
	list, active := s.activeList()
	s.Require().Len(active, 1)
	s.Equal("a-2026", active[0].Model)
	s.Equal("llm-openai-a", list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: "a-2026"}, pricing.Quantities{pricing.InputTokens: 1}).SKU)

	again := priced("a", 1)
	_, back, _, err := s.store.CreatePriceChange(s.ctx, apiChange("a_again", "key-again", again), accept)
	s.Require().NoError(err)
	s.Equal("llm-openai-a-2", back[0].SKU, "the old name, priced again, cannot take the SKU its successor has")
}

// TestASKUAnotherItemHasInForceIsRefused: a new item may only take a SKU whose item the same
// change withdraws, or that no item has in force, and none that a later change gives away.
func (s *StoreSuite) TestASKUAnotherItemHasInForceIsRefused() {
	s.seeded(priced("a", 1), priced("b", 1))
	taken := priced("c", 1)
	taken.SKU = "llm-openai-a"

	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("take_a", "key-take", taken), accept)
	s.ErrorIs(err, ErrSKUInUse)
	s.ErrorContains(err, "llm-openai-a is the SKU of llm/openai/a")

	tomorrow := time.Now().Add(24 * time.Hour)
	scheduled := apiChange("rename_b", "key-rename-b", pricing.Entry{Modality: "llm", Provider: "openai", Model: "b-2026",
		SKU: "llm-openai-b", Rates: priced("b", 1).Rates})
	scheduled.Entries = append(scheduled.Entries, withdrawal("b"))
	scheduled.EffectiveAt = &tomorrow
	_, _, _, err = s.store.CreatePriceChange(s.ctx, scheduled, accept)
	s.Require().NoError(err)
	early := priced("d", 1)
	early.SKU = "llm-openai-b"
	early.Prices = early.Rates
	write := apiChange("early_b", "key-early-b", early)
	write.Entries = append(write.Entries, withdrawal("b"))
	_, _, _, err = s.store.CreatePriceChange(s.ctx, write, accept)
	s.ErrorIs(err, ErrSKUInUse, "a change scheduled later gives llm-openai-b to b-2026")
	s.ErrorContains(err, "a change scheduled later gives llm-openai-b to llm/openai/b-2026")
}

func (s *StoreSuite) TestAnItemInForceKeepsItsSKU() {
	s.seeded(priced("a", 1))
	renamed := priced("a", 0.5)
	renamed.SKU = "another"

	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("resku", "key-resku", renamed), accept)

	s.ErrorIs(err, ErrSKUChanged)
	s.ErrorContains(err, "llm/openai/a has the SKU llm-openai-a, not another")
}

// TestTwoItemsOfOneBaseSKUAreNumbered: gpt-4.1 and gpt-4-1 make the same SKU, so the item
// priced second takes the next number.
func (s *StoreSuite) TestTwoItemsOfOneBaseSKUAreNumbered() {
	s.seeded(priced("gpt-4.1", 1))

	_, rows, _, err := s.store.CreatePriceChange(s.ctx, apiChange("dashed", "key-dashed", priced("gpt-4-1", 1)), accept)

	s.Require().NoError(err)
	s.Equal("llm-openai-gpt-4-1-2", rows[0].SKU)
}
