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

// apiChange is a change through the management API to the entries, under key.
func apiChange(name, key string, entries ...pricing.Entry) PriceChangeWrite {
	writes := make([]PriceEntryWrite, 0, len(entries))
	for _, entry := range entries {
		writes = append(writes, PriceEntryWrite{Entry: entry})
	}
	return PriceChangeWrite{Name: name, Reason: "The vendor halved its rate.", Source: "https://vendor.example/pricing, 2026-10-23",
		Entries: writes, IdempotencyKey: key, ActorID: "staff:ada", ActorName: "Ada"}
}

// accept is a check that takes every change.
func accept(PriceListAt) error { return nil }

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

	again, rows, replayed, err := s.store.CreatePriceChange(s.ctx, write, func(PriceListAt) error {
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

	_, _, _, err := s.store.CreatePriceChange(s.ctx, apiChange("checked", "key-check", priced("a", 0.5)), func(before PriceListAt) error {
		s.Len(before.Entries, 1, "the check is handed the list the change lands on")
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
	write.MarkupBps = markup(2_500)
	change, _, _, err := s.store.CreatePriceChange(s.ctx, write, accept)
	s.Require().NoError(err)
	source := s.store.PriceSource()

	entries, markupBps, next, err := source.Active(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Equal(seed.Name, entries[0].Version)
	s.Zero(markupBps)
	s.True(next.Equal(tomorrow))
	upcoming, err := source.Upcoming(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Require().Len(upcoming, 1)
	s.Equal(change.Name, upcoming[0].Version)
	s.Equal(write.Reason, upcoming[0].Reason)

	entries, markupBps, next, err = source.Active(s.ctx, tomorrow)
	s.Require().NoError(err)
	s.Equal(change.Name, entries[0].Version)
	s.EqualValues(2_500, markupBps)
	s.True(next.IsZero())
}
