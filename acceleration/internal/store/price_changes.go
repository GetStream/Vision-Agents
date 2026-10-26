package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// What a price change made through the management API is refused for.
var (
	// ErrPriceListNotSeeded is a change asked for before the router's first start loaded the
	// price files: a change written first would stop that load from seeding the list.
	ErrPriceListNotSeeded = errors.New("store: the price list is not seeded yet")
	// ErrPriceChangeKeyReused is an idempotency key that already holds another change.
	ErrPriceChangeKeyReused = errors.New("store: the idempotency key holds another price change")
	// ErrPriceChangeExists is a change named as one already made in the same second.
	ErrPriceChangeExists = errors.New("store: a price change of that name was made this second")
	// ErrPriceChangeInForce is a cancel of a change already in force, or one the API did not
	// make: neither can be cancelled, only followed by a newer change.
	ErrPriceChangeInForce = errors.New("store: the price change is in force or was not made through the API")
	// ErrPriceChangeInPast is an effective time more than PriceChangeGrace ago.
	ErrPriceChangeInPast = errors.New("store: a price change cannot take effect in the past")
	// ErrPriceListBusy is a write that waited its lock timeout for another write to the price
	// list, or a router's start loading its price files, to finish: one to send again shortly.
	ErrPriceListBusy = errors.New("store: another write to the price list held it past the lock timeout")
)

// lockNotAvailable is the SQLSTATE a statement that waited past lock_timeout fails with.
const lockNotAvailable = "55P03"

// busyOr is ErrPriceListBusy for a write that gave up waiting for the price list's lock, and
// err for any other failure.
func busyOr(err error) error {
	if isViolation(err, lockNotAvailable) {
		return ErrPriceListBusy
	}
	return err
}

// PriceChangeGrace is how far in the past a change may say it takes effect: one up to this
// long ago is taken as now, which absorbs a caller's clock running behind the database's, and
// one further back is refused, so no change rewrites what a row already billed.
const PriceChangeGrace = 5 * time.Minute

// PriceSource is the price list in force as pricing.Active reads it.
func (s *Store) PriceSource() pricing.Source {
	return priceSource{store: s}
}

type priceSource struct {
	store *Store
}

// Active reads the entries, the markup and the next change in one snapshot, so a change
// committed between the reads cannot leave the list with half of it.
func (p priceSource) Active(ctx context.Context, at time.Time) ([]pricing.Entry, int64, time.Time, error) {
	var entries []pricing.Entry
	var markup int64
	var next time.Time
	err := p.store.db.RunInTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true}, func(ctx context.Context, tx bun.Tx) error {
		rows, err := activePriceEntries(ctx, tx, at)
		if err != nil {
			return err
		}
		entries = make([]pricing.Entry, 0, len(rows))
		for _, row := range rows {
			entry := row.Entry()
			entry.EffectiveAt = row.EffectiveAt.UTC()
			entries = append(entries, entry)
		}
		if markup, err = activeMarkup(ctx, tx, at); err != nil {
			return err
		}
		found, ok, err := nextPriceChangeAt(ctx, tx, at)
		if ok {
			next = found
		}
		return err
	})
	if err != nil {
		return nil, 0, time.Time{}, stack.Wrap(fmt.Errorf("store: the price list in force: %w", err))
	}
	return entries, markup, next, nil
}

func (p priceSource) Upcoming(ctx context.Context, after time.Time) ([]pricing.UpcomingEntry, error) {
	rows, err := upcomingPriceEntries(ctx, p.store.db, after)
	if err != nil {
		return nil, err
	}
	upcoming := make([]pricing.UpcomingEntry, 0, len(rows))
	for _, row := range rows {
		upcoming = append(upcoming, pricing.UpcomingEntry{
			Entry: row.Entry(), EffectiveAt: row.EffectiveAt, Withdrawn: row.Withdrawn, Reason: row.Reason,
		})
	}
	return upcoming, nil
}

func (p priceSource) UpcomingMarkups(ctx context.Context, after time.Time) ([]pricing.UpcomingMarkup, error) {
	var changes []PriceChange
	err := p.store.db.NewSelect().Model(&changes).
		Where("cancelled_at IS NULL").
		Where("effective_at > ?", after).
		Where("markup_bps IS NOT NULL").
		Order("effective_at", "name").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: upcoming markups: %w", err))
	}
	markups := make([]pricing.UpcomingMarkup, 0, len(changes))
	for _, change := range changes {
		markups = append(markups, pricing.UpcomingMarkup{
			Change: change.Name, EffectiveAt: change.EffectiveAt, MarkupBps: *change.MarkupBps, Reason: change.Reason,
		})
	}
	return markups, nil
}

// PriceListSnapshot is the price list in force at a moment and what is scheduled after it,
// read in one snapshot so a change committed between the reads cannot show half of itself.
type PriceListSnapshot struct {
	Entries   []PriceEntry
	MarkupBps int64
	// Upcoming are the entries of the changes not cancelled that take effect later, soonest
	// first, and UpcomingMarkupBps the list's markup at each one's effective time, by the
	// time's Unix microseconds.
	Upcoming          []UpcomingPriceEntry
	UpcomingMarkupBps map[int64]int64
}

// PriceListAt reads the price list in force at at, and what is scheduled after it.
func (s *Store) PriceListAt(ctx context.Context, at time.Time) (PriceListSnapshot, error) {
	var list PriceListSnapshot
	err := s.db.RunInTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true}, func(ctx context.Context, tx bun.Tx) error {
		var err error
		if list.Entries, err = activePriceEntries(ctx, tx, at); err != nil {
			return err
		}
		if list.MarkupBps, err = activeMarkup(ctx, tx, at); err != nil {
			return err
		}
		if list.Upcoming, err = upcomingPriceEntries(ctx, tx, at); err != nil {
			return err
		}
		list.UpcomingMarkupBps = map[int64]int64{}
		for _, entry := range list.Upcoming {
			key := entry.EffectiveAt.UnixMicro()
			if _, ok := list.UpcomingMarkupBps[key]; ok {
				continue
			}
			if list.UpcomingMarkupBps[key], err = activeMarkup(ctx, tx, entry.EffectiveAt); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return PriceListSnapshot{}, stack.Wrap(fmt.Errorf("store: price list at %s: %w", at.Format(time.RFC3339), err))
	}
	return list, nil
}

// PriceEntryWrite is an entry a change writes: the item's whole entry under its When, at the
// vendor's rates, or its withdrawal.
type PriceEntryWrite struct {
	pricing.Entry
	// Withdrawn stops pricing the item under Entry.When from the change's effective time on;
	// its entries under other Whens are untouched.
	Withdrawn bool
}

// PriceChangeWrite is a change made through the management API.
type PriceChangeWrite struct {
	// Name is the caller's, of [a-z0-9_]; the change is named <UTC yyyymmddhhmmss>_<Name>.
	Name   string
	Reason string
	Source string
	// EffectiveAt nil, or no later than now, is now.
	EffectiveAt *time.Time
	// MarkupBps sets the list's markup from the effective time on; nil leaves it.
	MarkupBps      *int64
	Entries        []PriceEntryWrite
	IdempotencyKey string
	ActorID        string
	ActorName      string
}

// PriceListAt is the price list in force at the moment a change would take effect, before it.
type PriceListAt struct {
	// Now is the database's clock, and At when the change would take effect.
	Now, At   time.Time
	Entries   []PriceEntry
	MarkupBps int64
}

// effectiveAt is when a change asked to take effect at asked does, by the database's clock.
func effectiveAt(now time.Time, asked *time.Time) (time.Time, error) {
	if asked == nil {
		return now, nil
	}
	if asked.Before(now.Add(-PriceChangeGrace)) {
		return time.Time{}, ErrPriceChangeInPast
	}
	if !asked.After(now) {
		return now, nil
	}
	return asked.UTC().Truncate(time.Microsecond), nil
}

// priceListAt reads the list a change asked to take effect at asked lands on.
func priceListAt(ctx context.Context, db bun.IDB, asked *time.Time) (PriceListAt, error) {
	var list PriceListAt
	if err := db.QueryRowContext(ctx, "SELECT now()").Scan(&list.Now); err != nil {
		return PriceListAt{}, err
	}
	var err error
	if list.At, err = effectiveAt(list.Now, asked); err != nil {
		return PriceListAt{}, err
	}
	if list.Entries, err = activePriceEntries(ctx, db, list.At); err != nil {
		return PriceListAt{}, err
	}
	if list.MarkupBps, err = activeMarkup(ctx, db, list.At); err != nil {
		return PriceListAt{}, err
	}
	return list, nil
}

// PriceListBefore is the list a change asked to take effect at asked would land on, writing
// nothing: what a preview compares the change with.
func (s *Store) PriceListBefore(ctx context.Context, asked *time.Time) (PriceListAt, error) {
	var list PriceListAt
	err := s.db.RunInTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true}, func(ctx context.Context, tx bun.Tx) error {
		var err error
		list, err = priceListAt(ctx, tx, asked)
		return err
	})
	if errors.Is(err, ErrPriceChangeInPast) {
		return PriceListAt{}, err
	}
	if err != nil {
		return PriceListAt{}, stack.Wrap(fmt.Errorf("store: price list before a change: %w", err))
	}
	return list, nil
}

// CreatePriceChange writes a change, its entries and its audit row in one transaction, under
// the lock a router's start loads its price files under. check is handed the list the change
// lands on and refuses the change by returning an error, which CreatePriceChange returns as it
// is. A change already written under the write's idempotency key is answered instead, with
// replayed true, when it is the same change, and ErrPriceChangeKeyReused when it is not.
func (s *Store) CreatePriceChange(ctx context.Context, write PriceChangeWrite, check func(PriceListAt) error) (PriceChange, []PriceEntry, bool, error) {
	var change PriceChange
	var rows []PriceEntry
	var replayed bool
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, pricesLockTimeout); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, pricesLock); err != nil {
			return err
		}
		if write.IdempotencyKey != "" {
			err := tx.NewSelect().Model(&change).Where("idempotency_key = ?", write.IdempotencyKey).Scan(ctx)
			switch {
			case err == nil:
				written, err := priceEntriesOf(ctx, tx, change.Name)
				if err != nil {
					return err
				}
				if !samePriceChange(change, written, write) {
					return ErrPriceChangeKeyReused
				}
				rows, replayed = written, true
				return nil
			case !errors.Is(err, sql.ErrNoRows):
				return err
			}
		}
		var seeded bool
		if err := tx.QueryRowContext(ctx, "SELECT EXISTS (SELECT 1 FROM price_changes)").Scan(&seeded); err != nil {
			return err
		}
		if !seeded {
			return ErrPriceListNotSeeded
		}
		before, err := priceListAt(ctx, tx, write.EffectiveAt)
		if err != nil {
			return err
		}
		if err := check(before); err != nil {
			return refusedByCheck{err}
		}
		change = PriceChange{
			Name:           before.Now.UTC().Format("20060102150405") + "_" + write.Name,
			Kind:           PriceChangeAPI,
			Reason:         write.Reason,
			Source:         write.Source,
			EffectiveAt:    before.At,
			MarkupBps:      write.MarkupBps,
			CreatedBy:      write.ActorID,
			IdempotencyKey: write.IdempotencyKey,
		}
		var taken bool
		if err := tx.QueryRowContext(ctx, "SELECT EXISTS (SELECT 1 FROM price_changes WHERE name = ?)", change.Name).Scan(&taken); err != nil {
			return err
		}
		if taken {
			return ErrPriceChangeExists
		}
		if _, err := tx.NewInsert().Model(&change).Returning("*").Exec(ctx); err != nil {
			return err
		}
		rows = make([]PriceEntry, 0, len(write.Entries))
		names := make([]string, 0, len(write.Entries))
		for _, entry := range write.Entries {
			row := priceEntryOf(change.Name, entry.Entry)
			row.Withdrawn = entry.Withdrawn
			rows = append(rows, row)
			name := prices.Name(entry.Entry)
			if entry.Withdrawn {
				name += " (withdrawn)"
			}
			names = append(names, name)
		}
		if len(rows) > 0 {
			if _, err := tx.NewInsert().Model(&rows).Returning("*").Exec(ctx); err != nil {
				return err
			}
		}
		changes := []AuditChange{{Field: "effective_at", After: change.EffectiveAt}}
		if len(names) > 0 {
			changes = append(changes, AuditChange{Field: "entries", After: names})
		}
		if change.MarkupBps != nil {
			changes = append(changes, AuditChange{Field: "markup_bps", Before: before.MarkupBps, After: *change.MarkupBps})
		}
		return recordAuditEntry(ctx, tx, &AuditEntry{
			ResourceType: AuditResourcePriceList,
			ResourceID:   change.Name,
			ResourceName: change.Reason,
			Action:       AuditCreated,
			Source:       AuditSourceManage,
			ActorID:      write.ActorID,
			ActorName:    write.ActorName,
			Changes:      changes,
		})
	})
	var refused refusedByCheck
	switch {
	case errors.As(err, &refused):
		return PriceChange{}, nil, false, refused.err
	case errors.Is(err, ErrPriceListNotSeeded), errors.Is(err, ErrPriceChangeKeyReused),
		errors.Is(err, ErrPriceChangeExists), errors.Is(err, ErrPriceChangeInPast):
		return PriceChange{}, nil, false, err
	case isViolation(err, lockNotAvailable):
		return PriceChange{}, nil, false, ErrPriceListBusy
	case err != nil:
		return PriceChange{}, nil, false, stack.Wrap(fmt.Errorf("store: create price change: %w", err))
	}
	return change, rows, replayed, nil
}

// refusedByCheck carries the error a change's check refused it with out of the transaction,
// to be returned as it is rather than as the database failing.
type refusedByCheck struct{ err error }

func (r refusedByCheck) Error() string { return r.err.Error() }

// samePriceChange reports whether a change already written is the one write asks for, so a
// resubmit is answered with it. The effective time asked for matches when it is the one
// written, or when it was no later than the moment the change was made, which it then took.
// Entries are compared in the order they were written, so a resubmit must list them in the
// order the first did: the same entries in another order are another body.
func samePriceChange(change PriceChange, rows []PriceEntry, write PriceChangeWrite) bool {
	_, name, _ := cutStamp(change.Name)
	if change.Kind != PriceChangeAPI || name != write.Name || change.Reason != write.Reason || change.Source != write.Source ||
		!reflect.DeepEqual(change.MarkupBps, write.MarkupBps) || len(rows) != len(write.Entries) {
		return false
	}
	if write.EffectiveAt != nil {
		asked := write.EffectiveAt.UTC().Truncate(time.Microsecond)
		if asked.Before(change.CreatedAt) {
			asked = change.CreatedAt
		}
		if !asked.Equal(change.EffectiveAt) {
			return false
		}
	}
	for i, row := range rows {
		asked := priceEntryOf(change.Name, write.Entries[i].Entry)
		asked.Withdrawn = write.Entries[i].Withdrawn
		if row.Withdrawn != asked.Withdrawn || !reflect.DeepEqual(row.Entry(), asked.Entry()) {
			return false
		}
	}
	return true
}

// cutStamp splits a change's name into the time it was made and the name it was given.
func cutStamp(name string) (stamp, given string, ok bool) {
	const stampLength = len("20060102150405")
	if len(name) <= stampLength || name[stampLength] != '_' {
		return "", name, false
	}
	return name[:stampLength], name[stampLength+1:], true
}

// CancelPriceChange cancels a change made through the API that is not yet in force, writing
// its cancel and an audit row in one transaction. A change already cancelled is answered as
// it stands, with cancelled false and nothing written, so a cancel sent again changes nothing.
// A change in force, or one the API did not make, is ErrPriceChangeInForce, and one nobody
// made ErrNoPriceChange.
func (s *Store) CancelPriceChange(ctx context.Context, name, reason, actorID, actorName string) (PriceChange, []PriceEntry, bool, error) {
	var change PriceChange
	var rows []PriceEntry
	var cancelled bool
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, pricesLockTimeout); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, pricesLock); err != nil {
			return err
		}
		err := tx.NewSelect().Model(&change).Where("name = ?", name).For("UPDATE").Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			return ErrNoPriceChange
		}
		if err != nil {
			return err
		}
		if rows, err = priceEntriesOf(ctx, tx, name); err != nil {
			return err
		}
		if change.CancelledAt != nil {
			return nil
		}
		var now time.Time
		if err := tx.QueryRowContext(ctx, "SELECT now()").Scan(&now); err != nil {
			return err
		}
		if change.Kind != PriceChangeAPI || !change.EffectiveAt.After(now) {
			return ErrPriceChangeInForce
		}
		change.CancelledAt, change.CancelledBy, change.CancelReason = &now, actorID, reason
		if _, err := tx.NewUpdate().Model(&change).Column("cancelled_at", "cancelled_by", "cancel_reason").WherePK().Exec(ctx); err != nil {
			return err
		}
		cancelled = true
		return recordAuditEntry(ctx, tx, &AuditEntry{
			ResourceType: AuditResourcePriceList,
			ResourceID:   change.Name,
			ResourceName: change.Reason,
			Action:       AuditCancelled,
			Source:       AuditSourceManage,
			ActorID:      actorID,
			ActorName:    actorName,
			Changes: []AuditChange{
				{Field: "cancelled_at", After: now},
				{Field: "cancel_reason", After: reason},
			},
		})
	})
	if err = busyOr(err); errors.Is(err, ErrNoPriceChange) || errors.Is(err, ErrPriceChangeInForce) || errors.Is(err, ErrPriceListBusy) {
		return PriceChange{}, nil, false, err
	}
	if err != nil {
		return PriceChange{}, nil, false, stack.Wrap(fmt.Errorf("store: cancel price change: %w", err))
	}
	return change, rows, cancelled, nil
}
