package pricing

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// ActiveVersion names a list read from a Source on a quote at an entry with no version of
// its own. Every entry a Source answers has one, the change that wrote it, so only a local
// entry with no rate would take it, and versionOf names nothing for that one.
const ActiveVersion = "database"

// reloadTimeout bounds one read of the list, so a stuck database cannot hold the reloads up.
const reloadTimeout = 15 * time.Second

// emptyEvery is how often a database with no entries is logged while it stays empty.
const emptyEvery = time.Minute

// UpcomingEntry is an entry a scheduled change will set, with what the change says.
type UpcomingEntry struct {
	// Entry is at the vendor's rates; its Version is the change's name.
	Entry
	EffectiveAt time.Time
	// Withdrawn stops pricing the item under Entry.When at EffectiveAt.
	Withdrawn bool
	// Reason is the change's.
	Reason string
}

// Source is where the list in force is read from: the shared database, through the store.
type Source interface {
	// Active is every entry in force at at, at the vendor's rates and each with its version,
	// the list's markup at at, and when the next change not cancelled takes effect after at
	// (zero when none is scheduled).
	Active(ctx context.Context, at time.Time) (entries []Entry, markupBps int64, next time.Time, err error)
	// Upcoming is every entry of a change not cancelled that takes effect after after.
	Upcoming(ctx context.Context, after time.Time) ([]UpcomingEntry, error)
}

// ActiveOptions configures an Active.
type ActiveOptions struct {
	Source Source
	// Fallback prices until the first read answers any entry: the price files the binary
	// carries, at the vendors' rates.
	Fallback *List
	// Local are the items the deployment runs itself, marked local in every list read.
	Local []Item
	// Interval is how often the list is read again.
	Interval time.Duration
	// RequireEntries keeps Ready false until a read answers at least one entry, for a
	// deployment whose fallback must never be what it serves on.
	RequireEntries bool
	Logger         *slog.Logger
}

// Active is the price list in force, read again every interval and at the moment the next
// scheduled change takes effect. Until the first read answers any entry it prices with the
// fallback. A failed read keeps the list it had.
type Active struct {
	source         Source
	local          []Item
	interval       time.Duration
	requireEntries bool
	logger         *slog.Logger
	ctx            context.Context

	current  atomic.Pointer[List]
	upcoming atomic.Pointer[[]UpcomingEntry]
	// unpriced is when each unpriced item was last logged, shared by every list read.
	unpriced *unpricedLog
	// read is whether any read succeeded, and entries whether one answered an entry.
	read, entries atomic.Bool

	// mu serialises reads, so an older read never replaces a newer one, and guards the rest.
	mu          sync.Mutex
	failing     bool
	emptyLogged time.Time
	timer       *time.Timer
	timerAt     time.Time
	// clock is the wall clock a timer that fired checks against; nil is time.Now. A test sets
	// it to step the clock back.
	clock func() time.Time
}

// NewActive reads the list once, then again every interval and at each scheduled change,
// until ctx ends.
func NewActive(ctx context.Context, options ActiveOptions) *Active {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	a := &Active{
		source:         options.Source,
		local:          options.Local,
		interval:       options.Interval,
		requireEntries: options.RequireEntries,
		logger:         logger,
		ctx:            ctx,
	}
	fallback := options.Fallback
	if fallback == nil {
		// An empty list cannot be refused.
		fallback, _ = NewList(ActiveVersion, nil, 0, logger)
	}
	// Every list read after the fallback logs an unpriced item as the fallback did, so one
	// logged a moment ago is not logged again because the list was read again.
	a.unpriced = fallback.unpriced
	a.current.Store(fallback)
	_ = a.reload(ctx, time.Now())
	go a.loop()
	return a
}

func (a *Active) loop() {
	ticker := time.NewTicker(a.interval)
	defer ticker.Stop()
	for {
		select {
		case <-a.ctx.Done():
			a.mu.Lock()
			if a.timer != nil {
				a.timer.Stop()
			}
			a.mu.Unlock()
			return
		case <-ticker.C:
			_ = a.reload(a.ctx, time.Now())
		}
	}
}

// Reload reads the list now. The router that wrote a change calls it, so the change prices
// its next request without waiting for the interval.
func (a *Active) Reload(ctx context.Context) error {
	return a.reload(ctx, time.Now())
}

func (a *Active) reload(ctx context.Context, at time.Time) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	ctx, cancel := context.WithTimeout(ctx, reloadTimeout)
	defer cancel()
	entries, markup, next, err := a.source.Active(ctx, at)
	var upcoming []UpcomingEntry
	if err == nil {
		upcoming, err = a.source.Upcoming(ctx, at)
	}
	var list *List
	if err == nil && len(entries) > 0 {
		list, err = NewList(ActiveVersion, WithLocal(entries, a.local), markup, a.logger)
		if list != nil {
			list.unpriced = a.unpriced
		}
	}
	if err != nil {
		if !a.failing {
			a.logger.Error("could not read the price list; pricing with the one read before", "error", err)
		}
		a.failing = true
		return err
	}
	if a.failing {
		a.logger.Info("read the price list again")
		a.failing = false
	}
	a.read.Store(true)
	a.upcoming.Store(&upcoming)
	a.arm(next)
	if list == nil {
		if now := time.Now(); now.Sub(a.emptyLogged) >= emptyEvery {
			a.logger.Error("the price list is empty: the router seeds it at startup")
			a.emptyLogged = now
		}
		return nil
	}
	a.entries.Store(true)
	a.current.Store(list)
	return nil
}

// arm reads the list again at next, so a change effective at midnight prices from midnight
// rather than up to an interval later. Zero disarms it.
func (a *Active) arm(next time.Time) {
	if next.Equal(a.timerAt) && a.timer != nil {
		return
	}
	if a.timer != nil {
		a.timer.Stop()
		a.timer = nil
	}
	a.timerAt = next
	if next.IsZero() {
		return
	}
	a.timer = time.AfterFunc(time.Until(next), func() { a.fire(next) })
}

// fire reads the list once the change due at next is in force. A timer runs on the monotonic
// clock and next is a wall-clock time, so when the wall clock stepped back while it ran, the
// rest of the wait is waited out rather than the change read early.
func (a *Active) fire(next time.Time) {
	if wait := next.Sub(a.wallClock()); wait > 0 {
		a.mu.Lock()
		defer a.mu.Unlock()
		if a.timerAt.Equal(next) && a.ctx.Err() == nil {
			a.timer = time.AfterFunc(wait, func() { a.fire(next) })
		}
		return
	}
	_ = a.reload(a.ctx, time.Now())
}

// wallClock is the wall-clock time now, stripped of the monotonic reading, as next is.
func (a *Active) wallClock() time.Time {
	if a.clock != nil {
		return a.clock()
	}
	return time.Now().Round(0)
}

// Ready says why the list is not yet one to serve on, or nil once it is: once a read
// succeeded, and with RequireEntries once a read answered an entry.
func (a *Active) Ready() error {
	switch {
	case !a.read.Load():
		return errors.New("the price list has not been read from the database yet")
	case a.requireEntries && !a.entries.Load():
		return errors.New("the price list in the database is empty: the router seeds it at startup")
	}
	return nil
}

// List is the list in force.
func (a *Active) List() *List { return a.current.Load() }

// Upcoming is the entries of the changes not cancelled and not yet in force, as the last
// read found them.
func (a *Active) Upcoming() []UpcomingEntry {
	if upcoming := a.upcoming.Load(); upcoming != nil {
		return *upcoming
	}
	return nil
}

func (a *Active) Quote(item Item, used Quantities) Quote { return a.List().Quote(item, used) }

func (a *Active) Quoted(item Item, micros int64) Quote { return a.List().Quoted(item, micros) }

func (a *Active) Billable(item Item) bool { return a.List().Billable(item) }

func (a *Active) Entry(item Item) (Entry, bool) { return a.List().Entry(item) }

func (a *Active) Charged(item Item) (Entry, bool) { return a.List().Charged(item) }
