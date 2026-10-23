package policy

import (
	"context"
	"slices"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const (
	// maxHolds is the most holds one account keeps apart. Past it, the oldest are folded,
	// which counts the same money and lets go of it no sooner.
	maxHolds = 10_000
	// maxFolded is the most usage ids one account remembers inside its folds, so that one
	// recorded or acked again finds its fold. Past it, a folded id is forgotten: recorded
	// again it is held a second time, and an ack for it waits for its fold's timer, so a
	// forgotten id only ever refuses sooner.
	maxFolded = 100_000
	// shippedChunk is how many usage ids one question to the shard names.
	shippedChunk = 1_000
)

// budgetIntervals is every interval a budget can reset on.
var budgetIntervals = []store.BudgetInterval{store.BudgetHourly, store.BudgetDaily, store.BudgetWeekly, store.BudgetMonthly}

// hold is what one row this router recorded spent, kept until the ledgers count it. Keyed by
// the row's usage id, so recording or acking it twice changes nothing. A hold made before the
// work starts, or one another router shares, would take the same shape.
//
// It counts twice over, and each is let go by its own ledger: the balance and the spend hard
// limit are the usage service's, read with the standings, and are let go once a standing read
// after the usage service had the row lands; the budgets are the shard's, summed by the budget
// loop, and are let go once a sum asked for after the shard wrote the row lands. A row the
// ledger is never known to have is let go PendingTTL after the later of when it was made and
// when that ledger last could not be read, once a read after that lands; until one lands it
// is kept, however long.
type hold struct {
	app    string
	micros int64
	// made is when it was recorded, which its timer runs from; at is the same moment by
	// Settings.Now, which places it in the UTC windows a limit or a budget counts over.
	made time.Time
	at   time.Time
	// balance and budget say whether it still counts against the balance and the spend hard
	// limit, and against the budgets.
	balance bool
	budget  bool
	// ackedAt is when this process knew the usage service had the row, writtenAt when it knew
	// the shard did; zero until then.
	ackedAt   time.Time
	writtenAt time.Time
	// group is, for a hold that carries others, the state they were folded in; nil for a row's own.
	group *fold
}

// fold is what holds are carried together by: one app, one UTC hour (every window a limit or
// a budget counts over starts on an hour), and the same state in each ledger.
type fold struct {
	app     string
	hour    int64
	balance bool
	budget  bool
	acked   bool
	written bool
}

// state is the fold the hold would be carried in.
func (h *hold) state() fold {
	return fold{
		app: h.app, hour: h.at.UTC().Truncate(time.Hour).Unix(), balance: h.balance, budget: h.budget,
		acked: !h.ackedAt.IsZero(), written: !h.writtenAt.IsZero(),
	}
}

// folded is a usage id carried in a fold, and its share of it.
type folded struct {
	into   uuid.UUID
	micros int64
}

// window is one running total of what an account holds: against the spend hard limit (budget
// false) or against a budget's scope, over one interval that starts at start.
type window struct {
	budget   bool
	kind     store.PolicyScope
	id       string
	interval string
	start    int64
}

// count adds sign times the hold to the account's totals, in each ledger it still counts in.
func (a *account) count(key string, h *hold, sign int64) {
	if h.balance {
		a.countBalance(h, sign)
	}
	if h.budget {
		a.countBudget(key, h, sign)
	}
}

func (a *account) countBalance(h *hold, sign int64) {
	a.held += sign * h.micros
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		start, _ := usage.Window(interval, h.at)
		a.add(window{interval: string(interval), start: start.Unix()}, sign*h.micros)
	}
}

func (a *account) countBudget(key string, h *hold, sign int64) {
	scopes := []scope{{kind: store.ScopeApp, id: h.app}}
	if billed(key) {
		scopes = append(scopes, scope{kind: store.ScopeOrganization, id: key})
	}
	for _, whose := range scopes {
		for _, interval := range budgetIntervals {
			start, _ := interval.Window(h.at)
			a.add(window{budget: true, kind: whose.kind, id: whose.id, interval: string(interval), start: start.Unix()}, sign*h.micros)
		}
	}
}

func (a *account) add(w window, micros int64) {
	if a.windows == nil {
		a.windows = map[window]int64{}
	}
	if a.windows[w] += micros; a.windows[w] == 0 {
		delete(a.windows, w)
	}
}

// heldAgainstLimit is what the account holds against its spend hard limit for the interval
// that now is in: only what was spent since that window began.
func (a *account) heldAgainstLimit(interval usage.Interval, now time.Time) int64 {
	start, _ := usage.Window(interval, now)
	return a.windows[window{interval: string(interval), start: start.Unix()}]
}

// heldAgainstBudget is what the account holds against the budget's scope in the window that
// now is in.
func (a *account) heldAgainstBudget(budget scoped, now time.Time) int64 {
	start, _ := budget.budget.Interval.Window(now)
	return a.windows[window{budget: true, kind: budget.scope, id: budget.id, interval: string(budget.budget.Interval), start: start.Unix()}]
}

// Spent holds what a unit of work recorded here cost against its owner's organization and app,
// under the row's usage id, until the ledgers count the row. Holding it again replaces the
// amount, restarts its timer and leaves what the ledgers were known to have as it was; holding
// one carried in a fold again changes nothing. It never waits.
func (e *Enforcer) Spent(owner routing.Owner, usageID uuid.UUID, micros int64) {
	if e == nil || owner.CustomerID == "" || usageID == uuid.Nil || micros <= 0 || e.money.off() {
		return
	}
	organization := owner.OrganizationID
	if decided, ok := e.decided(owner.CustomerID); ok && decided.organization != "" {
		organization = decided.organization
	}
	if organization == "" {
		organization = appAccount(owner.CustomerID)
	}
	m := &e.money
	now, at := time.Now(), m.settings.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	var held *hold
	if previous, ok := m.holdOf[usageID]; ok {
		if other := m.accounts[previous]; other != nil {
			if _, carried := other.folded[usageID]; carried {
				return
			}
			if held = other.holds[usageID]; held != nil {
				other.count(previous, held, -1)
				delete(other.holds, usageID)
			}
		}
	}
	acct := m.accounts[organization]
	if acct == nil {
		acct = &account{wanted: now}
		m.accounts[organization] = acct
	}
	acct.used = now
	if acct.holds == nil {
		acct.holds = map[uuid.UUID]*hold{}
	}
	if held == nil {
		held = &hold{at: at, balance: m.billing(organization), budget: true}
	}
	held.app, held.micros, held.made = owner.CustomerID, micros, now
	acct.holds[usageID] = held
	acct.count(organization, held, 1)
	m.holdOf[usageID] = organization
}

// Acked notes the rows the usage service has: a hold kept for one of them stops counting
// against the balance and the spend hard limit once a standing asked for after now lands. An
// id this process holds nothing for, as another router's row, is ignored. It is
// WriterOptions.Shipped and SweeperOptions.Shipped.
func (e *Enforcer) Acked(acked []routing.Acked) {
	if e == nil || e.money.off() {
		return
	}
	e.money.settled(acked, func(h *hold) *time.Time { return &h.ackedAt }, func(f *fold) *bool { return &f.acked })
}

// Written notes the rows the shard has, which is what a budget is summed from: a hold kept for
// one of them stops counting against the budgets once a sum asked for after now lands. It is
// WriterOptions.Written.
func (e *Enforcer) Written(written []routing.Acked) {
	if e == nil || e.money.off() {
		return
	}
	e.money.settled(written, func(h *hold) *time.Time { return &h.writtenAt }, func(f *fold) *bool { return &f.written })
}

// settled notes that a ledger has the rows: when, in the hold of each, or by moving a folded
// row into the fold that ledger already has.
func (m *money) settled(rows []routing.Acked, when func(*hold) *time.Time, has func(*fold) *bool) {
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, row := range rows {
		key, ok := m.holdOf[row.UsageID]
		if !ok {
			continue
		}
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		if held := acct.holds[row.UsageID]; held != nil {
			if at := when(held); at.IsZero() {
				*at = now
			}
			continue
		}
		carried, ok := acct.folded[row.UsageID]
		if !ok {
			continue
		}
		from := acct.holds[carried.into]
		if from == nil {
			delete(acct.folded, row.UsageID)
			delete(m.holdOf, row.UsageID)
			continue
		}
		state := from.state()
		if *has(&state) {
			continue
		}
		*has(&state) = true
		into, intoID := acct.foldInto(state)
		from.micros -= carried.micros
		into.micros += carried.micros
		into.made = later(into.made, from.made)
		into.ackedAt, into.writtenAt = later(into.ackedAt, from.ackedAt), later(into.writtenAt, from.writtenAt)
		if at := when(into); at.IsZero() {
			*at = now
		}
		acct.folded[row.UsageID] = folded{into: intoID, micros: carried.micros}
		if from.micros == 0 {
			acct.drop(carried.into, from)
		}
	}
}

// billing reports whether a hold against the account counts against a balance: an
// organization's, with a usage service to bill it.
func (m *money) billing(key string) bool { return m.settings.Usage != nil && billed(key) }

// release lets go, under m.mu, of what the reads that landed now count: against the balance,
// holds acked before the last standing that landed was asked for; against the budgets, holds
// written before the last sum that landed was asked for, and every hold once no budget is in
// use. A hold the ledger is never known to have goes PendingTTL after the later of when it was
// made and when that ledger last failed to read, once a read asked for after that lands.
// Past maxHolds the oldest are folded.
func (m *money) release(key string, acct *account, now time.Time) {
	ttl := m.settings.PendingTTL
	var expired, unwritten, waiting, waited int64
	var emptied []uuid.UUID
	for id, held := range acct.holds {
		if held.balance {
			switch {
			case !held.ackedAt.IsZero() && held.ackedAt.Before(acct.standingFrom):
				acct.letGoBalance(held)
			case !acct.standingFrom.Before(later(held.made, acct.standingFailed).Add(ttl)):
				expired += held.micros
				acct.letGoBalance(held)
			case now.Sub(held.made) > ttl:
				waiting++
				waited += held.micros
			}
		}
		if held.budget {
			switch {
			case !acct.budgeted:
				acct.letGoBudget(key, held)
			case !held.writtenAt.IsZero() && held.writtenAt.Before(acct.measuredFrom):
				acct.letGoBudget(key, held)
			case !acct.measuredFrom.Before(later(held.made, acct.measureFailed).Add(ttl)):
				unwritten += held.micros
				acct.letGoBudget(key, held)
			}
		}
		if held.balance || held.budget {
			continue
		}
		delete(acct.holds, id)
		if held.group == nil {
			delete(m.holdOf, id)
		} else {
			emptied = append(emptied, id)
		}
	}
	if len(emptied) > 0 {
		for id, carried := range acct.folded {
			if slices.Contains(emptied, carried.into) {
				delete(acct.folded, id)
				delete(m.holdOf, id)
			}
		}
	}
	logger := m.enforcer.logger
	if expired > 0 {
		logger.Warn("usage held against an "+noun(key)+" expired before the usage service was known to count it",
			append(who(key), "micros", expired, "pending_ttl", ttl)...)
	}
	if unwritten > 0 {
		logger.Warn("usage held against an "+noun(key)+"'s budgets expired before the shard was known to have it",
			append(who(key), "micros", unwritten, "pending_ttl", ttl)...)
	}
	if waiting > 0 && m.waiting.due(key, logEvery) {
		logger.Warn("usage held against an "+noun(key)+" is past its pending_ttl with no read of its balance to count it, so it is kept",
			append(who(key), "holds", waiting, "micros", waited, "last_failed_read", acct.standingFailed)...)
	}
	if len(acct.holds) > maxHolds {
		m.carry(key, acct)
	}
}

func (a *account) letGoBalance(h *hold) {
	a.countBalance(h, -1)
	h.balance = false
	a.reopen(h)
}

func (a *account) letGoBudget(key string, h *hold) {
	a.countBudget(key, h, -1)
	h.budget = false
	a.reopen(h)
}

// reopen forgets a fold as the one to carry its old state in, now that its state changed.
func (a *account) reopen(h *hold) {
	if h.group == nil {
		return
	}
	if id, ok := a.folds[*h.group]; ok && a.holds[id] == h {
		delete(a.folds, *h.group)
	}
}

// drop removes an emptied fold.
func (a *account) drop(id uuid.UUID, h *hold) {
	a.reopen(h)
	delete(a.holds, id)
}

// foldInto is the fold the account carries holds of state in, made when it has none open.
func (a *account) foldInto(state fold) (*hold, uuid.UUID) {
	if id, ok := a.folds[state]; ok {
		if into := a.holds[id]; into != nil && into.group != nil {
			return into, id
		}
	}
	if a.folds == nil {
		a.folds = map[fold]uuid.UUID{}
	}
	id := uuid.New()
	group := state
	into := &hold{app: state.app, at: time.Unix(state.hour, 0).UTC(), balance: state.balance, budget: state.budget, group: &group}
	a.holds[id] = into
	a.folds[state] = id
	return into, id
}

// carry folds the account's oldest holds until it keeps half of maxHolds apart, under m.mu.
// A fold is held as its holds were, in the same windows and ledgers, and expires with the
// last of them, so it counts the same money and lets go of it no sooner. Each folded id is
// remembered up to maxFolded, so recording or acking it again still finds it.
func (m *money) carry(key string, acct *account) {
	own := make([]uuid.UUID, 0, len(acct.holds))
	for id, held := range acct.holds {
		if held.group == nil {
			own = append(own, id)
		}
	}
	if len(own) <= maxHolds/2 {
		return
	}
	slices.SortFunc(own, func(a, b uuid.UUID) int { return acct.holds[a].made.Compare(acct.holds[b].made) })
	if acct.folded == nil {
		acct.folded = map[uuid.UUID]folded{}
	}
	var carried, forgotten, micros int64
	for _, id := range own[:len(own)-maxHolds/2] {
		held := acct.holds[id]
		into, intoID := acct.foldInto(held.state())
		into.micros += held.micros
		into.made = later(into.made, held.made)
		into.ackedAt, into.writtenAt = later(into.ackedAt, held.ackedAt), later(into.writtenAt, held.writtenAt)
		delete(acct.holds, id)
		carried++
		micros += held.micros
		if len(acct.folded) < maxFolded {
			acct.folded[id] = folded{into: intoID, micros: held.micros}
			continue
		}
		forgotten++
		delete(m.holdOf, id)
	}
	m.enforcer.logger.Warn("an "+noun(key)+" holds more usage than the router keeps apart, so the oldest is folded",
		append(who(key), "holds", carried, "micros", micros, "kept", maxHolds/2, "forgotten_ids", forgotten)...)
}

// shipped asks the shard which of the rows held against a balance with no ack it has shipped
// to the usage service, as another router's sweeper does without this one hearing of it, and
// notes them acked now. It runs on the budget loop, which is the one that reads Postgres.
func (m *money) shipped(ctx context.Context) {
	db := m.enforcer.config.DB()
	if m.settings.Usage == nil || db == nil {
		return
	}
	now := time.Now()
	var asked []uuid.UUID
	m.mu.Lock()
	for _, acct := range m.accounts {
		for id, held := range acct.holds {
			if held.group == nil && held.balance && held.ackedAt.IsZero() && now.Sub(held.made) > m.settings.Fresh {
				asked = append(asked, id)
			}
		}
	}
	m.mu.Unlock()
	for start := 0; start < len(asked); start += shippedChunk {
		call, cancel := context.WithTimeout(ctx, readTimeout)
		found, err := db.Shipped(call, asked[start:min(start+shippedChunk, len(asked))])
		cancel()
		if err != nil {
			m.failedRead("could not ask the shard which held rows it has shipped", err, "rows", len(asked))
			return
		}
		m.settled(acks(found), func(h *hold) *time.Time { return &h.ackedAt }, func(f *fold) *bool { return &f.acked })
	}
}

func acks(ids []uuid.UUID) []routing.Acked {
	out := make([]routing.Acked, len(ids))
	for i, id := range ids {
		out[i] = routing.Acked{UsageID: id}
	}
	return out
}

// later is the later of two times; a zero time is the earliest.
func later(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}
