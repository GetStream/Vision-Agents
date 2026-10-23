package policy

import (
	"context"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const (
	// maxHolds is the most holds one account keeps apart. Past it, the oldest are folded,
	// which counts the same money and lets go of it no sooner.
	maxHolds = 10_000
	// maxFolded is the most usage ids one account remembers inside its folds, so that one
	// recorded or acked again finds its fold, and maxFoldedInProcess the most every account
	// of every Enforcer in the process remembers together. Past either, a folded id is
	// forgotten: recorded again it is held a second time, and an ack for it waits for its
	// fold's timer, so a forgotten id only ever refuses sooner, never lets money go twice.
	//
	// A remembered id costs about 120 bytes (its entry in folded and in holdOf), and a hold
	// kept apart about 250, so an account holds at most about 15 MB and the folded ids of the
	// whole process at most about 60 MB.
	maxFolded          = 100_000
	maxFoldedInProcess = 500_000
	// shippedChunk is how many usage ids one question to the shard names, and shippedPerPass
	// how many the budget loop asks about at most each pass, the oldest first.
	shippedChunk   = 1_000
	shippedPerPass = 10_000
	// shippedTimeout bounds asking the shard on each pass, apart from the budget sums'.
	shippedTimeout = 15 * time.Second
)

// foldedInProcess is how many folded ids every account of every Enforcer remembers.
var foldedInProcess atomic.Int64

var forgottenFolded = promauto.NewCounter(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "admission_folded_ids_forgotten_total",
	Help:      "Usage ids folded past what an account or the process remembers, which are then held until their fold's timer rather than their ack",
})

// hold is what one row this router recorded spent, kept until the ledgers count it. Keyed by
// the row's usage id, so recording or acking it twice changes nothing. A hold made before the
// work starts, or one another router shares, would take the same shape.
//
// It counts twice over, and each is let go by its own ledger: the balance and the spend hard
// limit are the usage service's, read with the standings, and are let go once a standing read
// after the usage service had the row lands. The customer's limits count what each of the
// row's subjects spent: from the usage service's counters, let go once a read of them asked
// for after an ack that says the counters took the row lands (or a read of the service itself
// after any ack); with no usage service from the shard, let go once a sum asked for after the
// shard wrote the row lands. A row the ledger is never known to have is let go PendingTTL
// after the later of when it was made and when that ledger last could not be read, once a
// read after that lands; until one lands it is kept, however long.
type hold struct {
	app    string
	micros int64
	// subjects are the counter keys of every subject the row counts for, and subjectsKey them
	// joined, which folds are told apart by.
	subjects    []string
	subjectsKey string
	// made is when it was recorded, which its timer runs from; at is the same moment by
	// Settings.Now, which places it in the UTC windows a limit counts over.
	made time.Time
	at   time.Time
	// balance and limits say whether it still counts against the balance and the spend hard
	// limit, and against the customer's limits.
	balance bool
	limits  bool
	// ackedAt is when this process knew the usage service had the row, and mirrored whether
	// that ack said its counters took it; writtenAt is when it knew the shard had it; zero
	// until then.
	ackedAt   time.Time
	mirrored  bool
	writtenAt time.Time
	// group is, for a hold that carries others, the state they were folded in; nil for a row's own.
	group *fold
}

// fold is what holds are carried together by: one app and the same subjects, one UTC hour
// (every window a limit counts over starts on an hour), and the same state in each ledger.
type fold struct {
	app      string
	subjects string
	hour     int64
	balance  bool
	limits   bool
	acked    bool
	mirrored bool
	written  bool
}

// state is the fold the hold would be carried in.
func (h *hold) state() fold {
	return fold{
		app: h.app, subjects: h.subjectsKey, hour: h.at.UTC().Truncate(time.Hour).Unix(), balance: h.balance, limits: h.limits,
		acked: !h.ackedAt.IsZero(), mirrored: h.mirrored, written: !h.writtenAt.IsZero(),
	}
}

// subjectsSeparator joins a hold's subject keys. No key holds a newline.
const subjectsSeparator = "\n"

// folded is a usage id carried in a fold, and its share of it.
type folded struct {
	into   uuid.UUID
	micros int64
}

// window is one running total of what an account holds: against the spend hard limit (subject
// empty) or against one subject's limits (its counter key), over one interval that starts at
// start.
type window struct {
	subject  string
	interval string
	start    int64
}

// count adds sign times the hold to the account's totals, in each ledger it still counts in.
func (a *account) count(h *hold, sign int64) {
	if h.balance {
		a.countBalance(h, sign)
	}
	if h.limits {
		a.countLimits(h, sign)
	}
}

func (a *account) countBalance(h *hold, sign int64) {
	a.held += sign * h.micros
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		start, _ := usage.Window(interval, h.at)
		a.add(window{interval: string(interval), start: start.Unix()}, sign*h.micros)
	}
}

func (a *account) countLimits(h *hold, sign int64) {
	for _, interval := range usage.Intervals {
		start, _ := usage.Window(interval, h.at)
		for _, subject := range h.subjects {
			a.add(window{subject: subject, interval: string(interval), start: start.Unix()}, sign*h.micros)
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

// heldAgainst is what the account holds against one subject's limits in the interval's window
// that now is in.
func (a *account) heldAgainst(ref usage.SubjectRef, interval usage.Interval, now time.Time) int64 {
	start, _ := usage.Window(interval, now)
	return a.windows[window{subject: ref.Key(usage.CounterPrefix), interval: string(interval), start: start.Unix()}]
}

// Spent holds what a unit of work recorded here cost against its owner's organization and
// every subject it counts for, under the row's usage id, until the ledgers count the row.
// Holding it again replaces the amount, restarts its timer and leaves what the ledgers were
// known to have as it was; holding one carried in a fold again changes nothing. It never waits.
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
				other.count(held, -1)
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
		held = &hold{at: at, balance: m.billing(organization), limits: true}
	}
	refs := subjectsOf(owner, limitsOrganization(organization))
	// Its subjects are read with the next spend read once the account has a limit on one,
	// which is what lets go of it.
	m.see(acct, refs, now)
	held.subjects = make([]string, len(refs))
	for i, ref := range refs {
		held.subjects[i] = ref.Key(usage.CounterPrefix)
	}
	held.subjectsKey = strings.Join(held.subjects, subjectsSeparator)
	held.app, held.micros, held.made = owner.CustomerID, micros, now
	acct.holds[usageID] = held
	acct.count(held, 1)
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
	e.money.settled(acked, true, func(h *hold) *time.Time { return &h.ackedAt }, func(f *fold) *bool { return &f.acked })
}

// Written notes the rows the shard has, which is what a deployment with no usage service sums a
// subject's spend from: a hold kept for one of them stops counting against the customer's
// limits once a sum asked for after now lands. It is WriterOptions.Written.
func (e *Enforcer) Written(written []routing.Acked) {
	if e == nil || e.money.off() {
		return
	}
	e.money.settled(written, false, func(h *hold) *time.Time { return &h.writtenAt }, func(f *fold) *bool { return &f.written })
}

// settled notes that a ledger has the rows: when, in the hold of each, or by moving a folded
// row into the fold that ledger already has. For acks it also keeps whether the usage
// service's counters took the row, and has an account whose counters missed one read the
// service itself next.
func (m *money) settled(rows []routing.Acked, acks bool, when func(*hold) *time.Time, has func(*fold) *bool) {
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
		if acks && !row.Mirrored {
			acct.readThrough = true
		}
		if held := acct.holds[row.UsageID]; held != nil {
			if at := when(held); at.IsZero() {
				*at = now
				if acks {
					held.mirrored = row.Mirrored
				}
			}
			continue
		}
		carried, ok := acct.folded[row.UsageID]
		if !ok {
			continue
		}
		from := acct.holds[carried.into]
		if from == nil {
			acct.forget(m, row.UsageID)
			continue
		}
		state := from.state()
		if *has(&state) {
			continue
		}
		*has(&state) = true
		if acks {
			state.mirrored = row.Mirrored
		}
		into, intoID := acct.foldInto(state)
		from.micros -= carried.micros
		into.micros += carried.micros
		into.made = later(into.made, from.made)
		into.ackedAt, into.writtenAt = later(into.ackedAt, from.ackedAt), later(into.writtenAt, from.writtenAt)
		// The fold this joins may have been acked before: it is let go only by a read asked for
		// after this ack too, so what it already held waits with what just joined it.
		*when(into) = now
		acct.folded[row.UsageID] = folded{into: intoID, micros: carried.micros}
		if from.micros == 0 {
			acct.drop(carried.into, from)
		}
	}
}

// billing reports whether a hold against the account counts against a balance: an
// organization's, with a usage service to bill it.
func (m *money) billing(key string) bool { return m.settings.Usage != nil && billed(key) }

// counted reports whether an account's subjects' spend is the usage service's, read from its
// counters, rather than summed from the shard.
func (m *money) counted(key string) bool { return m.settings.Usage != nil && billed(key) }

// release lets go, under m.mu, of what the reads that landed now count: against the balance,
// holds acked before the last standing that landed was asked for; against the customer's
// limits, holds the last read of the subjects' spend that landed counts (counted after their
// ack, or summed after the shard wrote them), and every hold while the account has no limit. A
// hold the ledger is never known to have goes PendingTTL after the later of when it was made
// and when that ledger last failed to read, once a read asked for after that lands. Past
// maxHolds the oldest are folded.
func (m *money) release(key string, acct *account, now time.Time) {
	counted := m.counted(key)
	ttl := m.settings.PendingTTL
	var expired, unwritten, waiting, waited int64
	var emptied []uuid.UUID
	for id, held := range acct.holds {
		if held.balance {
			switch {
			case !held.ackedAt.IsZero() && held.ackedAt.Before(acct.standingFrom):
				acct.letGoBalance(held)
			case !acct.standingFrom.Before(later(held.made, acct.balanceFailed).Add(ttl)):
				expired += held.micros
				acct.letGoBalance(held)
			case now.Sub(held.made) > ttl:
				waiting++
				waited += held.micros
			}
		}
		if held.limits {
			acked := !held.ackedAt.IsZero()
			switch {
			case !acct.limited:
				acct.letGoLimits(held)
			case counted && acked && held.ackedAt.Before(acct.throughFrom),
				counted && acked && held.mirrored && held.ackedAt.Before(acct.spentFrom):
				acct.letGoLimits(held)
			case !counted && !held.writtenAt.IsZero() && held.writtenAt.Before(acct.spentFrom):
				acct.letGoLimits(held)
			case !acct.spentFrom.Before(later(held.made, acct.spendFailed).Add(ttl)):
				unwritten += held.micros
				acct.letGoLimits(held)
			}
		}
		if held.balance || held.limits {
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
				acct.forget(m, id)
			}
		}
	}
	logger := m.enforcer.logger
	if expired > 0 {
		logger.Warn("usage held against an "+noun(key)+" expired before the usage service was known to count it",
			append(who(key), "micros", expired, "pending_ttl", ttl)...)
	}
	if unwritten > 0 {
		logger.Warn("usage held against an "+noun(key)+"'s spend limits expired before what they read was known to count it",
			append(who(key), "micros", unwritten, "pending_ttl", ttl)...)
	}
	if waiting > 0 && m.waiting.due(key, logEvery) {
		logger.Warn("usage held against an "+noun(key)+" is past its pending_ttl with no read of its balance to count it, so it is kept",
			append(who(key), "holds", waiting, "micros", waited, "last_failed_read", acct.balanceFailed)...)
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

func (a *account) letGoLimits(h *hold) {
	a.countLimits(h, -1)
	h.limits = false
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

// remember keeps a folded id, unless the account or the process remembers as many as it may;
// it says whether it did.
func (a *account) remember(id uuid.UUID, into folded) bool {
	if len(a.folded) >= maxFolded || foldedInProcess.Load() >= maxFoldedInProcess {
		return false
	}
	a.folded[id] = into
	foldedInProcess.Add(1)
	return true
}

// forget lets go of a folded id, and of what it is held under.
func (a *account) forget(m *money, id uuid.UUID) {
	if _, ok := a.folded[id]; ok {
		delete(a.folded, id)
		foldedInProcess.Add(-1)
	}
	delete(m.holdOf, id)
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
	into := &hold{
		app: state.app, subjectsKey: state.subjects, at: time.Unix(state.hour, 0).UTC(),
		balance: state.balance, limits: state.limits, mirrored: state.mirrored, group: &group,
	}
	if state.subjects != "" {
		into.subjects = strings.Split(state.subjects, subjectsSeparator)
	}
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
	var carried, forgotten, micros int64
	for _, id := range own[:len(own)-maxHolds/2] {
		micros += acct.holds[id].micros
		carried++
		if !m.fold(acct, id) {
			forgotten++
		}
	}
	m.enforcer.logger.Warn("an "+noun(key)+" holds more usage than the router keeps apart, so the oldest is folded",
		append(who(key), "holds", carried, "micros", micros, "kept", maxHolds/2, "forgotten_ids", forgotten,
			"folded_ids_in_process", foldedInProcess.Load())...)
}

// fold carries one of the account's own holds in the fold of its state, under m.mu, and says
// whether its id is remembered there.
func (m *money) fold(acct *account, id uuid.UUID) bool {
	if acct.folded == nil {
		acct.folded = map[uuid.UUID]folded{}
	}
	held := acct.holds[id]
	into, intoID := acct.foldInto(held.state())
	into.micros += held.micros
	into.made = later(into.made, held.made)
	into.ackedAt, into.writtenAt = later(into.ackedAt, held.ackedAt), later(into.writtenAt, held.writtenAt)
	delete(acct.holds, id)
	if acct.remember(id, folded{into: intoID, micros: held.micros}) {
		return true
	}
	forgottenFolded.Inc()
	delete(m.holdOf, id)
	return false
}

// forgetFolded lets go of every folded id the enforcer remembers, as it stops.
func (m *money) forgetFolded() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, acct := range m.accounts {
		foldedInProcess.Add(-int64(len(acct.folded)))
		acct.folded = nil
	}
}

// shipped asks the shard which of the rows held against a balance with no ack it has shipped
// to the usage service, as another router's sweeper does without this one hearing of it, and
// notes them acked now, as acks whose counters are not known to have the rows. It runs on the
// limits loop, which is the one that reads Postgres, every SumEvery, at most shippedPerPass of
// them, the oldest first, within shippedTimeout.
//
// A shard that does not answer is a failed read of the balance for each account asked about:
// the row it may have shipped is not known to be counted, so the timer of each unacked hold
// starts again, as it does through an outage of the usage service.
func (m *money) shipped(ctx context.Context) {
	if m.settings.Usage == nil {
		return
	}
	db := m.enforcer.config.DB()
	if db == nil {
		return
	}
	type candidate struct {
		id   uuid.UUID
		key  string
		made time.Time
	}
	now := time.Now()
	var asked []candidate
	m.mu.Lock()
	for key, acct := range m.accounts {
		for id, held := range acct.holds {
			if held.group == nil && held.balance && held.ackedAt.IsZero() && now.Sub(held.made) > m.settings.Fresh {
				asked = append(asked, candidate{id: id, key: key, made: held.made})
			}
		}
	}
	m.mu.Unlock()
	slices.SortFunc(asked, func(a, b candidate) int { return a.made.Compare(b.made) })
	asked = asked[:min(len(asked), shippedPerPass)]
	for start := 0; start < len(asked); start += shippedChunk {
		part := asked[start:min(start+shippedChunk, len(asked))]
		ids := make([]uuid.UUID, len(part))
		for i, each := range part {
			ids[i] = each.id
		}
		call, cancel := context.WithTimeout(ctx, readTimeout)
		found, err := db.Shipped(call, ids)
		cancel()
		if err != nil {
			m.failedRead("could not ask the shard which held rows it has shipped, so their timers start again", err, "rows", len(asked))
			failed := time.Now()
			m.mu.Lock()
			for _, each := range asked[start:] {
				if acct := m.accounts[each.key]; acct != nil {
					acct.balanceFailed = failed
				}
			}
			m.mu.Unlock()
			return
		}
		m.settled(acks(found), true, func(h *hold) *time.Time { return &h.ackedAt }, func(f *fold) *bool { return &f.acked })
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
