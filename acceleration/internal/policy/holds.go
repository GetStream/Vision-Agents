package policy

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
	// acklessAges is how many PendingTTLs a limits hold is kept at most while the usage service
	// acks none of the account's rows (ackless): an hour at the default.
	acklessAges = 6
	// uncountable is how long a grant hold is kept after the shard wrote its row with nothing
	// counting it: each reconcile reads a shard back store.GrantReconcileOverlap from where the
	// last got, so one that has not found the row two passes past that never will.
	uncountable = store.GrantReconcileOverlap + 2*reconcileEvery
)

// ReserveTTL is how long a reservation of request-and-answer work is kept unless its work is
// recorded or released first, and what a reservation with no ttl of its own is kept for.
const ReserveTTL = 2 * time.Minute

var _ routing.Spender = (*Enforcer)(nil)

// foldedInProcess is how many folded ids every account of every Enforcer remembers.
var foldedInProcess atomic.Int64

var forgottenFolded = promauto.NewCounter(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "admission_folded_ids_forgotten_total",
	Help:      "Usage ids folded past what an account or the process remembers, which are then held until their fold's timer rather than their ack",
})

// hold is money set aside for one unit of work, from before the work starts until the ledgers
// have counted what it cost. Keyed by the unit's usage id, so reserving, recording or acking it
// twice changes nothing. Reserved before the work starts (Reserve), it is unsettled: it holds
// the estimate, and expires at expires unless the work is recorded first. Recorded (Settle),
// it holds the cost, and the rules below let go of it. Every router shares its holds through
// Redis (shared.go), so each counts the others' as it counts its own.
//
// It counts twice over, and each is let go by its own ledger: the balance and the spend hard
// limit are the usage service's, read with the standings, and are let go once a standing read
// after the usage service had the row lands. The customer's limits count what each of the
// row's subjects spent: from the usage service's counters, let go once a read of them asked
// for after an ack that says the counters took the row lands (or a read of the service itself
// after any ack); with no usage service from the shard, let go once a sum asked for after the
// shard wrote the row lands. A row the ledger is never known to have is let go PendingTTL
// after the later of when it was made and when that ledger last could not be read, once a
// read after that lands; until one lands it is kept, however long. For the limits of an
// account the usage service counts, a stretch in which it acks none of the account's rows is
// a failed read too (ackless), so the limits' holds outlast it.
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
	// settled says the work was recorded and micros is its cost; until then micros is the
	// estimate it was reserved at, let go at expires if the work is never recorded.
	settled bool
	expires time.Time
	// grantsLetGo is when a count of the row let it go from its grants here, zero until then
	// and for one let go uncounted.
	grantsLetGo time.Time
	// balance and limits say whether it still counts against the balance and the spend hard
	// limit, and against the customer's limits; granted whether it still counts against the
	// grants it was drawn from, which is until a read of what they paid for, asked after the
	// writer counted the row toward them, lands.
	balance bool
	limits  bool
	granted bool
	grants  []int64
	// ackedAt is when this process knew the usage service had the row, and mirrored whether
	// that ack said its counters took it; writtenAt is when it knew the shard had it, and
	// countedAt when it knew the row counted toward its grants; zero until then.
	ackedAt   time.Time
	mirrored  bool
	writtenAt time.Time
	countedAt time.Time
	// group is, for a hold that carries others, the state they were folded in; nil for a row's own.
	group *fold
}

// fold is what holds are carried together by: one app and the same subjects, one UTC hour
// (every window a limit counts over starts on an hour), and the same state in each ledger.
type fold struct {
	app      string
	subjects string
	grants   string
	hour     int64
	balance  bool
	limits   bool
	granted  bool
	acked    bool
	mirrored bool
	written  bool
	counted  bool
}

// state is the fold the hold would be carried in.
func (h *hold) state() fold {
	return fold{
		app: h.app, subjects: h.subjectsKey, grants: grantsKey(h.grants), hour: h.at.UTC().Truncate(time.Hour).Unix(),
		balance: h.balance, limits: h.limits, granted: h.granted,
		acked: !h.ackedAt.IsZero(), mirrored: h.mirrored, written: !h.writtenAt.IsZero(), counted: !h.countedAt.IsZero(),
	}
}

// grantsKey is a hold's grants as a fold tells them apart.
func grantsKey(grants []int64) string {
	words := make([]string, len(grants))
	for i, id := range grants {
		words[i] = strconv.FormatInt(id, 10)
	}
	return strings.Join(words, ",")
}

// grantsOfKey is grantsKey read back.
func grantsOfKey(key string) []int64 {
	if key == "" {
		return nil
	}
	words := strings.Split(key, ",")
	grants := make([]int64, 0, len(words))
	for _, word := range words {
		if id, err := strconv.ParseInt(word, 10, 64); err == nil {
			grants = append(grants, id)
		}
	}
	return grants
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
	if h.granted {
		a.countGrants(h, sign)
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

// accountOf is the key of the account an owner's work is held against: its organization, or
// for an app in none, the app's own.
func (e *Enforcer) accountOf(owner routing.Owner) string {
	organization := owner.OrganizationID
	if decided, ok := e.decided(owner.CustomerID); ok && decided.organization != "" {
		organization = decided.organization
	}
	if organization == "" {
		if pending, ok := e.members.Load(owner.CustomerID); ok {
			organization, _ = pending.(string)
		}
	}
	if organization == "" {
		organization = appAccount(owner.CustomerID)
	}
	return organization
}

// Reserve sets micros aside for one unit of work before it starts, under the usage id its row
// will carry, for at most ttl unless the work is recorded first (Settle) or never reaches a
// vendor (Release). It answers from memory: the hold counts at once in this process and
// reaches the other routers with the next flush, and nothing waits on Redis. A nil answer is
// the admission.
//
// It checks what Admit checks for billable work, in the same order, counting every hold, the
// other routers' included: suspension, then whether holds could be shared, then credit, the
// spend hard limit, and the customer's limits on every subject the work counts for, a spent
// one passing on a live grant with room, which holds the work too. A refusal is the error Admit
// would give and holds nothing. An estimate larger than the room left is admitted while there
// is any room: the last unit may take the balance below zero by its estimate.
//
// Reserving again under the same id replaces the first reservation, which is how a lease is
// renewed, and its own amount is no reason to refuse it; one refused keeps the first until it
// expires. An id already recorded keeps its hold, and is only checked. In shadow a refusal is
// logged as would refuse and the work held anyway; off, nothing is checked or held.
func (e *Enforcer) Reserve(owner routing.Owner, usageID uuid.UUID, micros int64, ttl time.Duration) error {
	if e == nil || owner.CustomerID == "" || e.money.off() {
		return nil
	}
	if ttl <= 0 {
		ttl = ReserveTTL
	}
	organization := e.accountOf(owner)
	decided, _ := e.decided(owner.CustomerID)
	m := &e.money
	now, at := time.Now(), m.settings.Now()
	m.mu.Lock()
	acct, known := m.accounts[organization]
	if !known {
		acct = &account{wanted: now}
		m.accounts[organization] = acct
	}
	acct.used = now
	if acct.holds == nil {
		acct.holds = map[uuid.UUID]*hold{}
	}
	refs := subjectsOf(owner, limitsOrganization(organization))
	m.see(acct, refs, now)

	// A reservation made again stops counting while it is checked, so a lease is never refused
	// for its own last minute; one already recorded, or folded, is only checked.
	holding := usageID != uuid.Nil && micros > 0
	var previous *hold
	var previousIn *account
	previousKey := ""
	if key, ok := m.holdOf[usageID]; ok && holding {
		if other := m.accounts[key]; other != nil {
			held := other.holds[usageID]
			_, folded := other.folded[usageID]
			switch {
			case folded || (held != nil && held.settled):
				holding = false
			case held != nil:
				previous, previousIn, previousKey = held, other, key
				other.count(held, -1)
				delete(other.holds, usageID)
			}
		}
	}
	var grants []int64
	refusal, spend, reason, named := m.refusal(acct, organization, owner, decided, now, &grants)
	seen := seenOf(acct)
	refused := refusal != nil || spend != nil
	shadow := m.settings.Mode == config.LimitsShadow
	if refused && !shadow || !holding {
		if previous != nil {
			previousIn.holds[usageID] = previous
			previousIn.count(previous, 1)
		}
		m.mu.Unlock()
		if !known {
			m.wake()
		}
		if refused {
			holdsRefused.WithLabelValues(reason).Inc()
		}
		refusal, spend = e.answer(organization, owner.CustomerID, refusal, spend, reason, named, seen, now)
		return errors.Join(refusal, spend)
	}
	held := &hold{
		app: owner.CustomerID, micros: micros, made: now, at: at, expires: now.Add(ttl),
		balance: m.billing(organization), limits: true, grants: grants, granted: len(grants) > 0,
	}
	held.subjects = make([]string, len(refs))
	for i, ref := range refs {
		held.subjects[i] = ref.Key(usage.CounterPrefix)
	}
	held.subjectsKey = strings.Join(held.subjects, subjectsSeparator)
	if previous != nil && previousKey != organization {
		m.touch(previousKey, usageID, now)
	}
	acct.holds[usageID] = held
	acct.count(held, 1)
	m.holdOf[usageID] = organization
	m.touch(organization, usageID, now)
	m.mu.Unlock()
	if !known {
		m.wake()
	}
	if refused {
		e.answer(organization, owner.CustomerID, refusal, spend, reason, named, seen, now)
	}
	return nil
}

// Release lets go of a reservation whose work never reached a vendor, so it bills nothing. A
// hold the work was recorded under is kept: its row is on its way to the ledger.
func (e *Enforcer) Release(usageID uuid.UUID) {
	if e == nil || usageID == uuid.Nil || e.money.off() {
		return
	}
	m := &e.money
	m.mu.Lock()
	defer m.mu.Unlock()
	key, ok := m.holdOf[usageID]
	if !ok {
		return
	}
	acct := m.accounts[key]
	if acct == nil {
		return
	}
	held := acct.holds[usageID]
	if held == nil || held.settled {
		return
	}
	acct.count(held, -1)
	delete(acct.holds, usageID)
	delete(m.holdOf, usageID)
	m.touch(key, usageID, time.Now())
}

// Settle sets the hold of a unit of work recorded here to what it cost, against its owner's
// organization and every subject it counts for, under the row's usage id, and keeps it until
// the ledgers count the row. A unit reserved before it started keeps the grants it was
// admitted on; one never reserved is held now, so nothing recorded goes uncounted. Settling it
// again replaces the amount, restarts its timer and leaves what the ledgers were known to have
// as it was; settling one carried in a fold again changes nothing. Settling at nothing lets go
// of a reservation, as free work holds nothing. It never waits.
//
// It answers the grants the work is drawn from, past limits of its subjects that are spent, as
// its admission was: the row names them, and they hold it until a read of what they paid for
// counts the row. Settled again, it answers the grants it was first held on.
func (e *Enforcer) Settle(owner routing.Owner, usageID uuid.UUID, micros int64) []int64 {
	if e == nil || owner.CustomerID == "" || usageID == uuid.Nil || e.money.off() {
		return nil
	}
	if micros <= 0 {
		e.Release(usageID)
		return nil
	}
	organization := e.accountOf(owner)
	m := &e.money
	now, at := time.Now(), m.settings.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	var held *hold
	if previous, ok := m.holdOf[usageID]; ok {
		if other := m.accounts[previous]; other != nil {
			if carried, ok := other.folded[usageID]; ok {
				if into := other.holds[carried.into]; into != nil {
					return slices.Clone(into.grants)
				}
				return nil
			}
			if held = other.holds[usageID]; held != nil {
				other.count(held, -1)
				delete(other.holds, usageID)
				if previous != organization {
					m.touch(previous, usageID, now)
				}
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
	refs := subjectsOf(owner, limitsOrganization(organization))
	if held == nil {
		held = &hold{at: at, balance: m.billing(organization), limits: true}
		if acct.limited && len(acct.grants) > 0 {
			held.grants = acct.drawnFor(refs, at)
			held.granted = len(held.grants) > 0
		}
	}
	// Its subjects are read with the next spend read once the account has a limit on one,
	// which is what lets go of it.
	m.see(acct, refs, now)
	held.subjects = make([]string, len(refs))
	for i, ref := range refs {
		held.subjects[i] = ref.Key(usage.CounterPrefix)
	}
	held.subjectsKey = strings.Join(held.subjects, subjectsSeparator)
	held.app, held.micros, held.made = owner.CustomerID, micros, now
	held.settled, held.expires = true, time.Time{}
	acct.holds[usageID] = held
	acct.count(held, 1)
	m.holdOf[usageID] = organization
	m.touch(organization, usageID, now)
	return slices.Clone(held.grants)
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

// Counted notes the rows that count toward their grants in the shared database: a hold kept for
// one of them stops counting against its grants once a read of them asked for after now lands.
// It is WriterOptions.Counted.
func (e *Enforcer) Counted(counted []routing.Acked) {
	if e == nil || e.money.off() {
		return
	}
	e.money.settled(counted, false, func(h *hold) *time.Time { return &h.countedAt }, func(f *fold) *bool { return &f.counted })
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
		if acks {
			acct.lastAck = now
			if !row.Mirrored {
				acct.readThrough = true
			}
		}
		if held := acct.holds[row.UsageID]; held != nil {
			if at := when(held); at.IsZero() {
				*at = now
				if acks {
					held.mirrored = row.Mirrored
				}
				m.touch(key, row.UsageID, now)
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
		into.countedAt = later(into.countedAt, from.countedAt)
		// The fold this joins may have been acked before: it is let go only by a read asked for
		// after this ack too, so what it already held waits with what just joined it.
		*when(into) = now
		acct.folded[row.UsageID] = folded{into: intoID, micros: carried.micros}
		if from.micros == 0 {
			acct.drop(carried.into, from)
		}
		m.touch(key, carried.into, now)
		m.touch(key, intoID, now)
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
	if counted && acct.ackless(now, m.settings.Fresh) {
		acct.acksFailed = now
	}
	var expired, unwritten, waiting, waited, unacked, aged, uncounted, stranded int64
	var emptied []uuid.UUID
	var strandedGrants []int64
	for id, held := range acct.holds {
		if !held.settled {
			// Reserved and never recorded: it goes at its expiry, and with it the work's claim.
			if now.Before(held.expires) {
				continue
			}
			acct.count(held, -1)
			held.balance, held.limits, held.granted = false, false, false
			delete(acct.holds, id)
			delete(m.holdOf, id)
			m.touch(key, id, now)
			continue
		}
		was := [3]bool{held.balance, held.limits, held.granted}
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
			case !acct.spentFrom.Before(later(held.made, later(acct.spendFailed, acct.acksFailed)).Add(ttl)):
				unwritten += held.micros
				acct.letGoLimits(held)
			case counted && !acked && now.Sub(held.made) > acklessAges*ttl:
				// However long the service acks nothing, a hold is kept at most this long.
				aged += held.micros
				acct.letGoLimits(held)
			case counted && !acked && now.Sub(held.made) > ttl && acct.acksFailed.Equal(now):
				unacked += held.micros
			}
		}
		if held.granted {
			// Only a count of the row lets it go, by the grant counter or found by
			// countedByReconcile: a row whose count the grant counter still tries may count
			// yet, and letting go of it first would overdraw the grant. A row this process
			// never knew the shard to have goes acklessAges PendingTTLs on, as the writer may
			// have lost it, and one it knew written but nothing counted goes uncountable after.
			switch {
			case !held.countedAt.IsZero() && held.countedAt.Before(acct.grantsFrom):
				// This router's read has the count; the others' may not for a while, so the
				// shared copy outlives it (grantSharedUntil).
				acct.letGoGrants(held)
				held.grantsLetGo = now
			case held.writtenAt.IsZero() && now.Sub(held.made) > acklessAges*ttl:
				uncounted += held.micros
				acct.letGoGrants(held)
			case held.countedAt.IsZero() && !held.writtenAt.IsZero() && now.Sub(held.writtenAt) > uncountable:
				// The reconcile misses a row written past its overlap after it started, and
				// AddGrantUses refuses one naming a grant long ended: neither ever counts.
				stranded += held.micros
				strandedGrants = append(strandedGrants, held.grants...)
				acct.letGoGrants(held)
			}
		}
		if was != [3]bool{held.balance, held.limits, held.granted} {
			m.touch(key, id, now)
		}
		// One let go of here still counts on the other routers until their reads have its row,
		// so it is kept, counting nothing here, for as long as it is shared.
		if held.balance || held.limits || held.granted || m.sharing(held, now) {
			continue
		}
		m.touch(key, id, now)
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
	if uncounted > 0 {
		logger.Warn("usage held against an "+noun(key)+"'s grants is let go after "+(acklessAges*ttl).String()+
			" without this router knowing the shard wrote it", append(who(key), "micros", uncounted)...)
	}
	if stranded > 0 {
		logger.Error("usage held against an "+noun(key)+"'s grants is let go "+uncountable.String()+
			" after the shard wrote it, as nothing counted it toward them",
			append(who(key), "grants", slices.Compact(slices.Sorted(slices.Values(strandedGrants))), "micros", stranded)...)
	}
	if aged > 0 {
		logger.Warn("usage held against an "+noun(key)+"'s spend limits is let go after "+(acklessAges*ttl).String()+
			" with the usage service acking none of its rows", append(who(key), "micros", aged, "last_ack", acct.lastAck)...)
	}
	if unacked > 0 && m.waiting.due(key+"\x00acks", logEvery) {
		logger.Warn("usage held against an "+noun(key)+"'s spend limits is past its pending_ttl while the usage service acks none of its rows, so it is kept",
			append(who(key), "micros", unacked, "last_ack", acct.lastAck)...)
	}
	if len(acct.holds) > maxHolds {
		m.carry(key, acct)
	}
}

// ackless reports whether the usage service is acking nothing of the account: it holds against
// the limits a row made Fresh ago or more that is not acked, and no row of the account was acked
// since that one was made. The counters cannot have such a row, so release takes that as a
// failed read of the limits, as long as it lasts, and their holds do not expire through it.
func (a *account) ackless(now time.Time, fresh time.Duration) bool {
	for _, held := range a.holds {
		if held.settled && held.limits && held.ackedAt.IsZero() && now.Sub(held.made) >= fresh && !a.lastAck.After(held.made) {
			return true
		}
	}
	return false
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
		app: state.app, subjectsKey: state.subjects, at: time.Unix(state.hour, 0).UTC(), settled: true,
		balance: state.balance, limits: state.limits, granted: state.granted, grants: grantsOfKey(state.grants),
		mirrored: state.mirrored, group: &group,
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
		// A reservation expires on its own, which a fold could not keep.
		if held.group == nil && held.settled {
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
	if key, ok := m.holdOf[id]; ok {
		now := time.Now()
		m.touch(key, id, now)
		m.touch(key, intoID, now)
	}
	into.micros += held.micros
	into.made = later(into.made, held.made)
	into.ackedAt, into.writtenAt = later(into.ackedAt, held.ackedAt), later(into.writtenAt, held.writtenAt)
	into.countedAt = later(into.countedAt, held.countedAt)
	into.grantsLetGo = later(into.grantsLetGo, held.grantsLetGo)
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
			if held.group == nil && held.settled && held.balance && held.ackedAt.IsZero() && now.Sub(held.made) > m.settings.Fresh {
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

// countedByReconcile asks the shared database which of the rows held against their grants,
// written Fresh ago or more and not known counted, count toward them, as the reconcile of any
// router counts a row the grant counter dropped or the shard already had, and notes them
// counted now. It runs on the grants loop before the grants are read, at most shippedPerPass
// of them, the oldest first, within grantsTimeout; with one database the writer counts every
// row as it writes it, so there is nothing to ask. A failed question keeps the holds.
func (m *money) countedByReconcile(ctx context.Context, db *store.Store) {
	if !db.Sharded() {
		return
	}
	type candidate struct {
		id     uuid.UUID
		grants int
		made   time.Time
	}
	now := time.Now()
	var asked []candidate
	due := func(held *hold) bool {
		return held != nil && held.granted && held.countedAt.IsZero() && !held.writtenAt.IsZero() && now.Sub(held.writtenAt) > m.settings.Fresh
	}
	m.mu.Lock()
	for _, acct := range m.accounts {
		for id, held := range acct.holds {
			if held.group == nil && due(held) {
				asked = append(asked, candidate{id: id, grants: len(held.grants), made: held.made})
			}
		}
		for id, carried := range acct.folded {
			if into := acct.holds[carried.into]; due(into) {
				asked = append(asked, candidate{id: id, grants: len(into.grants), made: into.made})
			}
		}
	}
	m.mu.Unlock()
	if len(asked) == 0 {
		return
	}
	slices.SortFunc(asked, func(a, b candidate) int { return a.made.Compare(b.made) })
	asked = asked[:min(len(asked), shippedPerPass)]
	call, cancel := context.WithTimeout(ctx, grantsTimeout)
	defer cancel()
	for start := 0; start < len(asked); start += shippedChunk {
		part := asked[start:min(start+shippedChunk, len(asked))]
		ids := make([]uuid.UUID, len(part))
		for i, each := range part {
			ids[i] = each.id
		}
		found, err := db.GrantUsesCounted(call, ids)
		if err != nil {
			m.failedRead("could not ask which rows held against their grants count toward them, so they are kept", err, "rows", len(asked))
			return
		}
		var counted []uuid.UUID
		for _, each := range part {
			// A row counts once it counts toward each grant it names.
			if found[each.id] >= each.grants {
				counted = append(counted, each.id)
			}
		}
		m.settled(acks(counted), false, func(h *hold) *time.Time { return &h.countedAt }, func(f *fold) *bool { return &f.counted })
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
