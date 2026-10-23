package policy

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// How the limits loop reads one account's subjects' spend.
type spendSource int

const (
	// fromCounters is the usage service's Redis counters, every SpendEvery.
	fromCounters spendSource = iota
	// fromService is the usage service itself (POST /v1/windows), which answers from its
	// Postgres: every Fresh with no counters to read, and at once when an ack said the counters
	// missed a row.
	fromService
	// fromShards sums the request rows of every shard, every SumEvery, with no usage service.
	fromShards
)

// spendRead is what one pass read of one account's subjects: by counter key, then interval.
// intervals is, by counter key, the intervals each subject has a limit in, which are the only
// ones summed from the shards. A read of the subjects the counters could not answer has the
// read it is part of as its parent, which fails with it; countersFailed says the counters'
// Redis failed it.
type spendRead struct {
	source         spendSource
	refs           []usage.SubjectRef
	intervals      map[string][]usage.Interval
	from           time.Time
	spent          map[string]map[usage.Interval]spentWindow
	failed         bool
	countersFailed bool
	parent         *spendRead
}

func (r *spendRead) fail() {
	r.failed = true
	if r.parent != nil {
		r.parent.failed = true
	}
}

// limitsPass reads, for every account in use, its limits and the resets of its current windows
// every Fresh, and what each subject with a limit spent, as each source's cadence allows; then
// lets go of the holds the reads now count. Every SumEvery it also asks the shard which held
// rows it shipped. Only an account or subject never read is read on the request path
// (firstRead).
func (m *money) limitsPass(ctx context.Context) {
	now := time.Now()
	if now.Sub(m.shippedAt) >= m.settings.SumEvery {
		m.shippedAt = now
		shipping, cancel := context.WithTimeout(ctx, shippedTimeout)
		m.shipped(shipping)
		cancel()
	}
	ctx, cancel := context.WithTimeout(ctx, passTimeout)
	defer cancel()
	keys := m.inUse(time.Now())
	if len(keys) == 0 {
		return
	}
	db := m.enforcer.config.DB()
	if db == nil {
		return
	}
	m.readAccounts(ctx, db, keys, false)
}

// readAccounts reads the accounts' limits and resets when due, then what their limited subjects
// spent, every one at once when now is set and as each source's cadence allows otherwise, and
// keeps what landed. It answers the accounts a read of did not land.
func (m *money) readAccounts(ctx context.Context, db *store.Store, keys []string, now bool) map[string]bool {
	failed := map[string]bool{}
	if !m.readLimits(ctx, db, keys) {
		for _, key := range keys {
			failed[key] = true
		}
	}
	reads := m.plan(keys, time.Now(), now)
	if len(reads) == 0 {
		return failed
	}
	m.readSpend(ctx, db, reads)
	for key := range m.land(reads) {
		failed[key] = true
	}
	return failed
}

// land keeps what the reads found, under m.mu, and lets go of the holds they now count. A read
// asked before the one a subject last kept is older, and changes nothing of it. It answers the
// accounts whose read failed.
func (m *money) land(reads map[string]*spendRead) map[string]bool {
	failed := map[string]bool{}
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, read := range reads {
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		if read.countersFailed {
			acct.countersFailed = now
		}
		if read.failed {
			failed[key] = true
			acct.spendFailed = now
		} else {
			acct.spentFrom = later(acct.spentFrom, read.from)
			if read.source != fromCounters {
				acct.throughFrom = later(acct.throughFrom, read.from)
				acct.readThrough = false
			}
		}
		for _, ref := range read.refs {
			subject := acct.subjects[ref.Key(usage.CounterPrefix)]
			spent, ok := read.spent[ref.Key(usage.CounterPrefix)]
			if subject == nil || !ok || read.from.Before(subject.from) {
				continue
			}
			subject.spent, subject.read, subject.from = spent, now, read.from
		}
		m.release(key, acct, now)
	}
	return failed
}

// firstRead reads, before an admission, what the customer's limits need that this process never
// read: the account's limits and resets the first time it sees the account, and what each of the
// owner's subjects with a limit spent the first time it sees the subject, or again once Idle
// dropped it. It waits at most FirstRead; the read carries on in the background and is kept when
// it lands. Limits that have not landed by then admit; a limited subject's spend that has not is
// the tier's fail mode to answer (refusal). One such read is under way per account at a time,
// read in a batch with the other accounts first seen within firstWindow (queueFirst). Once an
// admission gave up on it, the next ones do not wait for it, and after one that did not land the
// next is asked only Fresh later, so an outage costs an admission at most one wait.
func (m *money) firstRead(ctx context.Context, key string, owner routing.Owner) {
	db := m.enforcer.config.DB()
	if db == nil || m.ctx == nil {
		return
	}
	m.want(key)
	deadline := time.Now().Add(m.settings.FirstRead)
	for {
		m.mu.Lock()
		acct := m.accounts[key]
		if acct == nil || m.stopping || !m.unread(acct, key, owner, time.Now()) {
			m.mu.Unlock()
			return
		}
		reading := acct.reading
		switch {
		case reading != nil && acct.readingLate:
			m.mu.Unlock()
			return
		case reading == nil && time.Since(acct.firstFailed) < m.settings.Fresh:
			m.mu.Unlock()
			return
		case reading == nil:
			reading = make(chan struct{})
			acct.reading = reading
			m.queueFirst(db, key)
		}
		m.mu.Unlock()
		timer := time.NewTimer(max(time.Until(deadline), 0))
		select {
		case <-reading:
			timer.Stop()
		case <-timer.C:
			m.mu.Lock()
			if acct.reading == reading {
				acct.readingLate = true
			}
			m.mu.Unlock()
			return
		case <-ctx.Done():
			timer.Stop()
			return
		}
	}
}

// unread reports, under m.mu, whether an admission of the owner's work needs a read that never
// landed: the account's limits, or what one of the owner's subjects with a limit spent. It notes
// the subjects it needs, so the read that follows reads them.
func (m *money) unread(acct *account, key string, owner routing.Owner, now time.Time) bool {
	if !acct.limitsRead.IsZero() && !acct.limited {
		return false
	}
	refs := subjectsOf(owner, limitsOrganization(key))
	if !acct.limitsRead.IsZero() && !slices.ContainsFunc(refs, func(ref usage.SubjectRef) bool {
		subject := acct.subjects[ref.Key(usage.CounterPrefix)]
		return (subject == nil || subject.read.IsZero()) && limited(acct.limits, ref)
	}) {
		return false
	}
	m.see(acct, refs, now)
	return true
}

// queueFirst adds an account to the batch of first reads gathering now, under m.mu, and starts
// one when none is. A batch gathers for firstWindow and at most firstBatches are read at once,
// so a cold start that sees thousands of accounts makes a few calls of each kind (LimitsFor,
// Resets, the service's windows) rather than a call of each per account.
func (m *money) queueFirst(db *store.Store, key string) {
	if m.firstQueue == nil {
		m.firstQueue = map[string]bool{}
	}
	m.firstQueue[key] = true
	if m.firstGathering {
		return
	}
	m.firstGathering = true
	m.reads.Add(1)
	go m.readFirsts(db)
}

// readFirsts reads one batch of first reads, and closes each account's reading once it is done.
func (m *money) readFirsts(db *store.Store) {
	defer m.reads.Done()
	select {
	case <-time.After(firstWindow):
	case <-m.ctx.Done():
	}
	m.mu.Lock()
	keys := slices.Sorted(mapsKeys(m.firstQueue))
	m.firstQueue, m.firstGathering = nil, false
	m.mu.Unlock()
	failed := map[string]bool{}
	select {
	case m.firstSlots <- struct{}{}:
		ctx, cancel := context.WithTimeout(m.ctx, readTimeout)
		failed = m.readAccounts(ctx, db, keys, true)
		cancel()
		<-m.firstSlots
	case <-m.ctx.Done():
		for _, key := range keys {
			failed[key] = true
		}
	}
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, key := range keys {
		acct := m.accounts[key]
		if acct == nil || acct.reading == nil {
			continue
		}
		if failed[key] {
			acct.firstFailed = now
		}
		close(acct.reading)
		acct.reading, acct.readingLate = nil, false
	}
}

// limitsStale reports, under m.mu, whether what the limits of the owner's work are checked
// against was last read MaxStale ago or never: the account's limits and resets, or what one of
// its subjects with a limit spent.
func (m *money) limitsStale(acct *account, refs []usage.SubjectRef, now time.Time) bool {
	if acct.limitsRead.IsZero() || now.Sub(acct.limitsRead) > m.settings.MaxStale {
		return true
	}
	for _, ref := range refs {
		if !limited(acct.limits, ref) {
			continue
		}
		subject := acct.subjects[ref.Key(usage.CounterPrefix)]
		if subject == nil || subject.read.IsZero() || now.Sub(subject.read) > m.settings.MaxStale {
			return true
		}
	}
	return false
}

// readLimits reads the limits and resets of the accounts whose were last read Fresh ago or
// never, in one query of each, and keeps them; a read that fails keeps the last ones. It says
// whether the read, when one was due, landed.
func (m *money) readLimits(ctx context.Context, db *store.Store, keys []string) bool {
	now := time.Now()
	due := map[string]string{}
	m.mu.Lock()
	for _, key := range keys {
		if acct := m.accounts[key]; acct != nil && now.Sub(acct.limitsRead) >= m.settings.Fresh {
			due[limitsOrganization(key)] = key
		}
	}
	m.mu.Unlock()
	if len(due) == 0 {
		return true
	}
	organizations := slices.Sorted(mapsKeys(due))
	call, cancel := context.WithTimeout(ctx, readTimeout)
	defer cancel()
	limits, err := db.LimitsFor(call, organizations)
	if err != nil {
		m.failedRead("could not read the customers' spend limits, answering with the last ones read", err, "organizations", len(organizations))
		return false
	}
	resets, err := db.Resets(call, organizations, earliestWindow(m.settings.Now()))
	if err != nil {
		m.failedRead("could not read the customers' usage resets, answering with the last ones read", err, "organizations", len(organizations))
		return false
	}
	byOrganization := map[string][]store.Limit{}
	for _, limit := range limits {
		byOrganization[limit.OrganizationID] = append(byOrganization[limit.OrganizationID], limit)
	}
	resetsOf := map[string][]store.UsageReset{}
	for _, reset := range resets {
		resetsOf[reset.OrganizationID] = append(resetsOf[reset.OrganizationID], reset)
	}
	landed := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for organization, key := range due {
		if acct := m.accounts[key]; acct != nil {
			m.keepLimits(acct, byOrganization[organization], resetsOf[organization], landed)
		}
	}
	return true
}

// grantsPass reads the live grants of every account with a limit, every Fresh, on a loop of its
// own: a slow or failed read of them never delays or skips the limits and resets, which their
// loop reads apart. With shards it first finds which held rows the reconcile counted.
func (m *money) grantsPass(ctx context.Context) {
	db := m.enforcer.config.DB()
	if db == nil {
		return
	}
	m.mu.Lock()
	var keys []string
	for key, acct := range m.accounts {
		if acct.limited {
			keys = append(keys, key)
		}
	}
	m.mu.Unlock()
	if len(keys) == 0 {
		return
	}
	slices.Sort(keys)
	m.countedByReconcile(ctx, db)
	m.readGrants(ctx, db, keys)
}

// readGrants reads the live grants of the accounts' organizations, within grantsTimeout, and
// keeps them under each account. A read that fails keeps the last ones read, and the holds
// against them, and leaves an account never read with no grant to draw.
func (m *money) readGrants(ctx context.Context, db *store.Store, keys []string) {
	organizations := make([]string, 0, len(keys))
	for _, key := range keys {
		organizations = append(organizations, limitsOrganization(key))
	}
	organizations = slices.Compact(slices.Sorted(slices.Values(organizations)))
	from := time.Now()
	call, cancel := context.WithTimeout(ctx, grantsTimeout)
	grants, err := db.LiveGrants(call, organizations, m.settings.Now())
	cancel()
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.failedRead("could not read the customers' grants, answering with the last ones read", err, "organizations", len(organizations))
		return
	}
	byOrganization := map[string][]store.LimitGrant{}
	for _, grant := range grants {
		byOrganization[grant.OrganizationID] = append(byOrganization[grant.OrganizationID], grant)
	}
	for _, key := range keys {
		if acct := m.accounts[key]; acct != nil {
			acct.keepGrants(byOrganization[limitsOrganization(key)], from)
		}
	}
}

// reconcilePass counts toward their grants the rows the shards took since the last reconcile
// that their writers did not, within reconcileTimeout, unless another router is at it. With one
// database the writer counts a row in the transaction that writes it, so there is nothing to
// reconcile. One cut short keeps how far it got, and the next goes on from there.
func (m *money) reconcilePass(ctx context.Context) {
	db := m.enforcer.config.DB()
	if db == nil || !db.Sharded() {
		return
	}
	call, cancel := context.WithTimeout(ctx, reconcileTimeout)
	counted, err := db.ReconcileGrantUses(call)
	cancel()
	if counted > 0 {
		m.enforcer.logger.Warn("counted usage rows toward their grants that their writer had not", "rows", counted)
	}
	if err != nil && !errors.Is(err, store.ErrReconcileBusy) && ctx.Err() == nil {
		m.enforcer.logger.Error("could not count the rows naming the customers' grants again", "error", err)
	}
}

// keepLimits replaces an account's limits and resets, under m.mu. An account that has a limit
// and whose grants were never read has them read at once.
func (m *money) keepLimits(acct *account, limits []store.Limit, resets []store.UsageReset, landed time.Time) {
	acct.limits, acct.resets, acct.limitsRead = indexLimits(limits), resets, landed
	acct.limited = len(limits) > 0
	if acct.limited && acct.grantsFrom.IsZero() && m.kickGrants != nil {
		nudge(m.kickGrants)
	}
	if acct.limited && m.kickLimits != nil {
		for _, subject := range acct.subjects {
			if subject.read.IsZero() && limited(acct.limits, subject.ref) {
				nudge(m.kickLimits)
				return
			}
		}
	}
}

// earliestWindow is when the earliest of the current windows began, which every reset that
// applies now is at or after.
func earliestWindow(now time.Time) time.Time {
	earliest := now
	for _, interval := range usage.Intervals {
		if start, _ := usage.Window(interval, now); start.Before(earliest) {
			earliest = start
		}
	}
	return earliest
}

// plan is, for every account with a limit, the subjects whose spend is due a read (every one
// when all is set) and where it is read from. It drops the subjects unused for Idle.
func (m *money) plan(keys []string, now time.Time, all bool) map[string]*spendRead {
	reads := map[string]*spendRead{}
	for _, key := range keys {
		if read := m.planAccount(key, now, all); read != nil {
			reads[key] = read
		}
	}
	return reads
}

// planAccount is plan for one account. m.mu is held only while the account's subjects are
// copied, one account at a time, and not while their limits are resolved, so an account with
// many subjects does not hold up the admissions of every other. The index of the limits is
// replaced whole when they are read again, never changed, so it is read outside the lock.
func (m *money) planAccount(key string, now time.Time, all bool) *spendRead {
	type seen struct {
		name   string
		ref    usage.SubjectRef
		unread bool
	}
	m.mu.Lock()
	acct := m.accounts[key]
	if acct == nil {
		m.mu.Unlock()
		return nil
	}
	for name, subject := range acct.subjects {
		if now.Sub(subject.used) > m.settings.Idle {
			delete(acct.subjects, name)
		}
	}
	for name, chosen := range acct.drawn {
		if now.Sub(chosen.at) > m.settings.Idle {
			delete(acct.drawn, name)
		}
	}
	if !acct.limited {
		m.mu.Unlock()
		return nil
	}
	source, every := fromShards, m.settings.SumEvery
	switch {
	case !m.counted(key):
	case acct.readThrough:
		source, every = fromService, 0
	case m.settings.Counters != nil && now.Sub(acct.countersFailed) >= m.settings.Fresh:
		source, every = fromCounters, m.settings.SpendEvery
	default:
		// No counters, or they failed within Fresh: the service, at its own pace.
		source, every = fromService, m.settings.Fresh
	}
	limits, last := acct.limits, later(acct.spentFrom, acct.spendFailed)
	subjects := make([]seen, 0, len(acct.subjects))
	for name, subject := range acct.subjects {
		subjects = append(subjects, seen{name: name, ref: subject.ref, unread: subject.read.IsZero()})
	}
	m.mu.Unlock()

	read := &spendRead{source: source, intervals: map[string][]usage.Interval{}}
	unread := false
	for _, subject := range subjects {
		for _, interval := range usage.Intervals {
			if _, ok := resolve(limits, subject.ref, interval); ok {
				read.intervals[subject.name] = append(read.intervals[subject.name], interval)
			}
		}
		if len(read.intervals[subject.name]) > 0 {
			read.refs = append(read.refs, subject.ref)
			unread = unread || subject.unread
		}
	}
	if len(read.refs) == 0 || (!all && !unread && now.Sub(last) < every) {
		return nil
	}
	slices.SortFunc(read.refs, func(a, b usage.SubjectRef) int { return cmp.Compare(a.Key(""), b.Key("")) })
	return read
}

// readSpend reads what each planned account's subjects spent in their current windows.
func (m *money) readSpend(ctx context.Context, db *store.Store, reads map[string]*spendRead) {
	var counters, service []*spendRead
	var sums []*spendRead
	for _, read := range reads {
		read.from = time.Now()
		read.spent = map[string]map[usage.Interval]spentWindow{}
		switch read.source {
		case fromCounters:
			counters = append(counters, read)
		case fromService:
			service = append(service, read)
		default:
			sums = append(sums, read)
		}
	}
	missing := m.readCounters(ctx, counters)
	service = append(service, missing...)
	m.readService(ctx, service)
	m.sumShards(ctx, db, sums)
}

// readCounters reads the subjects' current fields from the usage service's counters, every
// account's in one pipeline. A subject whose current field of any interval is missing (Redis was
// flushed, or it never spent) is read from the service instead, which writes the counters again;
// so is one the counters' Redis failed to answer within countersTimeout, and its account reads
// the service, not the counters, for the next Fresh. Those come back as reads of their own.
func (m *money) readCounters(ctx context.Context, reads []*spendRead) []*spendRead {
	if len(reads) == 0 {
		return nil
	}
	now := m.settings.Now()
	fields := make([]string, len(usage.Intervals))
	starts := make([]time.Time, len(usage.Intervals))
	for i, interval := range usage.Intervals {
		starts[i], _ = usage.Window(interval, now)
		fields[i] = usage.Field(interval, starts[i])
	}
	client := m.settings.Counters
	var commands rueidis.Commands
	for _, read := range reads {
		for _, ref := range read.refs {
			commands = append(commands, client.B().Hmget().Key(ref.Key(usage.CounterPrefix)).Field(fields...).Build())
		}
	}
	call, cancel := context.WithTimeout(ctx, countersTimeout)
	defer cancel()
	answers := client.DoMulti(call, commands...)
	var missing []*spendRead
	next := 0
	logged := false
	for _, read := range reads {
		through := &spendRead{source: fromService, from: read.from, spent: read.spent, parent: read}
		for _, ref := range read.refs {
			answer := answers[next]
			next++
			values, err := answer.ToArray()
			if err != nil {
				if !logged {
					logged = true
					m.failedRead("could not read the usage counters of a customer's limited subjects, reading the usage service instead", err)
				}
				read.countersFailed = true
				through.refs = append(through.refs, ref)
				continue
			}
			spent := map[usage.Interval]spentWindow{}
			whole := len(values) == len(fields)
			for i := 0; whole && i < len(values); i++ {
				text, err := values[i].ToString()
				if err != nil {
					whole = false
					break
				}
				micros, err := strconv.ParseInt(text, 10, 64)
				if err != nil {
					whole = false
					break
				}
				spent[usage.Intervals[i]] = spentWindow{start: starts[i], micros: micros}
			}
			if !whole {
				through.refs = append(through.refs, ref)
				continue
			}
			read.spent[ref.Key(usage.CounterPrefix)] = spent
		}
		if len(through.refs) > 0 {
			missing = append(missing, through)
		}
	}
	return missing
}

// readService reads the subjects' current windows from the usage service, which answers from
// its Postgres. The accounts are read together, in calls of whole accounts up to
// usage.MaxSubjects subjects, so a call that fails fails only the accounts it named.
func (m *money) readService(ctx context.Context, reads []*spendRead) {
	var batch []*spendRead
	subjects := 0
	for _, read := range reads {
		if len(read.refs) == 0 {
			continue
		}
		if len(batch) > 0 && subjects+len(read.refs) > usage.MaxSubjects {
			m.readServiceBatch(ctx, batch)
			batch, subjects = nil, 0
		}
		batch, subjects = append(batch, read), subjects+len(read.refs)
	}
	if len(batch) > 0 {
		m.readServiceBatch(ctx, batch)
	}
}

func (m *money) readServiceBatch(ctx context.Context, reads []*spendRead) {
	var refs []usage.SubjectRef
	whose := map[string]*spendRead{}
	for _, read := range reads {
		refs = append(refs, read.refs...)
		for _, ref := range read.refs {
			whose[ref.Key(usage.CounterPrefix)] = read
		}
	}
	call, cancel := context.WithTimeout(ctx, readTimeout)
	windows, err := m.settings.Usage.Windows(call, refs)
	cancel()
	if err != nil {
		for _, read := range reads {
			read.fail()
		}
		m.failedRead("could not read the usage windows of customers' limited subjects, answering with what was last read", err,
			"accounts", len(reads), "subjects", len(refs))
		return
	}
	for _, window := range windows {
		key := window.SubjectRef.Key(usage.CounterPrefix)
		read := whose[key]
		if read == nil {
			continue
		}
		if read.spent[key] == nil {
			read.spent[key] = map[usage.Interval]spentWindow{}
		}
		read.spent[key][window.Interval] = spentWindow{start: window.WindowStart, micros: window.SpentMicros}
	}
}

// sumShards sums each subject's billable spend in each interval it has a limit in from the
// request rows of every shard, sumReads at a time.
func (m *money) sumShards(ctx context.Context, db *store.Store, reads []*spendRead) {
	now := m.settings.Now()
	var mu sync.Mutex
	slots := make(chan struct{}, sumReads)
	var reading sync.WaitGroup
	for _, read := range reads {
		for _, ref := range read.refs {
			for _, interval := range read.intervals[ref.Key(usage.CounterPrefix)] {
				start, _ := usage.Window(interval, now)
				slots <- struct{}{}
				reading.Go(func() {
					defer func() { <-slots }()
					call, cancel := context.WithTimeout(ctx, readTimeout)
					spent, err := db.SubjectSpendSince(call, ref, start)
					cancel()
					mu.Lock()
					defer mu.Unlock()
					if err != nil {
						read.failed = true
						m.failedRead("could not sum what a customer's limited subject spent, answering with what was last read", err,
							"subject", ref.Subject)
						return
					}
					key := ref.Key(usage.CounterPrefix)
					if read.spent[key] == nil {
						read.spent[key] = map[usage.Interval]spentWindow{}
					}
					read.spent[key][interval] = spentWindow{start: start, micros: spent}
				})
			}
		}
	}
	reading.Wait()
}
