package policy

import (
	"cmp"
	"context"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/redis/rueidis"

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
// ones summed from the shards. A read of the subjects the counters were missing has the read
// it is part of as its parent, which fails with it.
type spendRead struct {
	source    spendSource
	refs      []usage.SubjectRef
	intervals map[string][]usage.Interval
	from      time.Time
	spent     map[string]map[usage.Interval]spentWindow
	failed    bool
	parent    *spendRead
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
// rows it shipped. Nothing here is ever asked on the request path.
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
	m.readLimits(ctx, db, keys)
	reads := m.plan(keys, time.Now())
	if len(reads) == 0 {
		return
	}
	m.readSpend(ctx, db, reads)

	now = time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, read := range reads {
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		if read.failed {
			acct.spendFailed = now
		} else {
			acct.spentFrom = read.from
			if read.source != fromCounters {
				acct.throughFrom = read.from
				acct.readThrough = false
			}
		}
		for _, ref := range read.refs {
			subject := acct.subjects[ref.Key(usage.CounterPrefix)]
			spent, ok := read.spent[ref.Key(usage.CounterPrefix)]
			if subject == nil || !ok {
				continue
			}
			subject.spent, subject.read = spent, now
		}
		m.release(key, acct, now)
	}
}

// readLimits reads the limits and resets of the accounts whose were last read Fresh ago or
// never, in one query of each, and keeps them; a read that fails keeps the last ones.
func (m *money) readLimits(ctx context.Context, db *store.Store, keys []string) {
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
		return
	}
	organizations := slices.Sorted(mapsKeys(due))
	call, cancel := context.WithTimeout(ctx, readTimeout)
	defer cancel()
	limits, err := db.LimitsFor(call, organizations)
	if err != nil {
		m.failedRead("could not read the customers' spend limits, answering with the last ones read", err, "organizations", len(organizations))
		return
	}
	resets, err := db.Resets(call, organizations, earliestWindow(m.settings.Now()))
	if err != nil {
		m.failedRead("could not read the customers' usage resets, answering with the last ones read", err, "organizations", len(organizations))
		return
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
}

// grantsPass reads the live grants of every account with a limit, every Fresh, on a loop of its
// own: a slow or failed read of them never delays or skips the limits and resets, which their
// loop reads apart. Every reconcileEvery it also counts again, from the shards, the rows naming
// each live grant, for a row a writer stopped before counting.
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
	m.readGrants(ctx, db, keys)
	m.reconcileGrants(ctx, db)
}

// readGrants reads the live grants of the accounts' organizations, within grantsTimeout, and
// keeps them under each account. A read that fails keeps the last ones read, is a failed read
// for the holds against them, and leaves an account never read with no grant to draw.
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
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		for _, key := range keys {
			if acct := m.accounts[key]; acct != nil {
				acct.grantsFailed = now
			}
		}
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

// reconcileGrants counts again, from the shards, the rows naming each live grant last counted
// reconcileEvery ago or never by this process. With one database the writer counts a row in the
// transaction that writes it, so there is nothing to reconcile.
func (m *money) reconcileGrants(ctx context.Context, db *store.Store) {
	if !db.Sharded() {
		return
	}
	now := time.Now()
	live := map[int64]bool{}
	var due []int64
	m.mu.Lock()
	for _, acct := range m.accounts {
		for _, grant := range acct.grants {
			id := grant.grant.ID
			live[id] = true
			if now.Sub(m.reconciled[id]) >= reconcileEvery {
				due = append(due, id)
			}
		}
	}
	for id := range m.reconciled {
		if !live[id] {
			delete(m.reconciled, id)
		}
	}
	m.mu.Unlock()
	if len(due) == 0 {
		return
	}
	slices.Sort(due)
	call, cancel := context.WithTimeout(ctx, readTimeout)
	counted, err := db.ReconcileGrantUses(call, slices.Compact(due))
	cancel()
	if err != nil {
		m.failedRead("could not count the rows naming the customers' grants again", err, "grants", len(due))
		return
	}
	if counted > 0 {
		m.enforcer.logger.Warn("counted usage rows toward their grants that their writer had not", "rows", counted)
	}
	m.mu.Lock()
	for _, id := range due {
		m.reconciled[id] = now
	}
	m.mu.Unlock()
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

// plan is, for every account with a limit, the subjects whose spend is due a read and where it
// is read from. It drops, under m.mu, the subjects unused for Idle.
func (m *money) plan(keys []string, now time.Time) map[string]*spendRead {
	reads := map[string]*spendRead{}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, key := range keys {
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		for name, subject := range acct.subjects {
			if now.Sub(subject.used) > m.settings.Idle {
				delete(acct.subjects, name)
			}
		}
		if !acct.limited {
			continue
		}
		source, every := fromShards, m.settings.SumEvery
		switch {
		case !m.counted(key):
		case acct.readThrough:
			source, every = fromService, 0
		case m.settings.Counters != nil:
			source, every = fromCounters, m.settings.SpendEvery
		default:
			source, every = fromService, m.settings.Fresh
		}
		read := &spendRead{source: source, intervals: map[string][]usage.Interval{}}
		unread := false
		for name, subject := range acct.subjects {
			for _, interval := range usage.Intervals {
				if _, ok := resolve(acct.limits, subject.ref, interval); ok {
					read.intervals[name] = append(read.intervals[name], interval)
				}
			}
			if len(read.intervals[name]) > 0 {
				read.refs = append(read.refs, subject.ref)
				unread = unread || subject.read.IsZero()
			}
		}
		last := later(acct.spentFrom, acct.spendFailed)
		if len(read.refs) == 0 || (!unread && now.Sub(last) < every) {
			continue
		}
		slices.SortFunc(read.refs, func(a, b usage.SubjectRef) int { return cmp.Compare(a.Key(""), b.Key("")) })
		reads[key] = read
	}
	return reads
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

// readCounters reads the subjects' current fields from the usage service's counters, in one
// pipeline. A subject whose current field of any interval is missing (Redis was flushed, or it
// never spent) is read from the service instead, which writes the counters again; those come
// back as reads of their own.
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
	call, cancel := context.WithTimeout(ctx, readTimeout)
	defer cancel()
	answers := client.DoMulti(call, commands...)
	var missing []*spendRead
	next := 0
	for _, read := range reads {
		through := &spendRead{source: fromService, from: read.from, spent: read.spent, parent: read}
		for _, ref := range read.refs {
			answer := answers[next]
			next++
			values, err := answer.ToArray()
			if err != nil {
				if !read.failed {
					m.failedRead("could not read the usage counters of a customer's limited subjects, answering with what was last read", err)
				}
				read.fail()
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
// its Postgres.
func (m *money) readService(ctx context.Context, reads []*spendRead) {
	for _, read := range reads {
		if len(read.refs) == 0 {
			continue
		}
		call, cancel := context.WithTimeout(ctx, readTimeout)
		windows, err := m.settings.Usage.Windows(call, read.refs)
		cancel()
		if err != nil {
			read.fail()
			m.failedRead("could not read the usage windows of a customer's limited subjects, answering with what was last read", err, "subjects", len(read.refs))
			continue
		}
		for _, window := range windows {
			key := window.SubjectRef.Key(usage.CounterPrefix)
			if read.spent[key] == nil {
				read.spent[key] = map[usage.Interval]spentWindow{}
			}
			read.spent[key][window.Interval] = spentWindow{start: window.WindowStart, micros: window.SpentMicros}
		}
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
