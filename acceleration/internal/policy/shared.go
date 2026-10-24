package policy

import (
	"context"
	_ "embed"
	"errors"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// ErrSpendCheckUnavailable is billable work refused because what this router holds could not
// be shared with the other routers for Settings.FlushMaxLag: admitting more would let an
// organization spend on every router at once unseen. Work that costs nothing is still admitted.
var ErrSpendCheckUnavailable = errors.New("policy: could not share spend with the other routers; retry shortly")

// errUnshared is the refusal of billable work while holds go unshared.
var errUnshared = refusedWith(ErrSpendCheckUnavailable, "could not share spend with the other routers; retry shortly")

// holdPrefix starts the key of a subject's shared holds, in the same cluster slot as its
// counters: hold:{<organization>}:<subject>:<ids>.
const holdPrefix = "hold"

// flushTimeout bounds one round trip of the flush, and of the refresh, each of at most
// flushChunk keys, so a router in use by many organizations never sends one unbounded pipeline.
const (
	flushTimeout = time.Second
	flushChunk   = 256
)

// reasonUnshared is the refusal of billable work while holds cannot be shared.
const reasonUnshared = "holds_unshared"

//go:embed holds.lua
var holdsSource string

var holdsScript = rueidis.NewLuaScript(holdsSource)

var (
	holdsFlushed = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "holds_flushed_total",
		Help:      "Changes to holds (made, settled, let go) written to the Redis every router shares them through",
	})
	holdsFlushErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "holds_flush_errors_total",
		Help:      "Writes of holds to the shared Redis that failed, which are tried again on the next flush",
	})
	holdsRefused = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "holds_refused_total",
		Help:      "Reservations refused before the work started, by why",
	}, []string{"reason"})
	holdsPerSubject = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "holds_per_subject",
		Help:      "Live holds in one subject's shared hash, as a flush or refresh found it",
		Buckets:   []float64{1, 10, 100, 500, 1_000, 2_500, 5_000, 10_000, 50_000},
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "holds_unflushed_oldest_seconds",
		Help:      "How long the oldest change to a hold has waited to be written to the shared Redis; past 10 s billable work is refused",
	}, unflushedOldest)
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "holds_live",
		Help:      "Holds this process keeps, made before work started or recorded after it, until the ledgers count them",
	}, func() float64 {
		var live int
		enforcers.Range(func(key, _ any) bool {
			live += key.(*Enforcer).money.live()
			return true
		})
		return float64(live)
	})
)

func init() { routing.ReportUnflushedHolds(unflushedOldest) }

// unflushedOldest is how long the oldest change to a hold of any Enforcer has waited to be
// shared, in seconds.
func unflushedOldest() float64 {
	var oldest time.Duration
	enforcers.Range(func(key, _ any) bool {
		oldest = max(oldest, key.(*Enforcer).money.unflushedFor(time.Now()))
		return true
	})
	return oldest.Seconds()
}

// dirtyHold is a hold changed since it was last written to the shared Redis: the account it is
// kept under, when it first changed unwritten, and which change this is.
type dirtyHold struct {
	key     string
	since   time.Time
	version uint64
}

// touch notes, under m.mu, that the hold of id kept under key changed, so the next flush writes
// it. With no shared Redis there is nothing to write.
func (m *money) touch(key string, id uuid.UUID, now time.Time) {
	if m.settings.Holds == nil {
		return
	}
	if m.dirty == nil {
		m.dirty = map[uuid.UUID]dirtyHold{}
	}
	d, ok := m.dirty[id]
	if !ok {
		d.since = now
		if m.unflushedSince.IsZero() {
			m.unflushedSince = now
		}
	}
	m.version++
	d.key, d.version = key, m.version
	m.dirty[id] = d
}

// unshared reports, under m.mu, whether a change to a hold has waited longer than FlushMaxLag to
// be shared, so billable work is refused until a flush lands.
func (m *money) unshared(now time.Time) bool {
	return m.settings.Holds != nil && !m.unflushedSince.IsZero() && now.Sub(m.unflushedSince) > m.settings.FlushMaxLag
}

func (m *money) unflushedFor(now time.Time) time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.unflushedSince.IsZero() {
		return 0
	}
	return now.Sub(m.unflushedSince)
}

func (m *money) live() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	var live int
	for _, acct := range m.accounts {
		live += len(acct.holds)
	}
	return live
}

// sharedUntil is when the other routers stop counting a hold: an unsettled one at its own
// expiry; a settled one 2 × Fresh after the usage service acked it, by when every router's
// standing and counter reads have the row (with no usage service, 2 × SumEvery after the shard
// wrote it); and one never known to reach the ledger PendingTTL after it was recorded.
func (m *money) sharedUntil(h *hold) time.Time {
	switch {
	case !h.settled:
		return h.expires
	case !h.ackedAt.IsZero():
		return h.ackedAt.Add(2 * m.settings.Fresh)
	case m.settings.Usage == nil && !h.writtenAt.IsZero():
		return h.writtenAt.Add(2 * m.settings.SumEvery)
	}
	return h.made.Add(m.settings.PendingTTL)
}

// grantSharedUntil is when the other routers stop counting a hold against its grants: while it
// counts here, as long as this router may keep it (release); once a count of its row let it go
// here, 2 × Fresh later, by when every router's read of the grants has the count; one let go
// uncounted (stranded, or never known written) at once.
func (m *money) grantSharedUntil(h *hold) time.Time {
	switch {
	case !h.settled:
		return h.expires
	case h.granted:
		return later(h.made.Add(acklessAges*m.settings.PendingTTL), h.writtenAt.Add(uncountable))
	case !h.grantsLetGo.IsZero():
		return h.grantsLetGo.Add(2 * m.settings.Fresh)
	}
	return time.Time{}
}

// sharing reports, under m.mu, whether a hold that counts nothing here any more is still shared
// with the other routers, and so kept until it is not.
func (m *money) sharing(h *hold, now time.Time) bool {
	if m.settings.Holds == nil || !h.settled || h.micros <= 0 {
		return false
	}
	return m.sharedUntil(h).After(now) || len(h.grants) > 0 && m.grantSharedUntil(h).After(now)
}

// holdKey is the key of a subject's shared holds, from the key of its counters.
func holdKey(counter string) string {
	return holdPrefix + strings.TrimPrefix(counter, usage.CounterPrefix)
}

// grantKey is the key of a grant's shared holds, in its organization's slot.
func grantKey(organization string, grant int64) string {
	return holdPrefix + ":{" + organization + "}:grant:" + strconv.FormatInt(grant, 10)
}

// shareOf is what the shared Redis should hold of one hold, by key: nothing for one that is gone
// or no longer shared.
func (m *money) shareOf(key string, h *hold, now time.Time) map[string]string {
	if h == nil || h.micros <= 0 {
		return nil
	}
	out := map[string]string{}
	if until := m.sharedUntil(h); until.After(now) {
		value := shareValue(h.micros, until, now)
		for _, subject := range h.subjects {
			out[holdKey(subject)] = value
		}
	}
	if until := m.grantSharedUntil(h); len(h.grants) > 0 && until.After(now) {
		value := shareValue(h.micros, until, now)
		organization := limitsOrganization(key)
		for _, id := range h.grants {
			out[grantKey(organization, id)] = value
		}
	}
	return out
}

// shareValue is a hold as the script takes it: its micros and how long from now it lives, which
// the script turns into an expiry by Redis's own clock, so no router's clock skews it.
func shareValue(micros int64, until, now time.Time) string {
	return strconv.FormatInt(micros, 10) + ":" + strconv.FormatInt(max(until.Sub(now).Milliseconds(), 1), 10)
}

// remoteSlot is what one shared key's total is, in an account: the other routers' holds
// against a subject (by its counter's key) or against a grant.
type remoteSlot struct {
	account string
	counter string
	grant   int64
}

// keyPlan is one call of the script: the changes for one key, and what its answer is.
type keyPlan struct {
	slot remoteSlot
	args []string
	// ids are the holds that change here, which are written once every key they change in is.
	ids []uuid.UUID
}

// flushing is one dirty hold as a flush took it: which change, and every key it changes.
type flushing struct {
	version uint64
	keys    []string
	desired []string
}

// flushPass writes every hold changed since the last flush to the shared Redis, one script call
// per key pipelined in one round trip, and every HoldRefresh reads the keys in use that had
// nothing to write. Each call answers what the other routers hold under the key, which is what
// admission counts as theirs from then on. A flush that fails keeps the holds to write again,
// and admission goes on from memory until they have waited FlushMaxLag.
func (m *money) flushPass(ctx context.Context) {
	now := time.Now()
	plans, taken := m.planFlush(now)
	if len(plans) == 0 {
		return
	}
	keys := make([]string, 0, len(plans))
	for key := range plans {
		keys = append(keys, key)
	}
	failed := map[string]bool{}
	answers := map[string]int64{}
	var firstErr error
	for start := 0; start < len(keys); start += flushChunk {
		chunk := keys[start:min(start+flushChunk, len(keys))]
		execs := make([]rueidis.LuaExec, len(chunk))
		for i, key := range chunk {
			execs[i] = rueidis.LuaExec{Keys: []string{key}, Args: append([]string{m.instance}, plans[key].args...)}
		}
		call, cancel := context.WithTimeout(ctx, flushTimeout)
		results := holdsScript.ExecMulti(call, m.settings.Holds, execs...)
		cancel()
		for i, key := range chunk {
			var values []int64
			err := errors.New("policy: the holds script gave no answer")
			if i < len(results) {
				values, err = results[i].AsIntSlice()
			}
			if err == nil && len(values) != 2 {
				err = errors.New("policy: the holds script answered " + strconv.Itoa(len(values)) + " values")
			}
			if err != nil {
				failed[key] = true
				if firstErr == nil {
					firstErr = err
				}
				continue
			}
			answers[key] = values[0]
			holdsPerSubject.Observe(float64(values[1]))
		}
	}
	m.landFlush(plans, taken, answers, failed, time.Now())
	if firstErr != nil {
		holdsFlushErrors.Inc()
		if m.flushLogged.due(logEvery) {
			m.enforcer.logger.Error("could not share holds with the other routers, admitting from memory",
				"keys", len(failed), "unflushed_for", m.unflushedFor(time.Now()).Round(time.Millisecond),
				"max_lag", m.settings.FlushMaxLag, "error", firstErr)
		}
	}
}

// planFlush takes, under m.mu, the changes of every dirty hold by key, and when a refresh is due
// a call with no change for each key in use that has none.
func (m *money) planFlush(now time.Time) (map[string]*keyPlan, map[uuid.UUID]flushing) {
	m.mu.Lock()
	defer m.mu.Unlock()
	plans := map[string]*keyPlan{}
	planFor := func(key string, slot remoteSlot) *keyPlan {
		plan := plans[key]
		if plan == nil {
			plan = &keyPlan{slot: slot}
			plans[key] = plan
		}
		// A key first planned for a drop knows no account; one wanted later fills it in, so its
		// answer is kept.
		if plan.slot.account == "" && slot.account != "" {
			plan.slot = slot
		}
		return plan
	}
	taken := make(map[uuid.UUID]flushing, len(m.dirty))
	for id, d := range m.dirty {
		var h *hold
		acct := m.accounts[d.key]
		if acct != nil {
			h = acct.holds[id]
		}
		desired := m.shareOf(d.key, h, now)
		field := id.String()
		took := flushing{version: d.version}
		for key, value := range desired {
			plan := planFor(key, slotOf(d.key, key))
			plan.args = append(plan.args, field, value)
			plan.ids = append(plan.ids, id)
			took.keys = append(took.keys, key)
			took.desired = append(took.desired, key)
		}
		for _, key := range m.shared[id] {
			if _, kept := desired[key]; kept {
				continue
			}
			// A key of the hold's own account answers for it; one left from another it moved
			// from answers for nobody.
			slot := remoteSlot{}
			if strings.HasPrefix(key, holdPrefix+":{"+limitsOrganization(d.key)+"}:") {
				slot = slotOf(d.key, key)
			}
			plan := planFor(key, slot)
			plan.args = append(plan.args, field, "")
			plan.ids = append(plan.ids, id)
			took.keys = append(took.keys, key)
		}
		taken[id] = took
	}
	if now.Sub(m.refreshedAt) >= m.settings.HoldRefresh {
		m.refreshedAt = now
		for key, acct := range m.accounts {
			if now.Sub(acct.used) > m.settings.Idle {
				continue
			}
			organization := limitsOrganization(key)
			org := usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectOrganization}.Key(usage.CounterPrefix)
			planFor(holdKey(org), remoteSlot{account: key, counter: org})
			if acct.limited {
				for counter, subject := range acct.subjects {
					if now.Sub(subject.used) <= m.settings.Idle {
						planFor(holdKey(counter), remoteSlot{account: key, counter: counter})
					}
				}
			}
			for _, grant := range acct.grants {
				planFor(grantKey(organization, grant.grant.ID), remoteSlot{account: key, grant: grant.grant.ID})
			}
		}
	}
	return plans, taken
}

// slotOf is what a shared key of an account's hold is the total of.
func slotOf(account, key string) remoteSlot {
	if _, grant, ok := strings.Cut(key, "}:grant:"); ok {
		id, _ := strconv.ParseInt(grant, 10, 64)
		return remoteSlot{account: account, grant: id}
	}
	return remoteSlot{account: account, counter: usage.CounterPrefix + strings.TrimPrefix(key, holdPrefix)}
}

// landFlush keeps, under m.mu, what the other routers hold under each key answered, and marks
// written each hold whose every key was, unless it changed again meanwhile.
func (m *money) landFlush(plans map[string]*keyPlan, taken map[uuid.UUID]flushing, answers map[string]int64, failed map[string]bool, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, others := range answers {
		slot := plans[key].slot
		acct := m.accounts[slot.account]
		if slot.account == "" || acct == nil {
			continue
		}
		if slot.counter != "" {
			if acct.remote == nil {
				acct.remote = map[string]int64{}
			}
			if others == 0 {
				delete(acct.remote, slot.counter)
			} else {
				acct.remote[slot.counter] = others
			}
			continue
		}
		if acct.remoteGrants == nil {
			acct.remoteGrants = map[int64]int64{}
		}
		if others == 0 {
			delete(acct.remoteGrants, slot.grant)
		} else {
			acct.remoteGrants[slot.grant] = others
		}
	}
	if m.shared == nil {
		m.shared = map[uuid.UUID][]string{}
	}
	for id, took := range taken {
		written := true
		for _, key := range took.keys {
			if failed[key] {
				written = false
				break
			}
		}
		if !written {
			// Some keys may have taken it: each is let go of when it no longer should.
			for _, key := range took.desired {
				if !slices.Contains(m.shared[id], key) {
					m.shared[id] = append(m.shared[id], key)
				}
			}
			continue
		}
		holdsFlushed.Inc()
		if len(took.desired) == 0 {
			delete(m.shared, id)
		} else {
			m.shared[id] = took.desired
		}
		if d, ok := m.dirty[id]; ok && d.version == took.version {
			delete(m.dirty, id)
		}
	}
	m.unflushedSince = time.Time{}
	for _, d := range m.dirty {
		if m.unflushedSince.IsZero() || d.since.Before(m.unflushedSince) {
			m.unflushedSince = d.since
		}
	}
}

// remoteOrganization is what the other routers hold against the account's organization, which
// the balance and the spend hard limit count as they count this router's holds.
func (a *account) remoteOrganization(key string) int64 {
	if len(a.remote) == 0 {
		return 0
	}
	return a.remote[usage.SubjectRef{OrganizationID: limitsOrganization(key), Subject: usage.SubjectOrganization}.Key(usage.CounterPrefix)]
}

// grantsHeld is what every router holds against each of the account's grants.
func (a *account) grantsHeld() map[int64]int64 {
	if len(a.remoteGrants) == 0 {
		return a.grantHeld
	}
	held := make(map[int64]int64, len(a.grantHeld)+len(a.remoteGrants))
	for id, micros := range a.grantHeld {
		held[id] += micros
	}
	for id, micros := range a.remoteGrants {
		held[id] += micros
	}
	return held
}
