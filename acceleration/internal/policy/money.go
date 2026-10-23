package policy

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// ErrBudgetSpent is billable work asked for by a customer whose app or organization has spent
// its budget. The message names which budget and when it resets.
var ErrBudgetSpent = errors.New("policy: the budget is spent")

// ErrNoCredits is billable work asked for by an organization whose prepaid balance is spent, or
// whose balance could not be read for long enough that its tier refuses work it cannot bill.
var ErrNoCredits = errors.New("policy: no credit left")

// ErrSuspended is any work asked for by an organization the operator has suspended. Why stays
// with the operator, as for a telephony suspension (internal/phone/phone.go).
var ErrSuspended = errors.New("policy: this organization is suspended; contact support")

// ErrSystemLimit is billable work asked for by an organization that has spent its spend hard
// limit for the day or the month (usage.Limits.SpendHardLimit), whatever its credit.
var ErrSystemLimit = errors.New("policy: the organization's spend hard limit is reached")

// refused is a refusal of one of the kinds above, which reads as its message alone.
type refused struct {
	kind    error
	message string
}

func (r *refused) Error() string { return r.message }
func (r *refused) Unwrap() error { return r.kind }

func refusedWith(kind error, format string, args ...any) error {
	return &refused{kind: kind, message: fmt.Sprintf(format, args...)}
}

// errUnreadable is the refusal of billable work for an organization whose tier fails closed
// and whose balance has not been read for MaxStale.
var errUnreadable = refusedWith(ErrNoCredits, "the organization's balance could not be read, so work that costs money "+
	"is refused until it can be; try again shortly")

// errNoBalance is the refusal of billable work for an organization with no credit left.
var errNoBalance = refusedWith(ErrNoCredits, "the organization has no credit left; add credit to use models that cost money")

const (
	// standingsChunk is how many organizations one standings call asks about.
	standingsChunk = 100
	// readTimeout bounds one standings call, and one budget's sum.
	readTimeout = 5 * time.Second
	// passTimeout bounds one pass of the money loop.
	passTimeout = time.Minute
	// logEvery spaces a failed read's error, and each organization's refusals in the log.
	logEvery = time.Minute
)

// The refusals the log tells apart.
const (
	reasonSuspended  = "suspended"
	reasonUnreadable = "balance_unreadable"
	reasonNoCredit   = "no_credit"
	reasonBudget     = "budget"
)

// enforcers is every Enforcer of this process with a running loop, which the oldest-read gauge
// looks at.
var enforcers sync.Map

var (
	refreshErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "admission_refresh_errors_total",
		Help:      "Background reads of a customer's policies, or of its organization's money, that failed and left the last ones answering",
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "admission_oldest_read_seconds",
		Help:      "How old the oldest read of an organization's money is, among the organizations in use; one never read counts from when it was first asked about",
	}, oldestRead)
)

func oldestRead() float64 {
	var oldest time.Duration
	enforcers.Range(func(key, _ any) bool {
		oldest = max(oldest, key.(*Enforcer).money.oldest())
		return true
	})
	return oldest.Seconds()
}

// money is what an Enforcer knows of each organization's money, and the loop that reads it.
type money struct {
	enforcer *Enforcer
	settings Settings

	mu       sync.Mutex
	accounts map[string]*account
	// holdOf is the organization each hold is kept under, by usage id, so an ack naming another
	// organization still finds it.
	holdOf map[uuid.UUID]string
	// failModes is what each tier string's row said at the last standing that named it, for an
	// organization not read yet.
	failModes map[string]usage.FailMode

	kick    chan struct{}
	cancel  context.CancelFunc
	done    chan struct{}
	stopped sync.Once

	errorLogged once
	refusals    throttle
}

// account is what an organization may do with money, read from the usage service (or from
// Postgres with no usage service), only ever in the background.
type account struct {
	// standing is the tier's settings with the spend hard limit, status, balance and spend this
	// day and month.
	standing usage.Standing
	// measured is every budget in its apps' decisions, with what was spent.
	measured []measured
	// read is when the standing last landed, zero until the first; readFrom is when the last
	// read of everything the account counts was asked for, so it counts every row the ledger had
	// by then.
	read     time.Time
	readFrom time.Time
	// wanted is when this process first asked for it, and used when it last did.
	wanted time.Time
	used   time.Time
	// holds is what this process recorded that the ledger may not count yet, by usage id.
	holds map[uuid.UUID]*hold
}

// measured is a budget and what was spent against it at the last read.
type measured struct {
	scoped
	spent  int64
	resets time.Time
}

// scope is whose spend a hold counts against.
type scope struct {
	kind store.PolicyScope
	id   string
}

// hold is money one unit of work has spent, kept off the balance until the ledger counts it.
// Keyed by the row's usage id, so recording or acking it twice changes nothing. Section B makes
// holds before the work starts and shares them between routers on this shape.
type hold struct {
	// scopes are the organization and the app.
	scopes  []scope
	micros  int64
	expires time.Time
	// ackedAt is when the ledger was known to have the row, zero until then.
	ackedAt time.Time
}

// start begins the loop, unless money is not enforced at all.
func (m *money) start(e *Enforcer) {
	m.enforcer, m.settings = e, e.settings
	m.accounts, m.holdOf, m.failModes = map[string]*account{}, map[uuid.UUID]string{}, map[string]usage.FailMode{}
	if m.settings.Mode == config.LimitsOff {
		return
	}
	m.kick = make(chan struct{}, 1)
	m.done = make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	enforcers.Store(e, struct{}{})
	go m.loop(ctx)
}

// stop ends the loop and waits for its pass to finish.
func (m *money) stop() {
	m.stopped.Do(func() {
		if m.cancel == nil {
			return
		}
		m.cancel()
		<-m.done
		enforcers.Delete(m.enforcer)
	})
}

// off says money is not enforced: nothing is read or held.
func (m *money) off() bool { return m.settings.Mode == config.LimitsOff }

// want notes an organization in use, and asks the loop to read it at once when it has never
// been asked for.
func (m *money) want(organizationID string) {
	if organizationID == "" || m.off() {
		return
	}
	now := time.Now()
	m.mu.Lock()
	acct, known := m.accounts[organizationID]
	if !known {
		acct = &account{wanted: now}
		m.accounts[organizationID] = acct
	}
	acct.used = now
	m.mu.Unlock()
	if !known {
		m.wake()
	}
}

func (m *money) wake() {
	select {
	case m.kick <- struct{}{}:
	default:
	}
}

func (m *money) loop(ctx context.Context) {
	defer close(m.done)
	ticker := time.NewTicker(m.settings.Fresh)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-m.kick:
		}
		m.pass(ctx)
	}
}

// pass reads the money of every organization in use: the standings of all of them from the
// usage service in calls of standingsChunk, and every budget their apps' decisions hold from
// Postgres. Then it drops what is no longer in use, the holds a read asked for after their ack
// counts, and the holds past their expiry.
func (m *money) pass(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, passTimeout)
	defer cancel()
	e := m.enforcer
	now := time.Now()
	budgets := map[string][]scoped{}
	organizations := map[string]bool{}
	for _, decided := range e.inUse(now) {
		if decided.organization == "" {
			continue
		}
		organizations[decided.organization] = true
		for _, budget := range decided.budgets {
			if !slices.Contains(budgets[decided.organization], budget) {
				budgets[decided.organization] = append(budgets[decided.organization], budget)
			}
		}
	}
	m.mu.Lock()
	for organization, acct := range m.accounts {
		if organizations[organization] {
			if acct.used.Before(now) {
				acct.used = now
			}
			continue
		}
		if now.Sub(acct.used) > m.settings.Idle && len(acct.holds) == 0 {
			delete(m.accounts, organization)
			continue
		}
		organizations[organization] = true
	}
	for organization := range organizations {
		if m.accounts[organization] == nil {
			m.accounts[organization] = &account{wanted: now, used: now}
		}
	}
	m.mu.Unlock()
	if len(organizations) == 0 {
		return
	}

	readFrom := time.Now()
	standings, landed, unread := m.standings(ctx, slices.Sorted(mapsKeys(organizations)))
	spent, unmeasured := m.measure(ctx, budgets)

	m.mu.Lock()
	defer m.mu.Unlock()
	expired := map[string]int64{}
	for organization := range organizations {
		acct := m.accounts[organization]
		if acct == nil {
			continue
		}
		if standing, ok := standings[organization]; ok {
			acct.standing, acct.read = standing, landed[organization]
			if standing.OrganizationTier != "" && standing.Tier.FailMode != "" {
				m.failModes[standing.OrganizationTier] = standing.Tier.FailMode
			}
		}
		acct.measured = remeasured(acct.measured, budgets[organization], spent)
		if !unread[organization] && !unmeasured[organization] {
			acct.readFrom = readFrom
		}
		for id, held := range acct.holds {
			switch {
			case !held.ackedAt.IsZero() && held.ackedAt.Before(acct.readFrom):
			case !now.Before(held.expires):
				if held.ackedAt.IsZero() {
					expired[organization] += held.micros
				}
			default:
				continue
			}
			delete(acct.holds, id)
			delete(m.holdOf, id)
		}
	}
	for organization, micros := range expired {
		e.logger.Warn("usage held against an organization expired before the ledger counted it",
			"organization", organization, "micros", micros, "pending_ttl", m.settings.PendingTTL)
	}
}

// standings reads the organizations' standings, each with the tier its last request carried,
// and says when each landed and which could not be read. With no usage service every one is
// active and unlimited, with no credit to check.
func (m *money) standings(ctx context.Context, organizations []string) (map[string]usage.Standing, map[string]time.Time, map[string]bool) {
	standings := make(map[string]usage.Standing, len(organizations))
	landed := make(map[string]time.Time, len(organizations))
	unread := map[string]bool{}
	if m.settings.Usage == nil {
		now := time.Now()
		for _, organization := range organizations {
			standings[organization] = usage.Standing{OrganizationID: organization, Status: usage.AccountActive}
			landed[organization] = now
		}
		return standings, landed, unread
	}
	for start := 0; start < len(organizations); start += standingsChunk {
		part := organizations[start:min(start+standingsChunk, len(organizations))]
		query := usage.StandingQuery{Organizations: make([]usage.StandingOrganization, len(part))}
		for i, organization := range part {
			query.Organizations[i] = usage.StandingOrganization{OrganizationID: organization, OrganizationTier: m.enforcer.tier(ctx, organization)}
		}
		call, cancel := context.WithTimeout(ctx, readTimeout)
		answered, err := m.settings.Usage.Standings(call, query)
		cancel()
		if err != nil {
			for _, organization := range part {
				unread[organization] = true
			}
			m.failedRead("could not read organizations' standings, answering with the last ones read", err, "organizations", len(part))
			continue
		}
		now := time.Now()
		for _, standing := range answered {
			standings[standing.OrganizationID] = standing
			landed[standing.OrganizationID] = now
		}
		for _, organization := range part {
			if _, ok := standings[organization]; !ok {
				unread[organization] = true
			}
		}
	}
	return standings, landed, unread
}

// measure sums what every budget's scope spent in its current interval, and says which
// organizations had a budget it could not read.
func (m *money) measure(ctx context.Context, budgets map[string][]scoped) (map[scoped]measured, map[string]bool) {
	spent := map[scoped]measured{}
	unmeasured := map[string]bool{}
	for organization, list := range budgets {
		for _, budget := range list {
			if _, done := spent[budget]; done {
				continue
			}
			call, cancel := context.WithTimeout(ctx, readTimeout)
			total, resets, err := m.enforcer.BudgetSpent(call, budget.scope, budget.id, budget.budget)
			cancel()
			if err != nil {
				unmeasured[organization] = true
				m.failedRead("could not read what a budget has spent, answering with what was last read", err,
					"scope", budget.scope, "id", budget.id)
				continue
			}
			spent[budget] = measured{scoped: budget, spent: total, resets: resets}
		}
	}
	return spent, unmeasured
}

// remeasured is the budgets the decisions hold now, each with what was just read for it, or
// what was read for it before when it could not be.
func remeasured(previous []measured, budgets []scoped, spent map[scoped]measured) []measured {
	out := make([]measured, 0, len(budgets))
	for _, budget := range budgets {
		if read, ok := spent[budget]; ok {
			out = append(out, read)
			continue
		}
		if index := slices.IndexFunc(previous, func(m measured) bool { return m.scoped == budget }); index >= 0 {
			out = append(out, previous[index])
		}
	}
	return out
}

// failedRead counts a background read that failed and logs it, at most once a minute.
func (m *money) failedRead(message string, err error, args ...any) {
	refreshErrors.Inc()
	if m.errorLogged.due(logEvery) {
		m.enforcer.logger.Error(message, append(args, "error", err)...)
	}
}

// oldest is how long ago the oldest account in use was read, or asked for when it never was.
func (m *money) oldest() time.Duration {
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	var oldest time.Duration
	for _, acct := range m.accounts {
		since := acct.read
		if since.IsZero() {
			since = acct.wanted
		}
		oldest = max(oldest, now.Sub(since))
	}
	return oldest
}

// Spent holds what a unit of work recorded here cost against its owner's organization and app,
// under the row's usage id, until a read that counts the row lands. Holding it again replaces
// the amount and leaves its ack as it was. It never waits.
func (e *Enforcer) Spent(owner routing.Owner, usageID uuid.UUID, micros int64) {
	if e == nil || owner.CustomerID == "" || usageID == uuid.Nil || micros <= 0 || e.money.off() {
		return
	}
	organization := owner.OrganizationID
	if decided, ok := e.decided(owner.CustomerID); ok && decided.organization != "" {
		organization = decided.organization
	}
	if organization == "" {
		return
	}
	m := &e.money
	now := time.Now()
	m.mu.Lock()
	acct := m.accounts[organization]
	if acct == nil {
		acct = &account{wanted: now}
		m.accounts[organization] = acct
	}
	acct.used = now
	if acct.holds == nil {
		acct.holds = map[uuid.UUID]*hold{}
	}
	if previous, ok := m.holdOf[usageID]; ok && previous != organization {
		if other := m.accounts[previous]; other != nil {
			if moved := other.holds[usageID]; moved != nil {
				acct.holds[usageID] = moved
			}
			delete(other.holds, usageID)
		}
	}
	held := acct.holds[usageID]
	if held == nil {
		held = &hold{}
		acct.holds[usageID] = held
	}
	held.scopes = []scope{{kind: store.ScopeOrganization, id: organization}, {kind: store.ScopeApp, id: owner.CustomerID}}
	held.micros, held.expires = micros, now.Add(m.settings.PendingTTL)
	m.holdOf[usageID] = organization
	m.mu.Unlock()
}

// Acked notes the rows the usage service has. A hold kept for one of them is dropped once a read
// asked for after now has landed; an id this process holds nothing for, as another router's
// row, is ignored. It is WriterOptions.Shipped and SweeperOptions.Shipped.
func (e *Enforcer) Acked(acked []routing.Acked) {
	if e == nil || e.money.off() {
		return
	}
	e.money.settled(acked)
}

// Written notes the rows the shard has, which is the ledger the money is read from when there is
// no usage service; with one it changes nothing, since the usage service is. It is
// WriterOptions.Written.
func (e *Enforcer) Written(written []routing.Acked) {
	if e == nil || e.money.off() || e.settings.Usage != nil {
		return
	}
	e.money.settled(written)
}

func (m *money) settled(rows []routing.Acked) {
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, row := range rows {
		organization, ok := m.holdOf[row.UsageID]
		if !ok {
			continue
		}
		acct := m.accounts[organization]
		if acct == nil {
			continue
		}
		if held := acct.holds[row.UsageID]; held != nil && held.ackedAt.IsZero() {
			held.ackedAt = now
		}
	}
}

// spendable is what the organization's money says of the app's work: a refusal of everything
// for a suspended organization, and otherwise why it may spend no money, if it may not. In
// shadow, both are logged as would refuse and neither is returned; off, neither is worked out.
//
// The order is suspension, then the balance, then the spend hard limit, then the budgets.
// Nothing a customer writes lifts the spend hard limit.
func (e *Enforcer) spendable(appID string, decided decision) (refusal, spend error) {
	m := &e.money
	organization := decided.organization
	if organization == "" {
		// Policies that could not be read name no organization; the one the app came in under
		// still has its money checked.
		if pending, ok := e.members.Load(appID); ok {
			organization, _ = pending.(string)
			decided.organization = organization
		}
	}
	if m.off() || organization == "" {
		return nil, nil
	}
	now := time.Now()
	m.mu.Lock()
	acct := m.accounts[organization]
	if acct == nil {
		m.mu.Unlock()
		m.want(organization)
		return nil, nil
	}
	acct.used = now
	refusal, spend, reason := m.refusal(acct, appID, decided, now)
	why := acct.standing.StatusReason
	m.mu.Unlock()

	if reason == reasonSuspended && why != "" && m.refusals.due(organization+"\x00suspended_reason", logEvery) {
		e.logger.Info("refused a suspended organization", "organization", organization, "app", appID, "reason", why)
	}
	if m.settings.Mode != config.LimitsShadow || reason == "" {
		return refusal, spend
	}
	if m.refusals.due(organization+"\x00"+reason, logEvery) {
		e.logger.Info("would refuse", "organization", organization, "app", appID, "reason", reason,
			"refusal", errors.Join(refusal, spend))
	}
	return nil, nil
}

// refusal works spendable's answer out of the account, under m.mu, and names the reason.
func (m *money) refusal(acct *account, appID string, decided decision, now time.Time) (refusal, spend error, reason string) {
	if acct.read.IsZero() {
		// Nothing read yet: admitted, unless it has gone unread for MaxStale and the tier it was
		// last seen on, or failing that the default, fails closed.
		if m.settings.Usage != nil && now.Sub(acct.wanted) > m.settings.MaxStale &&
			m.failModes[m.enforcer.knownTier(decided.organization)] != usage.FailOpen {
			return nil, errUnreadable, reasonUnreadable
		}
		return nil, nil, ""
	}
	standing := acct.standing
	if standing.Status == usage.AccountSuspended {
		return ErrSuspended, nil, reasonSuspended
	}
	if m.settings.Usage != nil && now.Sub(acct.read) > m.settings.MaxStale && standing.Tier.FailMode != usage.FailOpen {
		return nil, errUnreadable, reasonUnreadable
	}
	pending := acct.pending(now, scope{})
	if m.settings.Usage != nil && standing.BalanceMicros-pending <= 0 {
		return nil, errNoBalance, reasonNoCredit
	}
	clock := m.settings.Now()
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		if err := overHardLimit(standing, interval, pending, clock); err != nil {
			return nil, err, string(interval) + "_spend_hard_limit"
		}
	}
	for _, budget := range decided.budgets {
		index := slices.IndexFunc(acct.measured, func(m measured) bool {
			return m.scope == budget.scope && m.id == budget.id && m.budget.Interval == budget.budget.Interval
		})
		if index < 0 {
			continue
		}
		read := acct.measured[index]
		read.budget = budget.budget
		if err := over(read, acct.pending(now, scope{kind: budget.scope, id: budget.id}), clock); err != nil {
			return nil, err, reasonBudget
		}
	}
	return nil, nil, ""
}

// pending is the sum of the account's live holds that count against whose, or all of them for
// the zero scope.
func (a *account) pending(now time.Time, whose scope) int64 {
	var sum int64
	for _, held := range a.holds {
		if !now.Before(held.expires) {
			continue
		}
		if whose != (scope{}) && !slices.Contains(held.scopes, whose) {
			continue
		}
		sum += held.micros
	}
	return sum
}

// overHardLimit is the refusal for an organization with no room left under its spend hard limit
// for the interval: the limit, less what the standing says it spent in the current window, less
// what is held. A window the standing read before the current one began counts as nothing
// spent.
func overHardLimit(standing usage.Standing, interval usage.Interval, pending int64, now time.Time) error {
	limit, ok := standing.Limits.SpendHardLimitMicros(interval)
	if !ok {
		return nil
	}
	window := standing.SpendWindows.Daily
	if interval == usage.Monthly {
		window = standing.SpendWindows.Monthly
	}
	start, end := usage.Window(interval, now)
	spent := int64(0)
	if window.WindowStart.Equal(start) {
		spent = window.SpentMicros
	}
	if limit-spent-pending > 0 {
		return nil
	}
	return refusedWith(ErrSystemLimit, "the organization's %s spend hard limit of %s is reached; it resets at %s",
		interval, dollars(limit), end.UTC().Format(time.RFC3339))
}

// over is the refusal for a budget that is spent, counting what is held against it, or nil for
// one that is not. A budget read in an interval that has since ended counts as nothing spent.
func over(m measured, pending int64, now time.Time) error {
	spent := m.spent
	resets := m.resets
	if !now.Before(m.resets) {
		spent = 0
		_, resets = m.budget.Interval.Window(now)
	}
	if spent+pending < m.budget.LimitMicros {
		return nil
	}
	return refusedWith(ErrBudgetSpent, "the %s's %s budget of %s is spent; it resets at %s",
		m.scope, m.budget.Interval, dollars(m.budget.LimitMicros), resets.UTC().Format(time.RFC3339))
}

// dollars writes micros as an amount of dollars with cents and thousands separators: $5,000.00.
func dollars(micros int64) string {
	sign := ""
	if micros < 0 {
		sign, micros = "-", -micros
	}
	cents := (micros + 5_000) / 10_000
	whole := strconv.FormatInt(cents/100, 10)
	var grouped strings.Builder
	for i, digit := range whole {
		if i > 0 && (len(whole)-i)%3 == 0 {
			grouped.WriteByte(',')
		}
		grouped.WriteRune(digit)
	}
	return fmt.Sprintf("%s$%s.%02d", sign, grouped.String(), cents%100)
}

// knownTier is the tier this process last knew the organization by, without reading anything.
func (e *Enforcer) knownTier(organizationID string) string {
	e.tiersMu.Lock()
	defer e.tiersMu.Unlock()
	return e.tiers[organizationID].tier
}

func mapsKeys[V any](m map[string]V) func(func(string) bool) {
	return func(yield func(string) bool) {
		for key := range m {
			if !yield(key) {
				return
			}
		}
	}
}
