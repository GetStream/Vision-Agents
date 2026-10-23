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

// Code is the short code a refusal of money is known by, as the API answers it and a campaign
// records it, or "" for any other error.
func Code(err error) string {
	switch {
	case errors.Is(err, ErrSuspended):
		return "account_suspended"
	case errors.Is(err, ErrNoCredits):
		return "insufficient_credits"
	case errors.Is(err, ErrSystemLimit):
		return "system_limit_reached"
	case errors.Is(err, ErrBudgetSpent):
		return "spend_limit_reached"
	}
	return ""
}

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
	// passTimeout bounds one pass of either loop.
	passTimeout = time.Minute
	// budgetReads is how many budgets are summed at once. Each is a sum over the request rows
	// of every shard since its window began, so a few at a time keep a slow Postgres from
	// being asked for all of them together.
	budgetReads = 4
	// logEvery spaces a failed read's error, and each organization's refusals in the log.
	logEvery = time.Minute
)

// The refusals the log and the would-refuse counter tell apart.
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
	wouldRefuse = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "admission_would_refuse_total",
		Help:      "Admissions that ROUTER_LIMITS_MODE=shadow admitted and enforce would have refused, by why",
	}, []string{"reason"})
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

// money is what an Enforcer knows of each organization's money, and the two loops that read it:
// standings from the usage service every Fresh, and budgets from Postgres every BudgetEvery,
// so how fresh a balance is never waits on Postgres.
type money struct {
	enforcer *Enforcer
	settings Settings

	mu       sync.Mutex
	accounts map[string]*account
	// holdOf is the account each hold is kept under, by usage id, so an ack naming another
	// organization still finds it.
	holdOf map[uuid.UUID]string
	// failModes is what each tier string's row said at the last standing that named it, for an
	// organization not read yet.
	failModes map[string]usage.FailMode

	kick, kickBudgets chan struct{}
	cancel            context.CancelFunc
	loops             sync.WaitGroup
	stopped           sync.Once

	errorLogged once
	refusals    throttle
	// waiting spaces the warning about holds that wait on a read, per account.
	waiting throttle
}

// account is what an organization may do with money, read from the usage service (or from
// Postgres with no usage service), only ever in the background. An app in no organization has
// one of its own (appAccount), for its budgets alone.
type account struct {
	// standing is the tier's settings with the spend hard limit, status, balance and spend this
	// day and month.
	standing usage.Standing
	// measured is every budget in its apps' decisions, with what was spent.
	measured []measured
	// budgeted is whether the decisions in use hold a budget of the account.
	budgeted bool
	// read is when the standing last landed, zero until the first. standingFrom and
	// measuredFrom are when the last standing that landed, and the last read of every budget
	// that landed, were asked for: each counts every row its ledger had by then.
	// balanceFailed is when what the usage service has of the account last could not be
	// known: a standing that did not land, or the shard not answering which held rows it
	// shipped. measureFailed is when a sum of its budgets last did not land.
	read          time.Time
	standingFrom  time.Time
	measuredFrom  time.Time
	balanceFailed time.Time
	measureFailed time.Time
	// wanted is when this process first asked for it, and used when it last did.
	wanted time.Time
	used   time.Time
	// holds is what this process recorded that the ledgers may not count yet, by usage id,
	// with its folds; folded is each usage id a fold carries, and folds the fold open for
	// each state.
	holds  map[uuid.UUID]*hold
	folded map[uuid.UUID]folded
	folds  map[fold]uuid.UUID
	// held is what the holds count against the balance, and windows what they count against
	// the spend hard limit and the budgets, kept as holds come and go so admission reads a sum.
	held    int64
	windows map[window]int64
}

// measured is a budget and what was spent against it at the last read.
type measured struct {
	scoped
	spent  int64
	resets time.Time
}

// budgetKey is which budget a measurement is of: the same scope over the same interval is the
// same spend, whatever the limit, so a limit changed is held to what was last measured at once.
type budgetKey struct {
	scope    store.PolicyScope
	id       string
	interval store.BudgetInterval
}

func keyOf(budget scoped) budgetKey {
	return budgetKey{scope: budget.scope, id: budget.id, interval: budget.budget.Interval}
}

// scope is whose spend a hold counts against.
type scope struct {
	kind store.PolicyScope
	id   string
}

// start begins the loops, unless money is not enforced at all.
func (m *money) start(e *Enforcer) {
	m.enforcer, m.settings = e, e.settings
	m.accounts, m.holdOf, m.failModes = map[string]*account{}, map[uuid.UUID]string{}, map[string]usage.FailMode{}
	if m.settings.Mode == config.LimitsOff {
		return
	}
	m.kick, m.kickBudgets = make(chan struct{}, 1), make(chan struct{}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	enforcers.Store(e, struct{}{})
	m.loops.Add(2)
	go m.every(ctx, m.settings.Fresh, m.kick, m.pass)
	go m.every(ctx, m.settings.BudgetEvery, m.kickBudgets, m.measurePass)
}

// stop ends the loops and waits for their passes to finish.
func (m *money) stop() {
	m.stopped.Do(func() {
		if m.cancel == nil {
			return
		}
		m.cancel()
		m.loops.Wait()
		m.forgetFolded()
		enforcers.Delete(m.enforcer)
	})
}

// off says money is not enforced: nothing is read or held.
func (m *money) off() bool { return m.settings.Mode == config.LimitsOff }

// want notes an organization in use, and asks both loops to read it at once when it has never
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
	for _, kick := range []chan struct{}{m.kick, m.kickBudgets} {
		select {
		case kick <- struct{}{}:
		default:
		}
	}
}

// every runs pass every interval, and at once when kicked, until ctx ends.
func (m *money) every(ctx context.Context, interval time.Duration, kick <-chan struct{}, pass func(context.Context)) {
	defer m.loops.Done()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-kick:
		}
		pass(ctx)
	}
}

// inUse is every account in use, and the budgets the decisions in use hold of each. It drops the
// decisions and the accounts unused for Idle, an account only once it holds nothing, and makes
// an account for an organization a decision names that has none.
func (m *money) inUse(now time.Time) ([]string, map[string][]scoped) {
	budgets := map[string][]scoped{}
	keys := map[string]bool{}
	for app, decided := range m.enforcer.inUse(now) {
		key := decided.organization
		if key == "" {
			if len(decided.budgets) == 0 {
				continue
			}
			key = appAccount(app)
		}
		keys[key] = true
		for _, budget := range decided.budgets {
			if !slices.Contains(budgets[key], budget) {
				budgets[key] = append(budgets[key], budget)
			}
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, acct := range m.accounts {
		if keys[key] {
			if acct.used.Before(now) {
				acct.used = now
			}
			continue
		}
		if now.Sub(acct.used) > m.settings.Idle && len(acct.holds) == 0 {
			delete(m.accounts, key)
			continue
		}
		keys[key] = true
	}
	for key := range keys {
		if m.accounts[key] == nil {
			m.accounts[key] = &account{wanted: now, used: now}
		}
	}
	return slices.Sorted(mapsKeys(keys)), budgets
}

// pass reads the standing of every organization in use from the usage service, in calls of
// standingsChunk, then lets go of the holds the reads now count.
func (m *money) pass(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, passTimeout)
	defer cancel()
	keys, budgets := m.inUse(time.Now())
	if len(keys) == 0 {
		return
	}
	readFrom := time.Now()
	standings, landed, unread := m.standings(ctx, keys)

	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, key := range keys {
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		if standing, ok := standings[key]; ok {
			acct.standing, acct.read = standing, landed[key]
			if standing.OrganizationTier != "" && standing.Tier.FailMode != "" {
				m.failModes[standing.OrganizationTier] = standing.Tier.FailMode
			}
		}
		if unread[key] {
			acct.balanceFailed = now
		} else {
			acct.standingFrom = readFrom
		}
		acct.budgeted = len(budgets[key]) > 0
		m.release(key, acct, now)
	}
}

// measurePass sums every budget the decisions in use hold, a few at a time, then lets go of the
// holds the sums now count. It also asks the shard which held rows it has shipped.
//
// It runs every BudgetEvery rather than every Fresh. A sum reads the request rows of every
// shard since the window began, up to a month of them, while what this router spends is held
// against the budget from the moment it is recorded; so the cadence only decides how soon
// another router's spend is counted, which is as soon as before admission moved off the
// request path, and a slow Postgres delays budgets alone, never a balance.
func (m *money) measurePass(ctx context.Context) {
	shipping, cancel := context.WithTimeout(ctx, shippedTimeout)
	m.shipped(shipping)
	cancel()
	// The sums have a pass's time of their own, however long the shard took.
	ctx, cancel = context.WithTimeout(ctx, passTimeout)
	defer cancel()
	_, budgets := m.inUse(time.Now())
	if len(budgets) == 0 {
		return
	}
	readFrom := time.Now()
	spent, unmeasured := m.measure(ctx, budgets)

	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, list := range budgets {
		acct := m.accounts[key]
		if acct == nil {
			continue
		}
		acct.measured = remeasured(acct.measured, list, spent)
		acct.budgeted = true
		if unmeasured[key] {
			acct.measureFailed = now
		} else {
			acct.measuredFrom = readFrom
		}
		m.release(key, acct, now)
	}
}

// standings reads the organizations' standings, each with the tier its last request carried,
// and says when each landed and which could not be read. With no usage service, and for an
// app's own account, every one is active and unlimited, with no credit to check.
func (m *money) standings(ctx context.Context, keys []string) (map[string]usage.Standing, map[string]time.Time, map[string]bool) {
	standings := make(map[string]usage.Standing, len(keys))
	landed := make(map[string]time.Time, len(keys))
	unread := map[string]bool{}
	asked := make([]string, 0, len(keys))
	now := time.Now()
	for _, key := range keys {
		if m.settings.Usage != nil && billed(key) {
			asked = append(asked, key)
			continue
		}
		standings[key] = usage.Standing{OrganizationID: key, Status: usage.AccountActive}
		landed[key] = now
	}
	for start := 0; start < len(asked); start += standingsChunk {
		part := asked[start:min(start+standingsChunk, len(asked))]
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

// measure sums what every budget's scope spent in its current interval, budgetReads at a time,
// and says which accounts had a budget it could not read.
func (m *money) measure(ctx context.Context, budgets map[string][]scoped) (map[budgetKey]measured, map[string]bool) {
	owners := map[budgetKey][]string{}
	asked := map[budgetKey]scoped{}
	for key, list := range budgets {
		for _, budget := range list {
			owners[keyOf(budget)] = append(owners[keyOf(budget)], key)
			asked[keyOf(budget)] = budget
		}
	}
	var mu sync.Mutex
	spent := make(map[budgetKey]measured, len(asked))
	unmeasured := map[string]bool{}
	slots := make(chan struct{}, budgetReads)
	var reading sync.WaitGroup
	for which, budget := range asked {
		slots <- struct{}{}
		reading.Go(func() {
			defer func() { <-slots }()
			call, cancel := context.WithTimeout(ctx, readTimeout)
			total, resets, err := m.enforcer.BudgetSpent(call, budget.scope, budget.id, budget.budget)
			cancel()
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				for _, key := range owners[which] {
					unmeasured[key] = true
				}
				m.failedRead("could not read what a budget has spent, answering with what was last read", err,
					"scope", budget.scope, "id", budget.id)
				return
			}
			spent[which] = measured{scoped: budget, spent: total, resets: resets}
		})
	}
	reading.Wait()
	return spent, unmeasured
}

// remeasured is the budgets the decisions hold now, each with what was just read for it, or
// what was read for the same scope and interval before when it could not be.
func remeasured(previous []measured, budgets []scoped, spent map[budgetKey]measured) []measured {
	out := make([]measured, 0, len(budgets))
	for _, budget := range budgets {
		read, ok := spent[keyOf(budget)]
		if !ok {
			index := slices.IndexFunc(previous, func(m measured) bool { return keyOf(m.scoped) == keyOf(budget) })
			if index < 0 {
				continue
			}
			read = previous[index]
		}
		read.scoped = budget
		out = append(out, read)
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

// spendable is what the organization's money says of the app's work: a refusal of everything
// for a suspended organization, and otherwise why it may spend no money, if it may not. In
// shadow, both are counted and logged as would refuse and neither is returned; off, neither is
// worked out.
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
	if m.off() {
		return nil, nil
	}
	if organization == "" {
		// An app in no organization bills nobody, and has only its own budgets to keep to.
		if len(decided.budgets) == 0 {
			return nil, nil
		}
		organization = appAccount(appID)
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
	refusal, spend, reason := m.refusal(acct, billed(organization), appID, decided, now)
	why, balance, held, read := acct.standing.StatusReason, acct.standing.BalanceMicros, acct.held, acct.read
	m.mu.Unlock()

	shadow := m.settings.Mode == config.LimitsShadow
	if reason == "" {
		return refusal, spend
	}
	named := who(organization)
	if billed(organization) {
		named = append(named, "app", appID)
	}
	if !shadow {
		if reason == reasonSuspended && why != "" && m.refusals.due(organization+"\x00suspended_reason", logEvery) {
			e.logger.Info("refused a suspended organization", append(named, "status_reason", why)...)
		}
		return refusal, spend
	}
	wouldRefuse.WithLabelValues(reason).Inc()
	if m.refusals.due(organization+"\x00"+reason, logEvery) {
		args := append(named, "reason", reason, "refusal", errors.Join(refusal, spend),
			"balance_micros", balance, "held_micros", held)
		if !read.IsZero() {
			args = append(args, "read_age", now.Sub(read).Round(time.Millisecond))
		}
		if reason == reasonSuspended {
			args = append(args, "status_reason", why)
		}
		e.logger.Info("would refuse", args...)
	}
	return nil, nil
}

// refusal works spendable's answer out of the account, under m.mu, and names the reason. An
// account that is not billed is an app's in no organization, which keeps to its budgets alone.
func (m *money) refusal(acct *account, isBilled bool, appID string, decided decision, now time.Time) (refusal, spend error, reason string) {
	billing := isBilled && m.settings.Usage != nil
	if acct.read.IsZero() {
		// Nothing read yet: admitted, unless it has gone unread for MaxStale and the tier it was
		// last seen on, or failing that the default, fails closed.
		if billing && now.Sub(acct.wanted) > m.settings.MaxStale &&
			m.failModes[m.enforcer.knownTier(decided.organization)] != usage.FailOpen {
			return nil, errUnreadable, reasonUnreadable
		}
		return nil, nil, ""
	}
	standing := acct.standing
	if standing.Status == usage.AccountSuspended {
		return ErrSuspended, nil, reasonSuspended
	}
	if billing && now.Sub(acct.read) > m.settings.MaxStale && standing.Tier.FailMode != usage.FailOpen {
		return nil, errUnreadable, reasonUnreadable
	}
	// The balance counts every hold; a limit or a budget only those made in the window it is
	// in now.
	if billing && standing.BalanceMicros-acct.held <= 0 {
		return nil, errNoBalance, reasonNoCredit
	}
	clock := m.settings.Now()
	for _, interval := range []usage.Interval{usage.Daily, usage.Monthly} {
		if err := overHardLimit(standing, interval, acct.heldAgainstLimit(interval, clock), clock); err != nil {
			return nil, err, string(interval) + "_spend_hard_limit"
		}
	}
	for _, budget := range decided.budgets {
		index := slices.IndexFunc(acct.measured, func(m measured) bool { return keyOf(m.scoped) == keyOf(budget) })
		if index < 0 {
			continue
		}
		read := acct.measured[index]
		read.budget = budget.budget
		if err := over(read, acct.heldAgainstBudget(budget, clock), clock); err != nil {
			return nil, err, reasonBudget
		}
	}
	return nil, nil, ""
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

// appAccountPrefix starts the key of an app's own account. No organization id holds a NUL.
const appAccountPrefix = "\x00app\x00"

// appAccount is the key of the account an app in no known organization keeps its own budgets
// on. It bills nobody, so the usage service is never asked about it.
func appAccount(appID string) string { return appAccountPrefix + appID }

// billed reports whether an account is an organization's, which the usage service bills.
func billed(key string) bool { return !strings.HasPrefix(key, appAccountPrefix) }

// noun is what an account is in a log message: an organization's, or an app's own.
func noun(key string) string {
	if billed(key) {
		return "organization"
	}
	return "app"
}

// who names an account in a log line: the organization, or the app whose own account it is.
func who(key string) []any {
	if app, own := strings.CutPrefix(key, appAccountPrefix); own {
		return []any{"app", app}
	}
	return []any{"organization", key}
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
