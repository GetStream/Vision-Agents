// Package policy enforces what an organization and its apps have decided about spend, data
// handling, prompt injection, which models may be used, how usage is labelled and what one
// end user may spend in a day, and what the operator has decided about an organization's
// money: whether it is suspended, its credit, and its spend hard limit.
//
// An organization's policy is a floor its apps can tighten and cannot loosen: a data policy
// is the stricter of the two, prompt injection is screened if either turns it on, only a
// model both allow may be routed to, and the organization's tags win over the app's. That is
// the only reading under which an organization's setting means anything, since an app that
// could switch it off would make it a suggestion. A daily limit per end user is the app's
// alone, so it has no organization's to meet.
//
// The customer's spend limits are rows of their own (store.Limit), one per subject, level and
// interval: on the organization, an app, a key, an end user, a usage category or a team. Work
// is refused once any subject it counts for has no room left under its limit, the most
// specific level's, so the organization's limit still refuses an app whose own has room.
//
// Admit is asked before every routed session and every LLM response, an in-process agent's
// speech included, so it answers from memory. What an app's policies say is its decision,
// read through appconfig and refreshed in the background once older than Settings.Fresh.
// What its organization may do with money is its account, read by two loops per Enforcer, for
// every organization in use at once: its standing from the usage service every Fresh, and its
// limits every Fresh with what their subjects spent, from the usage service's counters every
// SpendEvery (or summed from Postgres with no usage service). Work recorded since those reads
// is held against it by usage id until a read that counts it has landed. The price is that an
// organization can overshoot by what it starts before its last unit of work is recorded.
package policy

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

var tracer = tracing.Tracer("policy")

// joinTimeout bounds recording which organization an app belongs to, and every read of an
// app's policies: in the background, and the one a request with no decision yet waits on.
const joinTimeout = 2 * time.Second

// tierTTL is how long an organization's tier is believed without a request naming it before
// organization_tiers is read again, so a tier another router was sent reaches this one's work,
// a running session's included, within it.
const tierTTL = 10 * time.Second

// Settings are how an Enforcer reads money. Production passes Mode and Usage; the rest have
// defaults, which tests shorten.
type Settings struct {
	// Mode is how much of credit, the spend hard limit and spend limits is enforced; empty is
	// enforce.
	Mode config.LimitsMode
	// Usage is the usage service. Nil: money is read from Postgres, with no credit, no tier and
	// no spend hard limit, as a deployment that bills nobody runs.
	Usage *usage.Client
	// Counters is the Redis the usage service keeps every subject's spend in, which the
	// customer's limits are read from. Nil: with a usage service, its spend is read from the
	// service itself every Fresh.
	Counters rueidis.Client
	// Fresh is how often the standings and the customer's limits are read, and how old a
	// decision gets before it is read again in the background; 5 s.
	Fresh time.Duration
	// SpendEvery is how often the limited subjects' spend is read from Counters; 1 s.
	SpendEvery time.Duration
	// SumEvery is how often the limited subjects' spend is summed from Postgres with no usage
	// service, and the shard asked which held rows it shipped; 10 s. What this router spends is
	// held against them at once, so this is how soon another router's is counted.
	SumEvery time.Duration
	// MaxStale is how long an account answers after its last read before its tier's fail
	// mode decides; 60 s.
	MaxStale time.Duration
	// Idle is how long an unused decision or account is kept; 10 minutes.
	Idle time.Duration
	// PendingTTL is how long a hold waits for its row to reach the ledger before it is
	// dropped; 10 minutes.
	PendingTTL time.Duration
	// Now says which UTC day and month the spend hard limit and budgets are counted in;
	// time.Now.
	Now func() time.Time
}

// Defaults of Settings.
const (
	defaultFresh      = 5 * time.Second
	defaultSpendEvery = time.Second
	defaultSumEvery   = 10 * time.Second
	defaultMaxStale   = time.Minute
	defaultIdle       = 10 * time.Minute
	defaultPendingTTL = 10 * time.Minute
)

func (s Settings) withDefaults() Settings {
	if s.Mode == "" {
		s.Mode = config.LimitsEnforce
	}
	if s.Fresh <= 0 {
		s.Fresh = defaultFresh
	}
	if s.SpendEvery <= 0 {
		s.SpendEvery = min(defaultSpendEvery, s.Fresh)
	}
	if s.SumEvery <= 0 {
		s.SumEvery = defaultSumEvery
	}
	if s.MaxStale <= 0 {
		s.MaxStale = defaultMaxStale
	}
	if s.Idle <= 0 {
		s.Idle = defaultIdle
	}
	if s.PendingTTL <= 0 {
		s.PendingTTL = defaultPendingTTL
	}
	if s.Now == nil {
		s.Now = time.Now
	}
	return s
}

// Enforcer answers what a customer's policies and its organization's money allow. A nil
// Enforcer allows everything.
//
// Every read of the policies fails open, as the daily quota does: a database blip should not
// become an outage of everything the policies were protecting. Money fails as the
// organization's tier says, once its account has gone unread for MaxStale.
type Enforcer struct {
	config   *appconfig.Store
	logger   *slog.Logger
	settings Settings

	mu        sync.Mutex
	decisions map[string]decision
	// generation numbers each decision as it is made, so a background read that finishes after
	// the decision was dropped or replaced does not put back what it read. forgotten counts the
	// times Save dropped decisions, so a read under way when a policy was saved keeps nothing.
	generation uint64
	forgotten  uint64
	// warming is the apps whose decision is being read in the background on their behalf.
	warming sync.Map
	// members is which organization each app has already been recorded under by this
	// process, so a membership is written once rather than on every request.
	members sync.Map
	// tiers is each organization's tier as this process last knew it, from a request or from
	// organization_tiers, and what it last wrote there.
	tiersMu sync.Mutex
	tiers   map[string]tierSeen
	// now is the clock the tiers are kept by.
	now func() time.Time

	money money
}

// tierSeen is an organization's tier, empty when none is known, and when it was last known to
// be so; and the tier this process last wrote to organization_tiers, and when.
type tierSeen struct {
	tier      string
	at        time.Time
	written   string
	writtenAt time.Time
}

// decision is what an app's policies say. Read through appconfig, so from process memory once
// read, and refreshed in the background once older than Fresh.
type decision struct {
	floor  options.DataPolicy
	models []string
	tags   routing.Tags
	screen bool
	// ownApp is a customer that must act in a Stream app of its own.
	ownApp bool
	// organization is the one the app is in, empty while it is not known.
	organization string
	// messages and tokens are the daily limits per end user the app's policy sets, nil
	// where it sets none.
	messages *int64
	tokens   *int64
	// unread is a decision made without the policies, which could not be read.
	unread error
	// read is when the policies were read, used when Admit last answered with it, and
	// refreshing whether a read of it is under way.
	read       time.Time
	used       time.Time
	refreshing bool
	generation uint64
}

// New returns an Enforcer reading policies from the configuration store and money as the
// settings say, and starts its money loop, which Close stops. With Mode off no money is read.
func New(config *appconfig.Store, settings Settings, logger *slog.Logger) (*Enforcer, error) {
	if config == nil {
		return nil, errors.New("policy: a configuration store is required")
	}
	if logger == nil {
		logger = slog.Default()
	}
	e := &Enforcer{
		config: config, logger: logger, settings: settings.withDefaults(),
		decisions: map[string]decision{}, tiers: map[string]tierSeen{}, now: time.Now,
	}
	e.money.start(e)
	return e, nil
}

// Close stops the money loop and waits for it. Admit keeps answering from what it last read.
func (e *Enforcer) Close() {
	if e == nil {
		return
	}
	e.money.stop()
}

// Admit refuses an owner whose organization is suspended, says in Admission.Spend why the
// owner's work may spend no money when it may not, and says what every one of the customer's
// requests is held to. The owner's key, user and categories are the subjects the customer's
// limits are checked for, beside its organization and app. It answers from memory: a customer
// it has no decision for has its policies read now, through appconfig, and its money left to
// the loops, which admit until they land.
func (e *Enforcer) Admit(ctx context.Context, owner routing.Owner) (routing.Admission, error) {
	customerID := owner.CustomerID
	if e == nil || customerID == "" {
		return routing.Admission{}, nil
	}
	ctx, span := tracer.Start(ctx, "policy.admit")
	defer span.End()
	decided := e.decide(ctx, customerID)
	refusal, spend, grants := e.spendable(owner, decided)
	return routing.Admission{
		DataPolicy: decided.floor, Models: decided.models, Tags: decided.tags, OrganizationID: decided.organization,
		Spend: spend, Grants: grants,
	}, stack.Wrap(refusal)
}

// Attribute says who a customer's work is billed to, that organization's tier now and the tags
// their policies lay over it, for a row written after the work. It reads no money, and keeps
// no decision: a lookup that fails is the caller's to try again, where Admit would keep
// allowing every request on the decision a failed read made.
func (e *Enforcer) Attribute(ctx context.Context, customerID string) (routing.Admission, error) {
	if e == nil || customerID == "" {
		return routing.Admission{}, nil
	}
	e.mu.Lock()
	cached, ok := e.decisions[customerID]
	e.mu.Unlock()
	if ok && cached.unread == nil && time.Since(cached.read) < 2*e.settings.Fresh {
		return routing.Admission{Tags: cached.tags, OrganizationID: cached.organization, OrganizationTier: e.tier(ctx, cached.organization)}, nil
	}
	app, organization, organizationID, err := e.policies(ctx, customerID)
	if err != nil {
		return routing.Admission{}, stack.Wrap(err)
	}
	return routing.Admission{Tags: labelled(organization, app), OrganizationID: organizationID, OrganizationTier: e.tier(ctx, organizationID)}, nil
}

// Limits returns what one of a customer's end users may spend in a day: the number the
// app's policy sets, or defaults where it sets none. A limit of zero is no cap. A policy
// that could not be read leaves defaults in place.
func (e *Enforcer) Limits(ctx context.Context, customerID string, defaults quota.Limits) quota.Limits {
	if e == nil || customerID == "" {
		return defaults
	}
	decided := e.decide(ctx, customerID)
	return quota.Limits{
		MessagesPerDay: orDefault(decided.messages, defaults.MessagesPerDay),
		TokensPerDay:   orDefault(decided.tokens, defaults.TokensPerDay),
	}
}

// Join records that an app was seen under an organization, which is what an organization's
// budget and policy are applied through, and the organization's tier as the request named it,
// which work no request carries is recorded under. It also starts reading the app's decision
// and its organization's account if this process has neither, so by the time the caller's
// work is admitted both are in memory. It returns at once and does all of it in the
// background.
func (e *Enforcer) Join(appID, organizationID, tier string) {
	if e == nil || appID == "" {
		return
	}
	e.warm(appID)
	if organizationID == "" {
		return
	}
	e.money.want(organizationID)
	e.seeTier(organizationID, tier)
	if known, ok := e.members.Load(appID); ok && known == organizationID {
		return
	}
	e.members.Store(appID, organizationID)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), joinTimeout)
		defer cancel()
		if err := e.config.JoinOrganization(ctx, appID, organizationID); err != nil {
			e.members.Delete(appID)
			e.logger.Error("could not record which organization an app belongs to",
				"app", appID, "organization", organizationID, "error", err)
			return
		}
		e.refresh(appID)
	}()
}

// warm starts reading an app's decision in the background when this process has none.
func (e *Enforcer) warm(appID string) {
	e.mu.Lock()
	_, known := e.decisions[appID]
	e.mu.Unlock()
	if known {
		return
	}
	if _, already := e.warming.LoadOrStore(appID, struct{}{}); already {
		return
	}
	go func() {
		defer e.warming.Delete(appID)
		e.mu.Lock()
		forgotten := e.forgotten
		e.mu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), joinTimeout)
		defer cancel()
		decided, err := e.configuration(ctx, appID)
		if err != nil {
			// The request that comes next reads them itself, and says so if it cannot.
			return
		}
		now := time.Now()
		decided.read, decided.used = now, now
		e.mu.Lock()
		if _, raced := e.decisions[appID]; !raced && e.forgotten == forgotten {
			e.generation++
			decided.generation = e.generation
			e.decisions[appID] = decided
		}
		e.mu.Unlock()
		e.money.want(decided.organization)
	}()
}

// seeTier keeps the tier a request named for an organization, and records it in the background
// when it is not the one this process last wrote. It writes at most once per organization per
// tierTTL, so requests from proxies that disagree for a moment, as one whose cache has not
// caught up does, are no stream of writes; a tier left unwritten is written by the first
// request naming it after that, or by tier once tierTTL has passed. Empty names none.
func (e *Enforcer) seeTier(organizationID, tier string) {
	if tier == "" {
		return
	}
	now := e.now()
	e.tiersMu.Lock()
	seen := e.tiers[organizationID]
	seen.tier, seen.at = tier, now
	due := seen.written != tier && now.Sub(seen.writtenAt) >= tierTTL
	if due {
		seen.written, seen.writtenAt = tier, now
	}
	e.tiers[organizationID] = seen
	e.tiersMu.Unlock()
	if due {
		e.writeTier(organizationID, tier, now, now)
	}
}

// writeTier records in the background that a request named the tier at seenAt, the write
// marked as made at writtenAt. A write that fails is forgotten, so it is made again.
func (e *Enforcer) writeTier(organizationID, tier string, seenAt, writtenAt time.Time) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), joinTimeout)
		defer cancel()
		if err := e.config.DB().SetOrganizationTier(ctx, organizationID, tier, seenAt); err != nil {
			e.tiersMu.Lock()
			if current := e.tiers[organizationID]; current.written == tier && current.writtenAt.Equal(writtenAt) {
				current.written, current.writtenAt = "", time.Time{}
				e.tiers[organizationID] = current
			}
			e.tiersMu.Unlock()
			e.logger.Error("could not record an organization's tier", "organization", organizationID, "tier", tier, "error", err)
		}
	}()
}

// tier is an organization's tier: the one a request last named here, else the one recorded in
// organization_tiers, read again once tierTTL has passed without a request. Empty when neither
// is known, or the organization is not, and the row keeps the tier it was opened with. What a
// read found, nothing included, and a read that failed, are each believed for tierTTL, so an
// organization no request names costs one read per tierTTL rather than one per row. A tier a
// request named that was never written, as one named within tierTTL of the last write before
// the organization went quiet, is written then rather than read over.
func (e *Enforcer) tier(ctx context.Context, organizationID string) string {
	if organizationID == "" {
		return ""
	}
	now := e.now()
	e.tiersMu.Lock()
	seen, known := e.tiers[organizationID]
	fresh := known && now.Sub(seen.at) < tierTTL
	unwritten := known && !fresh && seen.tier != "" && seen.tier != seen.written
	if unwritten {
		current := seen
		current.at, current.written, current.writtenAt = now, seen.tier, now
		e.tiers[organizationID] = current
	}
	e.tiersMu.Unlock()
	if fresh {
		return seen.tier
	}
	if unwritten {
		// Named at seen.at, so a newer tier another router wrote meanwhile is kept.
		e.writeTier(organizationID, seen.tier, seen.at, now)
		return seen.tier
	}
	recorded, err := e.config.DB().TierOf(ctx, organizationID)
	e.tiersMu.Lock()
	defer e.tiersMu.Unlock()
	current := e.tiers[organizationID]
	if current.at.After(seen.at) {
		// A request named a tier while it was read, which is the newer.
		return current.tier
	}
	// A read that failed, or found none, keeps the tier last known: the tier is a label on the
	// row, and a Postgres blip is no reason to relabel it.
	if err == nil && recorded != "" {
		current.tier, current.written = recorded, recorded
	}
	current.at = e.now()
	e.tiers[organizationID] = current
	return current.tier
}

// RequiresOwnStreamApp reports whether a customer must act in a Stream app of its own. It is
// the one thing here that fails closed: a policy that could not be read is taken to
// require it, since the other answer writes a customer into an app it may have been kept
// out of on purpose.
func (e *Enforcer) RequiresOwnStreamApp(ctx context.Context, customerID string) (bool, error) {
	if e == nil || customerID == "" {
		return false, nil
	}
	decided := e.decide(ctx, customerID)
	if decided.unread != nil {
		return true, decided.unread
	}
	return decided.ownApp, nil
}

// Save replaces a scope's policy and drops the decisions it may have changed, so the next
// request of this process reads them again.
func (e *Enforcer) Save(ctx context.Context, scope store.PolicyScope, id string, document store.PolicyDocument) error {
	if err := e.config.SavePolicy(ctx, scope, id, document); err != nil {
		return err
	}
	if scope == store.ScopeApp {
		e.forget(id)
		return nil
	}
	e.mu.Lock()
	clear(e.decisions)
	e.forgotten++
	e.mu.Unlock()
	return nil
}

// decide answers from the customer's decision whatever its age and starts a background
// refresh when it is older than Fresh. With no decision at all, the policies are read now,
// through appconfig, so a data policy or a model allow-list is never skipped; the money is
// left to the loop, which admits until it lands.
func (e *Enforcer) decide(ctx context.Context, customerID string) decision {
	now := time.Now()
	e.mu.Lock()
	cached, ok := e.decisions[customerID]
	if ok {
		cached.used = now
		due := !cached.refreshing && now.Sub(cached.read) >= e.settings.Fresh
		if due {
			cached.refreshing = true
		}
		e.decisions[customerID] = cached
		e.mu.Unlock()
		if due {
			go e.refresh(customerID)
		}
		return cached
	}
	forgotten := e.forgotten
	e.mu.Unlock()

	// The request waits on this read, so it is bounded on its own as the background ones are.
	reading, cancel := context.WithTimeout(ctx, joinTimeout)
	decided, err := e.configuration(reading, customerID)
	cancel()
	if err != nil {
		// This request is admitted, which is what a budget the customer set for itself can
		// afford; its money is read apart. Nothing is kept, so the next request reads again,
		// and a read in the background keeps the decision when it lands.
		e.logger.Error("could not read a customer's policies, allowing the request",
			"customer", customerID, "error", err)
		decided.unread = err
		e.warm(customerID)
		return decided
	}
	decided.read, decided.used = now, now
	e.mu.Lock()
	// A policy saved while this was read answers this request, and the next reads it again.
	if e.forgotten == forgotten {
		e.generation++
		decided.generation = e.generation
		e.decisions[customerID] = decided
	}
	e.mu.Unlock()
	e.money.want(decided.organization)
	return decided
}

// refresh reads a customer's decision again in the background and replaces the one in memory
// with it. A read that fails keeps the decision there, and tries again after Fresh.
func (e *Enforcer) refresh(customerID string) {
	e.mu.Lock()
	generation := e.decisions[customerID].generation
	e.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), joinTimeout)
	defer cancel()
	decided, err := e.configuration(ctx, customerID)
	now := time.Now()
	e.mu.Lock()
	current, ok := e.decisions[customerID]
	if !ok || current.generation != generation {
		// Dropped or replaced while it was read: what was read may be older than what is there.
		e.mu.Unlock()
		return
	}
	if err != nil {
		current.read, current.refreshing = now, false
		e.decisions[customerID] = current
		e.mu.Unlock()
		e.money.failedRead("could not read a customer's policies again, answering with the last ones read", err, "customer", customerID)
		return
	}
	decided.read, decided.used = now, current.used
	e.generation++
	decided.generation = e.generation
	e.decisions[customerID] = decided
	e.mu.Unlock()
	e.money.want(decided.organization)
}

// policies reads the app's policy, the organization it is in and the organization's policy.
func (e *Enforcer) policies(ctx context.Context, appID string) (app, organization store.PolicyDocument, organizationID string, err error) {
	if app, err = e.config.Policy(ctx, store.ScopeApp, appID); err != nil {
		return app, organization, "", err
	}
	if organizationID, err = e.config.OrganizationOf(ctx, appID); err != nil {
		return app, organization, "", err
	}
	// An app seen for the first time is recorded under its organization in the background.
	// Until that lands, the organization it came in under is the one whose floor applies:
	// an app's first requests are not the ones to escape it.
	if pending, ok := e.members.Load(appID); ok && organizationID == "" {
		organizationID, _ = pending.(string)
	}
	if organizationID != "" {
		if organization, err = e.config.Policy(ctx, store.ScopeOrganization, organizationID); err != nil {
			return app, organization, "", err
		}
	}
	return app, organization, organizationID, nil
}

// labelled is the tags both policies lay over a request: the organization's over the app's, as
// both are over the request's.
func labelled(organization, app store.PolicyDocument) routing.Tags {
	return routing.Admission{Tags: organization.Tags}.Labelled(app.Tags)
}

// configuration reads what the app's and the organization's policies say. Spend limits are not
// policies: the limits loop reads them, with what their subjects spent.
func (e *Enforcer) configuration(ctx context.Context, appID string) (decision, error) {
	app, organization, organizationID, err := e.policies(ctx, appID)
	if err != nil {
		return decision{}, err
	}

	decided := decision{
		floor:  organization.DataPolicy.Stricter(app.DataPolicy),
		models: allowedByBoth(organization.AllowedModels, app.AllowedModels),
		tags:   labelled(organization, app),
		screen: enabled(organization.PromptInjection) || enabled(app.PromptInjection),
		// Either scope requiring it requires it: an app can ask for it and cannot opt out.
		ownApp:       enabled(organization.RequireOwnStreamApp) || enabled(app.RequireOwnStreamApp),
		organization: organizationID,
	}
	if app.Quota != nil {
		decided.messages = app.Quota.MessagesPerUserPerDay
		decided.tokens = app.Quota.TokensPerUserPerDay
	}
	return decided, nil
}

// forget drops one customer's decision so the next request reads it again, and keeps whatever
// read of it is under way from putting back what it read.
func (e *Enforcer) forget(customerID string) {
	e.mu.Lock()
	delete(e.decisions, customerID)
	e.forgotten++
	e.mu.Unlock()
}

// inUse is the decisions used within Idle, by app, dropping the rest.
func (e *Enforcer) inUse(now time.Time) map[string]decision {
	e.mu.Lock()
	defer e.mu.Unlock()
	used := make(map[string]decision, len(e.decisions))
	for app, decided := range e.decisions {
		if now.Sub(decided.used) > e.settings.Idle {
			delete(e.decisions, app)
			continue
		}
		used[app] = decided
	}
	return used
}

// decided is the decision this process holds for an app, if any.
func (e *Enforcer) decided(appID string) (decision, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	decided, ok := e.decisions[appID]
	return decided, ok
}

func enabled(flag *bool) bool { return flag != nil && *flag }

func orDefault(limit *int64, fallback int64) int64 {
	if limit == nil {
		return fallback
	}
	return *limit
}

// allowedByBoth is the models both scopes allow: nil where neither has a list, the one
// list where only one has, and what is on both where both have.
func allowedByBoth(organization, app *[]string) []string {
	switch {
	case organization == nil && app == nil:
		return nil
	case organization == nil:
		return append([]string{}, *app...)
	case app == nil:
		return append([]string{}, *organization...)
	}
	both := []string{}
	for _, model := range *app {
		if slices.Contains(*organization, model) {
			both = append(both, model)
		}
	}
	return both
}

// throttle says whether a line keyed by key is due, at most once every every.
type throttle struct {
	mu   sync.Mutex
	last map[string]time.Time
}

func (t *throttle) due(key string, every time.Duration) bool {
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	if last, ok := t.last[key]; ok && now.Sub(last) < every {
		return false
	}
	if t.last == nil {
		t.last = map[string]time.Time{}
	}
	t.last[key] = now
	return true
}

// once is a throttle of one line, in Unix nanoseconds.
type once struct{ last atomic.Int64 }

func (o *once) due(every time.Duration) bool {
	now := time.Now().UnixNano()
	previous := o.last.Load()
	if previous != 0 && now-previous < every.Nanoseconds() {
		return false
	}
	return o.last.CompareAndSwap(previous, now)
}
