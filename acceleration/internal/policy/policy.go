// Package policy enforces what an organization and its apps have decided about spend, data
// handling, prompt injection, which models may be used, how usage is labelled and what one
// end user may spend in a day.
//
// An organization's policy is a floor its apps can tighten and cannot loosen: both budgets
// apply, a data policy is the stricter of the two, prompt injection is screened if either
// turns it on, only a model both allow may be routed to, and the organization's tags win
// over the app's. That is the only reading under which an organization's setting means
// anything, since an app that could switch it off would make it a suggestion. A daily limit
// per end user is the app's alone, so it has no organization's to meet.
//
// Decisions are cached per customer for a few seconds, because Admit is asked before every
// routed session and every LLM response, and a budget is a sum over the request rows. The
// price is that a budget can be overshot by what a customer spends inside that window.
package policy

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

var tracer = tracing.Tracer("policy")

// ErrBudgetSpent is what a customer whose app or organization has spent its budget is
// refused with. The message names which budget and when it resets.
var ErrBudgetSpent = errors.New("policy: the budget is spent")

// decisionTTL is how long a customer's decision is reused before the policies and the
// spend are read again.
const decisionTTL = 10 * time.Second

// joinTimeout bounds recording which organization an app belongs to.
const joinTimeout = 2 * time.Second

// tierTTL is how long an organization's tier is believed without a request naming it before
// organization_tiers is read again, so a tier another router was sent reaches this one's work,
// a running session's included, within it.
const tierTTL = decisionTTL

// Enforcer answers what a customer's policies allow. A nil Enforcer allows everything.
//
// Every read fails open, as the daily quota does: a database blip should not become an
// outage of everything the policies were protecting.
type Enforcer struct {
	config *appconfig.Store
	logger *slog.Logger

	mu        sync.Mutex
	decisions map[string]decision
	// members is which organization each app has already been recorded under by this
	// process, so a membership is written once rather than on every request.
	members sync.Map
	// tiers is each organization's tier as this process last knew it, from a request or from
	// organization_tiers, and what it last wrote there.
	tiersMu sync.Mutex
	tiers   map[string]tierSeen
	// now is the clock the tiers are kept by.
	now func() time.Time
}

// tierSeen is an organization's tier, empty when none is known, and when it was last known to
// be so; and the tier this process last wrote to organization_tiers, and when.
type tierSeen struct {
	tier      string
	at        time.Time
	written   string
	writtenAt time.Time
}

// decision is what one customer's policies came to.
type decision struct {
	refusal error
	floor   options.DataPolicy
	models  []string
	tags    routing.Tags
	screen  bool
	// ownApp is a customer that must act in a Stream app of its own.
	ownApp bool
	// organization is the one the app is in, empty while it is not known.
	organization string
	// messages and tokens are the daily limits per end user the app's policy sets, nil
	// where it sets none.
	messages *int64
	tokens   *int64
	// unread is a decision made without the policies, which could not be read.
	unread  error
	expires time.Time
}

// New returns an Enforcer reading policies from the configuration store.
func New(config *appconfig.Store, logger *slog.Logger) (*Enforcer, error) {
	if config == nil {
		return nil, errors.New("policy: a configuration store is required")
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Enforcer{config: config, logger: logger, decisions: map[string]decision{}, tiers: map[string]tierSeen{}, now: time.Now}, nil
}

// Admit refuses a customer whose app or organization has spent its budget, and says what
// every one of their requests is held to. The admission is returned with a refusal too.
func (e *Enforcer) Admit(ctx context.Context, customerID string) (routing.Admission, error) {
	if e == nil || customerID == "" {
		return routing.Admission{}, nil
	}
	ctx, span := tracer.Start(ctx, "policy.admit")
	defer span.End()
	decided := e.decide(ctx, customerID)
	return routing.Admission{
		DataPolicy: decided.floor, Models: decided.models, Tags: decided.tags, OrganizationID: decided.organization,
	}, stack.Wrap(decided.refusal)
}

// Attribute says who a customer's work is billed to, that organization's tier now and the tags
// their policies lay over it, for a row written after the work. It reads no spend, and keeps no decision: a lookup that
// fails is the caller's to try again, where Admit would keep allowing every request on the
// decision a failed read made.
func (e *Enforcer) Attribute(ctx context.Context, customerID string) (routing.Admission, error) {
	if e == nil || customerID == "" {
		return routing.Admission{}, nil
	}
	e.mu.Lock()
	cached, ok := e.decisions[customerID]
	e.mu.Unlock()
	if ok && cached.unread == nil && time.Now().Before(cached.expires) {
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
// that could not be read leaves defaults in place; a spend that could not be read leaves
// the limits the policy sets.
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
// which work no request carries is recorded under. It returns at once and writes in the
// background.
func (e *Enforcer) Join(appID, organizationID, tier string) {
	if e == nil || appID == "" || organizationID == "" {
		return
	}
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
		e.forget(appID)
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

// Save replaces a scope's policy and drops the decisions it may have changed.
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
	e.mu.Unlock()
	return nil
}

// Spent returns what a scope has spent in the budget's current interval, and when that
// interval ends.
func (e *Enforcer) Spent(ctx context.Context, scope store.PolicyScope, id string, budget store.Budget) (int64, time.Time, error) {
	start, end := budget.Interval.Window(time.Now())
	if scope == store.ScopeOrganization {
		spent, err := e.config.DB().OrganizationSpendSince(ctx, id, start)
		return spent, end, err
	}
	spent, err := e.config.DB().SpendSince(ctx, id, start)
	return spent, end, err
}

// decide returns the customer's cached decision, working it out again once it has expired.
func (e *Enforcer) decide(ctx context.Context, customerID string) decision {
	now := time.Now()
	e.mu.Lock()
	cached, ok := e.decisions[customerID]
	e.mu.Unlock()
	if ok && now.Before(cached.expires) {
		return cached
	}

	decided, err := e.work(ctx, customerID)
	if err != nil {
		e.logger.Error("could not read a customer's policies, allowing the request",
			"customer", customerID, "error", err)
		decided.unread = err
	}
	decided.expires = now.Add(decisionTTL)
	e.mu.Lock()
	e.decisions[customerID] = decided
	e.mu.Unlock()
	return decided
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

// work reads the app's and the organization's policies and what each has spent.
func (e *Enforcer) work(ctx context.Context, appID string) (decision, error) {
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
	if organization.Budget != nil {
		if decided.refusal, err = e.over(ctx, store.ScopeOrganization, organizationID, *organization.Budget); err != nil {
			return decided, err
		}
	}
	if decided.refusal == nil && app.Budget != nil {
		if decided.refusal, err = e.over(ctx, store.ScopeApp, appID, *app.Budget); err != nil {
			return decided, err
		}
	}
	return decided, nil
}

// over returns the refusal for a budget that is spent, or nil for one that is not. The
// second error is a spend that could not be read.
func (e *Enforcer) over(ctx context.Context, scope store.PolicyScope, id string, budget store.Budget) (refusal, err error) {
	spent, resets, err := e.Spent(ctx, scope, id, budget)
	if err != nil {
		return nil, err
	}
	if spent < budget.LimitMicros {
		return nil, nil
	}
	return stack.Wrap(fmt.Errorf("%w: the %s's %s budget of $%.2f is spent; it resets at %s",
		ErrBudgetSpent, scope, budget.Interval, float64(budget.LimitMicros)/1e6,
		resets.Format(time.RFC3339))), nil
}

// forget drops one customer's decision so the next request reads it again.
func (e *Enforcer) forget(customerID string) {
	e.mu.Lock()
	delete(e.decisions, customerID)
	e.mu.Unlock()
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
