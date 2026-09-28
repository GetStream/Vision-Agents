// Package policy enforces what an organization and its apps have decided about spend, data
// handling and prompt injection.
//
// An organization's policy is a floor its apps can tighten and cannot loosen: both budgets
// apply, a data policy is the stricter of the two, and prompt injection is screened if
// either turns it on. That is the only reading under which an organization's setting means
// anything, since an app that could switch it off would make it a suggestion.
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
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrBudgetSpent is what a customer whose app or organization has spent its budget is
// refused with. The message names which budget and when it resets.
var ErrBudgetSpent = errors.New("policy: the budget is spent")

// decisionTTL is how long a customer's decision is reused before the policies and the
// spend are read again.
const decisionTTL = 10 * time.Second

// joinTimeout bounds recording which organization an app belongs to.
const joinTimeout = 2 * time.Second

// Enforcer answers what a customer's policies allow. A nil Enforcer allows everything.
//
// Every read fails open, as the daily quota does: a database blip should not become an
// outage of everything the policies were protecting.
type Enforcer struct {
	store  *store.Store
	logger *slog.Logger

	mu        sync.Mutex
	decisions map[string]decision
	// members is which organization each app has already been recorded under by this
	// process, so a membership is written once rather than on every request.
	members sync.Map
}

// decision is what one customer's policies came to.
type decision struct {
	refusal error
	floor   options.DataPolicy
	screen  bool
	expires time.Time
}

// New returns an Enforcer reading policies from the store.
func New(db *store.Store, logger *slog.Logger) (*Enforcer, error) {
	if db == nil {
		return nil, errors.New("policy: a store is required")
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Enforcer{store: db, logger: logger, decisions: map[string]decision{}}, nil
}

// Admit refuses a customer whose app or organization has spent its budget, and otherwise
// returns the data policy every one of their requests is held to.
func (e *Enforcer) Admit(ctx context.Context, customerID string) (options.DataPolicy, error) {
	if e == nil || customerID == "" {
		return options.DataPolicy{}, nil
	}
	decided := e.decide(ctx, customerID)
	return decided.floor, decided.refusal
}

// Join records that an app was seen under an organization, which is what an organization's
// budget and policy are applied through. It returns at once and writes in the background.
func (e *Enforcer) Join(appID, organizationID string) {
	if e == nil || appID == "" || organizationID == "" {
		return
	}
	if known, ok := e.members.Load(appID); ok && known == organizationID {
		return
	}
	e.members.Store(appID, organizationID)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), joinTimeout)
		defer cancel()
		if err := e.store.JoinOrganization(ctx, appID, organizationID); err != nil {
			e.members.Delete(appID)
			e.logger.Error("could not record which organization an app belongs to",
				"app", appID, "organization", organizationID, "error", err)
			return
		}
		e.forget(appID)
	}()
}

// Save replaces a scope's policy and drops the decisions it may have changed.
func (e *Enforcer) Save(ctx context.Context, scope store.PolicyScope, id string, document store.PolicyDocument) error {
	if err := e.store.SavePolicy(ctx, scope, id, document); err != nil {
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
		spent, err := e.store.OrganizationSpendSince(ctx, id, start)
		return spent, end, err
	}
	spent, err := e.store.SpendSince(ctx, id, start)
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
	}
	decided.expires = now.Add(decisionTTL)
	e.mu.Lock()
	e.decisions[customerID] = decided
	e.mu.Unlock()
	return decided
}

// work reads the app's and the organization's policies and what each has spent.
func (e *Enforcer) work(ctx context.Context, appID string) (decision, error) {
	app, err := e.store.Policy(ctx, store.ScopeApp, appID)
	if err != nil {
		return decision{}, err
	}
	organizationID, err := e.store.OrganizationOf(ctx, appID)
	if err != nil {
		return decision{}, err
	}
	var organization store.PolicyDocument
	if organizationID != "" {
		if organization, err = e.store.Policy(ctx, store.ScopeOrganization, organizationID); err != nil {
			return decision{}, err
		}
	}

	decided := decision{
		floor:  organization.DataPolicy.Stricter(app.DataPolicy),
		screen: enabled(organization.PromptInjection) || enabled(app.PromptInjection),
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
	return fmt.Errorf("%w: the %s's %s budget of $%.2f is spent; it resets at %s",
		ErrBudgetSpent, scope, budget.Interval, float64(budget.LimitMicros)/1e6,
		resets.Format(time.RFC3339)), nil
}

// forget drops one customer's decision so the next request reads it again.
func (e *Enforcer) forget(customerID string) {
	e.mu.Lock()
	delete(e.decisions, customerID)
	e.mu.Unlock()
}

func enabled(flag *bool) bool { return flag != nil && *flag }
