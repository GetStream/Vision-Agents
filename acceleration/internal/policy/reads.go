package policy

import (
	"context"
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// What the limits API reads beside the table: a subject's spend now, and the operator's spend
// hard limit, which it shows beside the customer's limits. These read the ledgers themselves,
// for a person asking, and are never asked by admission.

// SubjectSpend is what the subject spent in the current window of each interval, as the usage
// service counts it (from its Postgres), or summed from the request rows of every shard when
// billed is false or there is no usage service: an app in no organization bills nobody, so
// the service never has its rows.
func (e *Enforcer) SubjectSpend(ctx context.Context, ref usage.SubjectRef, billed bool) (map[usage.Interval]usage.WindowSpend, error) {
	spent, err := e.SubjectsSpend(ctx, []usage.SubjectRef{ref}, billed)
	if err != nil {
		return nil, err
	}
	return spent[0], nil
}

// SubjectsSpend is SubjectSpend of every subject, in the order given, read from the usage
// service in one call.
func (e *Enforcer) SubjectsSpend(ctx context.Context, refs []usage.SubjectRef, billed bool) ([]map[usage.Interval]usage.WindowSpend, error) {
	now := e.settings.Now()
	spent := make([]map[usage.Interval]usage.WindowSpend, len(refs))
	for i := range spent {
		spent[i] = make(map[usage.Interval]usage.WindowSpend, len(usage.Intervals))
	}
	if billed && e.settings.Usage != nil {
		windows, err := e.settings.Usage.Windows(ctx, refs)
		if err != nil {
			return nil, err
		}
		read := make(map[usage.SubjectRef]map[usage.Interval]usage.WindowSpend, len(refs))
		for _, window := range windows {
			if read[window.SubjectRef] == nil {
				read[window.SubjectRef] = map[usage.Interval]usage.WindowSpend{}
			}
			read[window.SubjectRef][window.Interval] = window
		}
		for i, ref := range refs {
			for interval, window := range read[ref] {
				spent[i][interval] = window
			}
		}
		return spent, nil
	}
	db := e.config.DB()
	if db == nil {
		return nil, errors.New("policy: no database to sum spend from")
	}
	for i, ref := range refs {
		for _, interval := range usage.Intervals {
			start, end := usage.Window(interval, now)
			micros, err := db.SubjectSpendSince(ctx, ref, start)
			if err != nil {
				return nil, err
			}
			spent[i][interval] = usage.WindowSpend{SubjectRef: ref, Interval: interval, WindowStart: start, WindowEnd: end, SpentMicros: micros}
		}
	}
	return spent, nil
}

// Account is the organization's account as the usage service answers the organization itself,
// on the tier the router knows it by, or false with no usage service.
func (e *Enforcer) Account(ctx context.Context, organizationID string) (usage.Account, bool, error) {
	if e.settings.Usage == nil {
		return usage.Account{}, false, nil
	}
	account, err := e.settings.Usage.Account(ctx, organizationID, e.tier(ctx, organizationID))
	return account, err == nil, err
}

// Credits is one page of the organization's credit entries as the usage service answers the
// organization itself, or false with no usage service.
func (e *Enforcer) Credits(ctx context.Context, organizationID string, limit int, cursor string) (usage.CustomerCreditPage, bool, error) {
	if e.settings.Usage == nil {
		return usage.CustomerCreditPage{}, false, nil
	}
	page, err := e.settings.Usage.Credits(ctx, organizationID, limit, cursor)
	return page, err == nil, err
}

// Teams is what the organization's teams spent in their current windows as the usage service
// counts it, or false with no usage service.
func (e *Enforcer) Teams(ctx context.Context, organizationID string) (usage.Teams, bool, error) {
	if e.settings.Usage == nil {
		return usage.Teams{}, false, nil
	}
	teams, err := e.settings.Usage.Teams(ctx, organizationID)
	return teams, err == nil, err
}

// Held is what every router holds against one organization's account, as admission counts it.
type Held struct {
	// BalanceMicros is held against its balance, which is what it has to spend less this.
	BalanceMicros int64
	// DailyMicros and MonthlyMicros are held against its spend hard limit in the current UTC
	// day and calendar month.
	DailyMicros   int64
	MonthlyMicros int64
}

// Held is what every router holds now against the organization's account, from memory: none
// for one this process has not read, as its admissions would count.
func (e *Enforcer) Held(organizationID string) Held {
	if e == nil || organizationID == "" || e.money.off() {
		return Held{}
	}
	m := &e.money
	clock := m.settings.Now()
	m.mu.Lock()
	defer m.mu.Unlock()
	acct := m.accounts[organizationID]
	if acct == nil {
		return Held{}
	}
	remote := acct.remoteOrganization(organizationID)
	return Held{
		BalanceMicros: acct.held + remote,
		DailyMicros:   acct.heldAgainstLimit(usage.Daily, clock) + remote,
		MonthlyMicros: acct.heldAgainstLimit(usage.Monthly, clock) + remote,
	}
}

// Standing is the organization's standing as the usage service answers it now, with its spend
// hard limit, or false with no usage service.
func (e *Enforcer) Standing(ctx context.Context, organizationID string) (usage.Standing, bool, error) {
	if e.settings.Usage == nil {
		return usage.Standing{}, false, nil
	}
	standings, err := e.settings.Usage.Standings(ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: organizationID, OrganizationTier: e.tier(ctx, organizationID)},
	}})
	if err != nil {
		return usage.Standing{}, false, err
	}
	for _, standing := range standings {
		if standing.OrganizationID == organizationID {
			return standing, true, nil
		}
	}
	return usage.Standing{}, false, nil
}

// LimitsChanged reads an organization's limits and resets again now, after this process wrote
// one, so its next admission holds to what was written; other routers read it within Fresh.
// organization is the one the limits name, the app's own id for an app in none.
func (e *Enforcer) LimitsChanged(ctx context.Context, organization string) {
	if e == nil || e.money.off() {
		return
	}
	db := e.config.DB()
	if db == nil {
		return
	}
	m := &e.money
	m.readGrants(ctx, db, []string{organization, appAccount(organization)})
	limits, err := db.LimitsFor(ctx, []string{organization})
	if err != nil {
		m.failedRead("could not read a customer's spend limits again after a write", err, "organization", organization)
		return
	}
	resets, err := db.Resets(ctx, []string{organization}, earliestWindow(m.settings.Now()))
	if err != nil {
		m.failedRead("could not read a customer's usage resets again after a write", err, "organization", organization)
		return
	}
	m.mu.Lock()
	for _, key := range []string{organization, appAccount(organization)} {
		if acct := m.accounts[key]; acct != nil {
			m.keepLimits(acct, limits, resets, time.Now())
		}
	}
	m.mu.Unlock()
	if m.kickLimits != nil {
		nudge(m.kickLimits)
	}
}

// Resolve is the limit a subject has for an interval among an organization's limits: the row
// of the most specific level that covers it.
func Resolve(limits []store.Limit, ref usage.SubjectRef, interval usage.Interval) (store.Limit, bool) {
	return resolve(indexLimits(limits), ref, interval)
}

// Baseline is what a subject had spent in the window that starts at start when it was last
// reset there, or 0.
func Baseline(resets []store.UsageReset, ref usage.SubjectRef, interval usage.Interval, start time.Time) int64 {
	return baselineOf(resets, ref, interval, start)
}

// Room is what a subject may still spend under a limit: the limit, less what it spent since its
// baseline.
func Room(limit store.Limit, baseline, spent int64) int64 {
	return room(limit, baseline, spent, 0)
}
