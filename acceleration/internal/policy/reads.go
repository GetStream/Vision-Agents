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
	now := e.settings.Now()
	spent := make(map[usage.Interval]usage.WindowSpend, len(usage.Intervals))
	if billed && e.settings.Usage != nil {
		windows, err := e.settings.Usage.Windows(ctx, []usage.SubjectRef{ref})
		if err != nil {
			return nil, err
		}
		for _, window := range windows {
			if window.SubjectRef == ref {
				spent[window.Interval] = window
			}
		}
		return spent, nil
	}
	db := e.config.DB()
	if db == nil {
		return nil, errors.New("policy: no database to sum spend from")
	}
	for _, interval := range usage.Intervals {
		start, end := usage.Window(interval, now)
		micros, err := db.SubjectSpendSince(ctx, ref, start)
		if err != nil {
			return nil, err
		}
		spent[interval] = usage.WindowSpend{SubjectRef: ref, Interval: interval, WindowStart: start, WindowEnd: end, SpentMicros: micros}
	}
	return spent, nil
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
