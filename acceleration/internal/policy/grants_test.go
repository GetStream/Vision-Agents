package policy

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

var grantNow = time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)

// spentAccount is an account of organization o1 whose limits are rows and where each subject
// named spent micros in the current window of every interval.
func spentAccount(rows []store.Limit, spent map[usage.SubjectRef]int64, grants ...grantState) *account {
	acct := &account{limits: indexLimits(rows), limited: true, subjects: map[string]*subjectSpend{}, grants: grants}
	for ref, micros := range spent {
		windows := map[usage.Interval]spentWindow{}
		for _, interval := range usage.Intervals {
			start, _ := usage.Window(interval, grantNow)
			windows[interval] = spentWindow{start: start, micros: micros}
		}
		acct.subjects[ref.Key(usage.CounterPrefix)] = &subjectSpend{ref: ref, spent: windows, read: grantNow}
	}
	return acct
}

func grantOn(id int64, ref usage.SubjectRef, amount int64, expires *time.Time) grantState {
	return grantState{grant: store.LimitGrant{
		ID: id, OrganizationID: ref.OrganizationID, Subject: ref.Subject, AppID: ref.AppID, APIKey: ref.APIKey,
		UserID: ref.UserID, Category: ref.Category, AmountMicros: amount, ExpiresAt: expires,
	}}
}

var finance = usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectCategory, Category: "team:finance"}

func financeOwner() routing.Owner {
	return routing.Owner{CustomerID: "a1", Attribution: routing.Attribution{UsageCategories: []string{"team:finance"}}}
}

func financeLimit(interval usage.Interval, micros int64) store.Limit {
	return store.Limit{Subject: usage.SubjectCategory, Level: store.LevelOrganization, OrganizationID: "o1",
		Category: "team:finance", Interval: interval, LimitMicros: micros}
}

func TestAGrantAdmitsPastASpentLimit(t *testing.T) {
	acct := spentAccount([]store.Limit{financeLimit(usage.Daily, 10)}, map[usage.SubjectRef]int64{finance: 10},
		grantOn(7, finance, 50, nil))

	verdict := acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)
	require.NoError(t, verdict.refusal)
	require.Equal(t, []int64{7}, verdict.grants)

	acct.grants[0].used = 50
	verdict = acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)
	require.ErrorIs(t, verdict.refusal, ErrBudgetSpent, "a used grant draws nothing")
	require.Empty(t, verdict.grants)
}

func TestAGrantLiftsEveryIntervalOfItsSubject(t *testing.T) {
	rows := []store.Limit{financeLimit(usage.Daily, 100), financeLimit(usage.Weekly, 10)}
	acct := spentAccount(rows, map[usage.SubjectRef]int64{finance: 10}, grantOn(7, finance, 50, nil))

	verdict := acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)
	require.NoError(t, verdict.refusal, "the weekly one is spent, the daily one not")
	require.Equal(t, []int64{7}, verdict.grants)

	acct = spentAccount(rows, map[usage.SubjectRef]int64{finance: 1}, grantOn(7, finance, 50, nil))
	verdict = acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)
	require.NoError(t, verdict.refusal)
	require.Empty(t, verdict.grants, "with room under both, nothing is drawn")
}

func TestAGrantDoesNotLiftAParent(t *testing.T) {
	organization := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectOrganization}
	rows := []store.Limit{
		financeLimit(usage.Daily, 10),
		{Subject: usage.SubjectOrganization, Level: store.LevelOrganization, OrganizationID: "o1", Interval: usage.Daily, LimitMicros: 10},
	}
	acct := spentAccount(rows, map[usage.SubjectRef]int64{finance: 10, organization: 10}, grantOn(7, finance, 50, nil))

	verdict := acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)

	require.ErrorIs(t, verdict.refusal, ErrBudgetSpent)
	require.Equal(t, organization, verdict.ref, "refused by the organization")
}

func TestAGrantOnADefaultsSubjectLiftsOneSubject(t *testing.T) {
	u1 := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u1"}
	u2 := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u2"}
	rows := []store.Limit{{Subject: usage.SubjectUser, Level: store.LevelApp, OrganizationID: "o1", AppID: "a1", Interval: usage.Daily, LimitMicros: 1}}
	acct := spentAccount(rows, map[usage.SubjectRef]int64{u1: 1, u2: 1}, grantOn(7, u1, 1, nil))
	as := func(user string) routing.Owner {
		return routing.Owner{CustomerID: "a1", Attribution: routing.Attribution{UserID: user}}
	}

	require.NoError(t, acct.overLimits(subjectsOf(as("u1"), "o1"), grantNow).refusal)
	require.ErrorIs(t, acct.overLimits(subjectsOf(as("u2"), "o1"), grantNow).refusal, ErrUserLimit)
}

func TestAnEarlierSubjectsRefusalLeavesALaterSubjectsGrantNamed(t *testing.T) {
	organization := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectOrganization}
	rows := []store.Limit{
		financeLimit(usage.Daily, 10),
		{Subject: usage.SubjectOrganization, Level: store.LevelOrganization, OrganizationID: "o1", Interval: usage.Daily, LimitMicros: 10},
	}
	acct := spentAccount(rows, map[usage.SubjectRef]int64{finance: 10, organization: 10}, grantOn(7, finance, 50, nil))

	verdict := acct.overLimits(subjectsOf(financeOwner(), "o1"), grantNow)

	require.Equal(t, organization, verdict.ref, "refused by the organization")
	require.Equal(t, []int64{7}, verdict.grants, "work let through anyway is drawn from the team's grant")
}

func TestSpentNamesTheGrantsItsAdmissionChose(t *testing.T) {
	refs := subjectsOf(financeOwner(), "o1")
	acct := spentAccount([]store.Limit{financeLimit(usage.Daily, 10)}, map[usage.SubjectRef]int64{finance: 10},
		grantOn(7, finance, 50, nil))
	require.Equal(t, []int64{7}, acct.drawnFor(refs, grantNow), "with no admission here, what admission would choose")

	acct.noteDrawn(refs, []int64{7}, grantNow)
	acct.grants[0].used = 50
	require.Equal(t, []int64{7}, acct.drawnFor(refs, grantNow), "the grant admitted on, though it is used up since")

	acct.noteDrawn(refs, nil, grantNow)
	acct.grants[0].used = 0
	require.Empty(t, acct.drawnFor(refs, grantNow), "admitted on no grant, the row names none")
}

func TestTheGrantThatExpiresFirstIsDrawnFirst(t *testing.T) {
	tomorrow, nextWeek := grantNow.Add(24*time.Hour), grantNow.Add(7*24*time.Hour)
	grants := []grantState{grantOn(1, finance, 5, nil), grantOn(2, finance, 5, &nextWeek), grantOn(3, finance, 5, &tomorrow)}

	found, ok := grantFor(grants, nil, finance, grantNow)
	require.True(t, ok)
	require.EqualValues(t, 3, found.grant.ID, "tomorrow's")

	found, _ = grantFor(grants, map[int64]int64{3: 5}, finance, grantNow)
	require.EqualValues(t, 2, found.grant.ID, "once it is used, next week's")
	found, _ = grantFor(grants, map[int64]int64{3: 5, 2: 5}, finance, grantNow)
	require.EqualValues(t, 1, found.grant.ID, "one with no expiry last")
}

func TestAnExpiredGrantIsNotDrawn(t *testing.T) {
	at := grantNow
	_, ok := grantFor([]grantState{grantOn(1, finance, 5, &at)}, nil, finance, grantNow)

	require.False(t, ok, "at expires_at exactly it is expired")
}
