package policy

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func keyRef(key string) usage.SubjectRef {
	return usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectAPIKey, AppID: "a1", APIKey: key}
}

func userRef(user string) usage.SubjectRef {
	return usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: user}
}

func TestANamedKeyBeatsTheAppDefaultWhichBeatsTheOrganizationDefault(t *testing.T) {
	organization := store.Limit{ID: 1, Subject: usage.SubjectAPIKey, Level: store.LevelOrganization, OrganizationID: "o1", Interval: usage.Daily, LimitMicros: 30}
	app := store.Limit{ID: 2, Subject: usage.SubjectAPIKey, Level: store.LevelApp, OrganizationID: "o1", AppID: "a1", Interval: usage.Daily, LimitMicros: 20}
	named := store.Limit{ID: 3, Subject: usage.SubjectAPIKey, Level: store.LevelAPIKey, OrganizationID: "o1", AppID: "a1", APIKey: "k1", Interval: usage.Daily, LimitMicros: 10}
	index := indexLimits([]store.Limit{organization, app, named})

	found, ok := resolve(index, keyRef("k1"), usage.Daily)
	require.True(t, ok)
	require.EqualValues(t, 3, found.ID, "the key's own")
	found, _ = resolve(index, keyRef("k2"), usage.Daily)
	require.EqualValues(t, 2, found.ID, "another key of the app has the app's default")
	found, _ = resolve(index, usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectAPIKey, AppID: "a2", APIKey: "k1"}, usage.Daily)
	require.EqualValues(t, 1, found.ID, "a key of another app has the organization's default")
	_, ok = resolve(index, keyRef("k1"), usage.Weekly)
	require.False(t, ok, "each interval resolves on its own")
}

func TestANamedUserBeatsTheDefaults(t *testing.T) {
	index := indexLimits([]store.Limit{
		{ID: 1, Subject: usage.SubjectUser, Level: store.LevelApp, OrganizationID: "o1", AppID: "a1", Interval: usage.Daily, LimitMicros: 1},
		{ID: 2, Subject: usage.SubjectUser, Level: store.LevelUser, OrganizationID: "o1", AppID: "a1", UserID: "u1", Interval: usage.Daily, LimitMicros: 5},
	})

	found, _ := resolve(index, userRef("u1"), usage.Daily)
	require.EqualValues(t, 2, found.ID)
	found, _ = resolve(index, userRef("u2"), usage.Daily)
	require.EqualValues(t, 1, found.ID)
	_, ok := resolve(index, usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a2", UserID: "u1"}, usage.Daily)
	require.False(t, ok, "the same user id in another app is another user")
}

func TestARowOfAnotherOrganizationLimitsNothing(t *testing.T) {
	row := store.Limit{Subject: usage.SubjectApp, Level: store.LevelApp, OrganizationID: "o2", AppID: "a1", Interval: usage.Daily}

	require.False(t, row.Covers(usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"}))
}

func TestAResetCountsFromItsBaseline(t *testing.T) {
	limit := store.Limit{LimitMicros: 10}
	start := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	ref := userRef("u1")
	resets := []store.UsageReset{
		{ID: 1, OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u1", Interval: usage.Daily, WindowStart: start, BaselineMicros: 4},
		{ID: 2, OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u1", Interval: usage.Daily, WindowStart: start, BaselineMicros: 9},
		{ID: 3, OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u1", Interval: usage.Daily, WindowStart: start.AddDate(0, 0, -1), BaselineMicros: 100},
	}

	require.EqualValues(t, 1, room(limit, 0, 9, 0), "spent 9 of 10")
	baseline := baselineOf(resets, ref, usage.Daily, start)
	require.EqualValues(t, 9, baseline, "the newest reset of this window")
	require.EqualValues(t, 10, room(limit, baseline, 9, 0), "a reset at 9 gives back the whole limit")
	require.EqualValues(t, 7, room(limit, baseline, 11, 1), "what was spent since, and what is held, count")
	require.Zero(t, baselineOf(resets, ref, usage.Weekly, start), "a daily reset is no weekly one")
}

func TestEveryCategoryIsChecked(t *testing.T) {
	now := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	start, _ := usage.Window(usage.Daily, now)
	b := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectCategory, Category: "b"}
	acct := &account{
		limits: indexLimits([]store.Limit{
			{ID: 1, Subject: usage.SubjectCategory, Level: store.LevelOrganization, OrganizationID: "o1", Category: "a", Interval: usage.Daily, LimitMicros: 10},
			{ID: 2, Subject: usage.SubjectCategory, Level: store.LevelOrganization, OrganizationID: "o1", Category: "b", Interval: usage.Daily, LimitMicros: 10},
		}),
		subjects: map[string]*subjectSpend{
			b.Key(usage.CounterPrefix): {ref: b, spent: map[usage.Interval]spentWindow{usage.Daily: {start: start, micros: 10}}},
		},
	}
	owner := routing.Owner{CustomerID: "a1", Attribution: routing.Attribution{UsageCategories: []string{"a", "b"}}}

	err, ref, limit := acct.overLimits(subjectsOf(owner, "o1"), now)

	require.ErrorIs(t, err, ErrBudgetSpent)
	require.Equal(t, b, ref)
	require.EqualValues(t, 2, limit.ID)
	require.EqualError(t, err, "the daily limit of $0.00 on category b is spent; it resets at 2026-10-24T00:00:00Z")
}

func TestASpendReadInAnEndedWindowCountsAsNothing(t *testing.T) {
	now := time.Date(2026, 10, 23, 0, 0, 1, 0, time.UTC)
	yesterday, _ := usage.Window(usage.Daily, now.Add(-time.Hour))
	app := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"}
	acct := &account{
		limits: indexLimits([]store.Limit{{Subject: usage.SubjectApp, Level: store.LevelApp, OrganizationID: "o1", AppID: "a1", Interval: usage.Daily, LimitMicros: 10}}),
		subjects: map[string]*subjectSpend{
			app.Key(usage.CounterPrefix): {ref: app, spent: map[usage.Interval]spentWindow{usage.Daily: {start: yesterday, micros: 50}}},
		},
	}

	err, _, _ := acct.overLimits(subjectsOf(routing.Owner{CustomerID: "a1"}, "o1"), now)

	require.NoError(t, err, "yesterday's spend is not today's")
}

func TestAUsersRefusalIsItsOwnKindAndSaysWhenItResets(t *testing.T) {
	resets := time.Date(2026, 10, 24, 0, 0, 0, 0, time.UTC)
	limit := store.Limit{Subject: usage.SubjectUser, Level: store.LevelApp, AppID: "a1", Interval: usage.Daily, LimitMicros: 1_000_000}

	err := limitRefusal(limit, userRef("u1"), resets)

	require.ErrorIs(t, err, ErrUserLimit)
	require.False(t, errors.Is(err, ErrBudgetSpent), "a user's limit stops the user alone")
	require.Equal(t, "user_limit_reached", Code(err))
	require.EqualError(t, err, "the daily limit of $1.00 on every user of app a1 is spent for user u1; it resets at 2026-10-24T00:00:00Z")
	at, ok := ResetsAt(err)
	require.True(t, ok)
	require.Equal(t, resets, at)
}

func TestARefusalNamesWhatItsLimitIsOn(t *testing.T) {
	cases := map[string]store.Limit{
		"the organization":              {Subject: usage.SubjectOrganization, Level: store.LevelOrganization},
		"app a1":                        {Subject: usage.SubjectApp, Level: store.LevelApp, AppID: "a1"},
		"every key of the organization": {Subject: usage.SubjectAPIKey, Level: store.LevelOrganization},
		"key k1 of app a1":              {Subject: usage.SubjectAPIKey, Level: store.LevelAPIKey, AppID: "a1", APIKey: "k1"},
		"user u1 of app a1":             {Subject: usage.SubjectUser, Level: store.LevelUser, AppID: "a1", UserID: "u1"},
		"team finance":                  {Subject: usage.SubjectCategory, Level: store.LevelOrganization, Category: "team:finance"},
		"team finance in app a1":        {Subject: usage.SubjectAppCategory, Level: store.LevelApp, AppID: "a1", Category: "team:finance"},
		"category project:x in app a1":  {Subject: usage.SubjectAppCategory, Level: store.LevelApp, AppID: "a1", Category: "project:x"},
	}
	for want, limit := range cases {
		require.Equal(t, want, whatLimits(limit))
	}
}

func TestAnOwnersSubjectsLeaveOutAnOriginAndAUserTheRecordsWouldDrop(t *testing.T) {
	owner := routing.Owner{CustomerID: "a1", Attribution: routing.Attribution{APIKey: "hook:chat", UserID: "auth0|abc", UsageCategories: []string{"team:x"}}}

	refs := subjectsOf(owner, "o1")

	require.Equal(t, []usage.SubjectRef{
		{OrganizationID: "o1", Subject: usage.SubjectOrganization},
		{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"},
		{OrganizationID: "o1", Subject: usage.SubjectCategory, Category: "team:x"},
		{OrganizationID: "o1", Subject: usage.SubjectAppCategory, AppID: "a1", Category: "team:x"},
	}, refs)
}
