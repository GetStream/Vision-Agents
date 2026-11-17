package usage

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAWindowStartsOnAUTCBoundary(t *testing.T) {
	// A Wednesday afternoon.
	now := time.Date(2026, 9, 23, 15, 42, 7, 0, time.UTC)

	cases := map[Interval][2]time.Time{
		Daily:   {time.Date(2026, 9, 23, 0, 0, 0, 0, time.UTC), time.Date(2026, 9, 24, 0, 0, 0, 0, time.UTC)},
		Weekly:  {time.Date(2026, 9, 21, 0, 0, 0, 0, time.UTC), time.Date(2026, 9, 28, 0, 0, 0, 0, time.UTC)},
		Monthly: {time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)},
	}
	for interval, want := range cases {
		start, end := Window(interval, now)
		require.Equal(t, want[0], start, interval)
		require.Equal(t, want[1], end, interval)
	}
}

func TestAWeeklyWindowOnASundayBelongsToTheWeekThatStartedMonday(t *testing.T) {
	start, _ := Window(Weekly, time.Date(2026, 9, 27, 23, 0, 0, 0, time.UTC))

	require.Equal(t, time.Date(2026, 9, 21, 0, 0, 0, 0, time.UTC), start)
}

func TestAWindowIsTheUTCOneWhateverTheZone(t *testing.T) {
	// 01:00 on the 1st in Berlin is still the last day of the month in UTC.
	berlin := time.FixedZone("CEST", 2*60*60)
	start, end := Window(Monthly, time.Date(2026, 10, 1, 1, 0, 0, 0, berlin))

	require.Equal(t, time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC), end)
}

func TestABiweeklyWindowCountsFromTheEpoch(t *testing.T) {
	start, end := Window(Biweekly, time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC))
	require.Equal(t, BiweeklyEpoch, start)
	require.Equal(t, time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC), end)

	start, _ = Window(Biweekly, time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC))
	require.Equal(t, time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC), start)

	start, end = Window(Biweekly, time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC))
	require.Equal(t, time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2026, 11, 2, 0, 0, 0, 0, time.UTC), end)
}

func TestABiweeklyWindowBeforeTheEpochEndsOnAWindowBoundary(t *testing.T) {
	start, end := Window(Biweekly, time.Date(2023, 12, 31, 23, 0, 0, 0, time.UTC))

	require.Equal(t, time.Date(2023, 12, 18, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, BiweeklyEpoch, end)
}

func TestFebruaryTwentyNinthIsInFebruarysWindow(t *testing.T) {
	leap := time.Date(2028, 2, 29, 23, 59, 59, 0, time.UTC)

	start, end := Window(Monthly, leap)
	require.Equal(t, time.Date(2028, 2, 1, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2028, 3, 1, 0, 0, 0, 0, time.UTC), end)
	start, end = Window(Daily, leap)
	require.Equal(t, time.Date(2028, 2, 29, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2028, 3, 1, 0, 0, 0, 0, time.UTC), end)
}

func TestDecembersWindowEndsOnTheFirstOfJanuary(t *testing.T) {
	start, end := Window(Monthly, time.Date(2026, 12, 31, 23, 59, 59, 0, time.UTC))
	require.Equal(t, time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC), end)

	start, end = Window(Monthly, time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC))
	require.Equal(t, time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC), start)
	require.Equal(t, time.Date(2027, 2, 1, 0, 0, 0, 0, time.UTC), end)
}

func TestAWeekSpanningNewYearIsOneWindow(t *testing.T) {
	// New Year's Day 2027 is a Friday, in the week from Monday 28 December.
	for _, at := range []time.Time{
		time.Date(2026, 12, 28, 0, 0, 0, 0, time.UTC),
		time.Date(2027, 1, 1, 12, 0, 0, 0, time.UTC),
		time.Date(2027, 1, 3, 23, 59, 59, 0, time.UTC),
	} {
		start, end := Window(Weekly, at)
		require.Equal(t, time.Date(2026, 12, 28, 0, 0, 0, 0, time.UTC), start, at)
		require.Equal(t, time.Date(2027, 1, 4, 0, 0, 0, 0, time.UTC), end, at)
	}
}

func TestABiweeklyWindowRunsAcrossAYearEnd(t *testing.T) {
	// 28 December 2026 is 78 fortnights after the epoch.
	for _, at := range []time.Time{
		time.Date(2026, 12, 28, 0, 0, 0, 0, time.UTC),
		time.Date(2027, 1, 10, 23, 59, 59, 0, time.UTC),
	} {
		start, end := Window(Biweekly, at)
		require.Equal(t, time.Date(2026, 12, 28, 0, 0, 0, 0, time.UTC), start, at)
		require.Equal(t, time.Date(2027, 1, 11, 0, 0, 0, 0, time.UTC), end, at)
	}
	start, _ := Window(Biweekly, time.Date(2026, 12, 27, 23, 59, 59, 0, time.UTC))
	require.Equal(t, time.Date(2026, 12, 14, 0, 0, 0, 0, time.UTC), start)
}

func TestAFieldNamesTheIntervalAndTheDayItStarts(t *testing.T) {
	start, _ := Window(Weekly, time.Date(2026, 10, 22, 9, 0, 0, 0, time.UTC))

	require.Equal(t, "weekly:20261019", Field(Weekly, start))
}

func TestSubjectsOfAnEvent(t *testing.T) {
	event := Event{OrganizationID: "o1", AppID: "a1", APIKey: "k1", UserID: "u1", UsageCategories: []string{"team:finance", "project:x", "team:finance"}}

	require.Equal(t, []SubjectRef{
		{OrganizationID: "o1", Subject: SubjectOrganization},
		{OrganizationID: "o1", Subject: SubjectApp, AppID: "a1"},
		{OrganizationID: "o1", Subject: SubjectAPIKey, AppID: "a1", APIKey: "k1"},
		{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1", UserID: "u1"},
		{OrganizationID: "o1", Subject: SubjectCategory, Category: "project:x"},
		{OrganizationID: "o1", Subject: SubjectAppCategory, AppID: "a1", Category: "project:x"},
		{OrganizationID: "o1", Subject: SubjectCategory, Category: "team:finance"},
		{OrganizationID: "o1", Subject: SubjectAppCategory, AppID: "a1", Category: "team:finance"},
	}, Subjects(event))
}

func TestAnEventWithNoUserOrCategoriesCountsForThreeSubjects(t *testing.T) {
	require.Len(t, Subjects(Event{OrganizationID: "o1", AppID: "a1", APIKey: "hook:chat"}), 3)
}

func TestASubjectsKeyKeepsItsOrganizationInOneSlot(t *testing.T) {
	require.Equal(t, "usage:{o1}:organization", SubjectRef{OrganizationID: "o1", Subject: SubjectOrganization}.Key("usage"))
	require.Equal(t, "usage:{o1}:user:a1|u42", SubjectRef{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1", UserID: "u42"}.Key("usage"))
	require.Equal(t, "hold:{o1}:api_key:a1|hook:chat", SubjectRef{OrganizationID: "o1", Subject: SubjectAPIKey, AppID: "a1", APIKey: "hook:chat"}.Key("hold"))
	require.Equal(t, "usage:{o1}:app_category:a1|team:x", SubjectRef{OrganizationID: "o1", Subject: SubjectAppCategory, AppID: "a1", Category: "team:x"}.Key("usage"))
}

func TestARefMustNameExactlyItsSubjectsIds(t *testing.T) {
	require.Empty(t, SubjectRef{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1", UserID: "u1"}.Problem())
	require.Equal(t, "user_id is required for subject user", SubjectRef{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1"}.Problem())
	require.Equal(t, "app_id is not an id of subject category", SubjectRef{OrganizationID: "o1", Subject: SubjectCategory, AppID: "a1", Category: "c"}.Problem())
	require.Equal(t, "organization_id is required", SubjectRef{Subject: SubjectOrganization}.Problem())
	require.Contains(t, SubjectRef{OrganizationID: "o1", Subject: "team"}.Problem(), "is not one of")
	require.Equal(t, "user_id contains |", SubjectRef{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1", UserID: "a|b"}.Problem())
	require.Equal(t, "user_id is longer than 255 bytes", SubjectRef{OrganizationID: "o1", Subject: SubjectUser, AppID: "a1", UserID: strings.Repeat("u", 256)}.Problem())
}

func TestTwoSubjectsWhoseKeysWouldMeetAreBothRefused(t *testing.T) {
	braced := SubjectRef{OrganizationID: "o}:app:x", Subject: SubjectOrganization}
	app := SubjectRef{OrganizationID: "o", Subject: SubjectApp, AppID: "x}:organization"}
	require.Equal(t, braced.Key("usage"), app.Key("usage"), "the same key, were both allowed")

	require.Equal(t, "organization_id contains }", braced.Problem())
	require.Equal(t, "app_id contains }", app.Problem())
	require.Equal(t, "organization_id contains {", SubjectRef{OrganizationID: "{o}", Subject: SubjectOrganization}.Problem())
}
