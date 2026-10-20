package scheduling

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// monday is 19 Oct 2026 at midday UTC, the "now" most parses are made at.
var monday = time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)

func parse(t *testing.T, input Input) *Schedule {
	t.Helper()
	schedule, err := Parse(input, monday)
	require.NoError(t, err)
	return schedule
}

func codeOf(t *testing.T, input Input) string {
	t.Helper()
	_, err := Parse(input, monday)
	require.Error(t, err)
	failure, ok := err.(*Error)
	require.True(t, ok, "refusal is not a scheduling.Error: %v", err)
	return failure.Code
}

func TestAOneShotParsesToItsInstant(t *testing.T) {
	schedule := parse(t, Input{At: "2026-10-21T18:00", Timezone: "Europe/Amsterdam"})
	first, ok := schedule.First()
	require.True(t, ok)
	require.Equal(t, "2026-10-21T16:00:00Z", first.Format(time.RFC3339))
	require.Equal(t, "Once, Wed 21 Oct 2026 at 18:00 (Europe/Amsterdam)", schedule.Describe())
}

func TestATimeOfDayMeansTheNextOne(t *testing.T) {
	schedule := parse(t, Input{At: "09:00", Timezone: "Europe/Amsterdam"})
	first, _ := schedule.First()
	// 09:00 on Monday has passed by midday UTC, so this is Tuesday's.
	require.Equal(t, "2026-10-20", first.In(schedule.Location()).Format("2006-01-02"))
}

func TestInResolvesFromNow(t *testing.T) {
	schedule := parse(t, Input{In: "PT20M", Timezone: "UTC"})
	first, _ := schedule.First()
	require.Equal(t, monday.Add(20*time.Minute), first)
}

func TestAPastOneShotIsRefused(t *testing.T) {
	require.Equal(t, CodeScheduleInPast, codeOf(t, Input{At: "2026-10-19T09:00", Timezone: "UTC"}))
}

func TestAOneShotOverAYearAwayIsRefused(t *testing.T) {
	require.Equal(t, CodeScheduleTooFar, codeOf(t, Input{At: "2027-11-01T09:00", Timezone: "UTC"}))
}

func TestAnUnknownZoneIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidTimezone, codeOf(t, Input{At: "2026-10-21T09:00", Timezone: "Mars/Olympus"}))
}

func TestNoZoneAtAllIsRefused(t *testing.T) {
	require.Equal(t, CodeTimezoneUnknown, codeOf(t, Input{At: "2026-10-21T09:00"}))
}

func TestBothAtAndInIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidSchedule, codeOf(t, Input{At: "2026-10-21T09:00", In: "PT5M", Timezone: "UTC"}))
}

func TestAWeekdayRepeatExpandsOnWeekdays(t *testing.T) {
	schedule := parse(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", Timezone: "Europe/Amsterdam",
	})
	upcoming := schedule.Upcoming(monday, 5)
	require.Len(t, upcoming, 5)
	for _, occurrence := range upcoming {
		require.NotEqual(t, time.Saturday, occurrence.In(schedule.Location()).Weekday())
		require.NotEqual(t, time.Sunday, occurrence.In(schedule.Location()).Weekday())
		require.Equal(t, "09:00", occurrence.In(schedule.Location()).Format("15:04"))
	}
	require.Equal(t, "Every weekday at 09:00 (Europe/Amsterdam)", schedule.Describe())
}

func TestNextIsStrictlyAfter(t *testing.T) {
	schedule := parse(t, Input{At: "2026-10-20T09:00", Repeat: "FREQ=DAILY", Timezone: "UTC"})
	first, _ := schedule.First()
	next, ok := schedule.Next(first)
	require.True(t, ok)
	require.Equal(t, first.AddDate(0, 0, 1), next)
}

func TestCountEndsTheSchedule(t *testing.T) {
	schedule := parse(t, Input{At: "2026-10-20T09:00", Repeat: "FREQ=DAILY;COUNT=3", Timezone: "UTC"})
	require.Len(t, schedule.Upcoming(monday, 10), 3)
	_, ok := schedule.Next(monday.AddDate(0, 0, 30))
	require.False(t, ok)
}

func TestUntilIsAnInclusiveLocalDate(t *testing.T) {
	schedule := parse(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=DAILY", Until: "2026-10-22", Timezone: "UTC",
	})
	upcoming := schedule.Upcoming(monday, 10)
	require.Len(t, upcoming, 3)
	require.Equal(t, "2026-10-22", upcoming[2].Format("2006-01-02"))
}

func TestABareMonthlyWeekdayIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidRepeat, codeOf(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=MONTHLY;BYDAY=MO", Timezone: "UTC",
	}))
}

func TestAnOrdinalMonthlyWeekdayExpands(t *testing.T) {
	schedule := parse(t, Input{
		At: "2026-11-02T09:00", Repeat: "FREQ=MONTHLY;BYDAY=1MO", Timezone: "UTC",
	})
	upcoming := schedule.Upcoming(monday, 2)
	require.Equal(t, "2026-11-02", upcoming[0].Format("2006-01-02"))
	require.Equal(t, "2026-12-07", upcoming[1].Format("2006-01-02"))
	require.Contains(t, schedule.Describe(), "first Mon")
}

func TestUntilInsideTheRepeatIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidRepeat, codeOf(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=DAILY;UNTIL=20261231", Timezone: "UTC",
	}))
}

func TestAnUnsupportedRepeatPartIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidRepeat, codeOf(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=DAILY;BYMONTH=10", Timezone: "UTC",
	}))
}

func TestTheSpringGapMovesForwardInNewYork(t *testing.T) {
	// time.Date resolves this backwards to 01:30 EST; the plan's rule is
	// forward by the gap, to 03:30 EDT.
	schedule, err := Parse(Input{At: "2026-03-08T02:30", Timezone: "America/New_York"},
		time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-03-08T07:30:00Z", first.Format(time.RFC3339))
	require.Equal(t, "03:30", first.In(schedule.Location()).Format("15:04"))
}

func TestTheFallOverlapTakesTheFirstOccurrenceInNewYork(t *testing.T) {
	schedule, err := Parse(Input{At: "2026-11-01T01:30", Timezone: "America/New_York"},
		time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-11-01T05:30:00Z", first.Format(time.RFC3339))
}

func TestTheSpringGapMovesForwardInAmsterdam(t *testing.T) {
	schedule, err := Parse(Input{At: "2026-03-29T02:30", Timezone: "Europe/Amsterdam"},
		time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-03-29T01:30:00Z", first.Format(time.RFC3339))
	require.Equal(t, "03:30", first.In(schedule.Location()).Format("15:04"))
}

func TestTheFallOverlapTakesTheFirstOccurrenceInAmsterdam(t *testing.T) {
	// time.Date resolves this to the second occurrence (CET); the plan's rule
	// is the first (CEST).
	schedule, err := Parse(Input{At: "2026-10-25T02:30", Timezone: "Europe/Amsterdam"},
		time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-10-25T00:30:00Z", first.Format(time.RFC3339))
}

func TestADailyTaskHoldsItsWallClockAcrossDaylightSaving(t *testing.T) {
	schedule, err := Parse(Input{At: "2026-03-06T09:00", Repeat: "FREQ=DAILY", Timezone: "America/New_York"},
		time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	upcoming := schedule.Upcoming(time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC), 4)
	require.Len(t, upcoming, 4)
	for _, occurrence := range upcoming {
		require.Equal(t, "09:00", occurrence.In(schedule.Location()).Format("15:04"))
	}
	// The Sunday of the change is an hour earlier in UTC than the Friday.
	require.Equal(t, "2026-03-06T14:00:00Z", upcoming[0].Format(time.RFC3339))
	require.Equal(t, "2026-03-08T13:00:00Z", upcoming[2].Format(time.RFC3339))
}

func TestAnInstantWithAnOffsetIsStoredAsLocalWallClock(t *testing.T) {
	schedule := parse(t, Input{At: "2026-10-21T16:00:00Z", Timezone: "Europe/Amsterdam"})
	require.Equal(t, "2026-10-21T18:00", schedule.StartLocal.Format("2006-01-02T15:04"))
}

func TestResolveZoneTakesTheFirstKnown(t *testing.T) {
	zone, err := ResolveZone("", "Europe/Amsterdam", "UTC", "America/New_York")
	require.NoError(t, err)
	require.Equal(t, "Europe/Amsterdam", zone)

	zone, err = ResolveZone("Asia/Tokyo", "Europe/Amsterdam", "", "")
	require.NoError(t, err)
	require.Equal(t, "Asia/Tokyo", zone)
}

func TestResolveZoneWithoutAnyZoneAsks(t *testing.T) {
	_, err := ResolveZone("", "", "", "")
	require.Error(t, err)
	require.Equal(t, CodeTimezoneUnknown, err.(*Error).Code)
}

func TestFromStoredRoundTrips(t *testing.T) {
	schedule := parse(t, Input{
		At: "2026-10-20T09:00", Repeat: "FREQ=WEEKLY;BYDAY=MO,WE", Until: "2026-12-31", Timezone: "UTC",
	})
	rebuilt, err := FromStored(schedule.StartLocal, schedule.Timezone, schedule.Repeat, schedule.Until)
	require.NoError(t, err)
	require.Equal(t, schedule.Describe(), rebuilt.Describe())
	require.Equal(t, schedule.Upcoming(monday, 3), rebuilt.Upcoming(monday, 3))
}

// Europe/Berlin falls back on Sun 25 Oct 2026 at 01:00 UTC: 03:00 CEST becomes
// 02:00 CET, so 02:00 to 03:00 reads twice, first at +02:00, then at +01:00.

func TestAnInLandingInTheSecondPassOfAFallBackHourKeepsItsInstant(t *testing.T) {
	// 01:50 CEST; an hour and a half on is 02:20 CET, not 02:20 CEST.
	now := time.Date(2026, 10, 24, 23, 50, 0, 0, time.UTC)
	for _, repeat := range []string{"", "FREQ=DAILY"} {
		schedule, err := Parse(Input{In: "PT1H30M", Repeat: repeat, Timezone: "Europe/Berlin"}, now)
		require.NoError(t, err)
		next, ok := schedule.Next(now)
		require.True(t, ok)
		require.Equal(t, "2026-10-25T01:20:00Z", next.Format(time.RFC3339), "repeat %q", repeat)
	}
}

func TestAnAtWithAnOffsetInTheSecondPassOfAFallBackHourKeepsItsInstant(t *testing.T) {
	// 02:10 CET, the second pass; 02:30 CET is twenty minutes on.
	now := time.Date(2026, 10, 25, 1, 10, 0, 0, time.UTC)
	schedule, err := Parse(Input{At: "2026-10-25T02:30:00+01:00", Timezone: "Europe/Berlin"}, now)
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-10-25T01:30:00Z", first.Format(time.RFC3339))
}

func TestAWallClockReadingInTheFallBackHourStillTakesTheFirstPass(t *testing.T) {
	// The same reading without an offset names no instant: the first pass.
	schedule, err := Parse(Input{At: "2026-10-25T02:30", Timezone: "Europe/Berlin"},
		time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	first, _ := schedule.First()
	require.Equal(t, "2026-10-25T00:30:00Z", first.Format(time.RFC3339))
}

func TestADayInIsTheSameClockReadingTomorrow(t *testing.T) {
	// 09:00 CEST on Saturday; Sunday is 25 hours long in Berlin.
	now := time.Date(2026, 10, 24, 7, 0, 0, 0, time.UTC)
	for in, want := range map[string]string{
		"P1D":    "2026-10-25T08:00:00Z", // 09:00 CET
		"P1DT1H": "2026-10-25T09:00:00Z", // 10:00 CET
		"PT24H":  "2026-10-25T07:00:00Z", // hours are exact: 08:00 CET
		"P1W":    "2026-10-31T08:00:00Z", // 09:00 CET a week on
	} {
		schedule, err := Parse(Input{In: in, Timezone: "Europe/Berlin"}, now)
		require.NoError(t, err, in)
		first, _ := schedule.First()
		require.Equal(t, want, first.Format(time.RFC3339), in)
	}
}

func TestAZeroInIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidSchedule, codeOf(t, Input{In: "P0D", Timezone: "UTC"}))
}

func TestTheRoutersOwnZoneIsRefused(t *testing.T) {
	require.Equal(t, CodeInvalidTimezone, codeOf(t, Input{At: "2026-10-21T09:00", Timezone: "Local"}))
	_, err := ResolveZone(" Local ", "", "", "UTC")
	require.Error(t, err)
	require.Equal(t, CodeInvalidTimezone, err.(*Error).Code)
}

func TestAPaddedZoneResolvesTrimmed(t *testing.T) {
	zone, err := ResolveZone(" Europe/Berlin ", "", "", "")
	require.NoError(t, err)
	require.Equal(t, "Europe/Berlin", zone)
	schedule := parse(t, Input{At: "2026-10-21T09:00", Timezone: " Europe/Berlin "})
	require.Equal(t, "Europe/Berlin", schedule.Timezone)
}

func TestAMonthlyRepeatFromThe31stSkipsShortMonths(t *testing.T) {
	// RFC 5545 3.3.10: an occurrence on a date that does not exist is skipped.
	schedule := parse(t, Input{At: "2026-10-31T09:00", Repeat: "FREQ=MONTHLY", Timezone: "UTC"})
	days := []string{}
	for _, occurrence := range schedule.Upcoming(monday, 3) {
		days = append(days, occurrence.Format("2006-01-02"))
	}
	require.Equal(t, []string{"2026-10-31", "2026-12-31", "2027-01-31"}, days)

	lastDay := parse(t, Input{At: "2026-10-31T09:00", Repeat: "FREQ=MONTHLY;BYMONTHDAY=-1", Timezone: "UTC"})
	days = []string{}
	for _, occurrence := range lastDay.Upcoming(monday, 3) {
		days = append(days, occurrence.Format("2006-01-02"))
	}
	require.Equal(t, []string{"2026-10-31", "2026-11-30", "2026-12-31"}, days)
}
