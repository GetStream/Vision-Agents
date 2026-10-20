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

func TestAMoveLandsNoSoonerThanMinMoveUnlessItKeepsTheCurrentRun(t *testing.T) {
	// A clock read mid-second, as a request's is: in PT15M still passes.
	now := monday.Add(500 * time.Millisecond)
	dueIn := func(d time.Duration) *time.Time {
		due := now.Add(d)
		return &due
	}
	for name, test := range map[string]struct {
		in      string
		current *time.Time
		refused bool
	}{
		"nothing pending, a minute":        {in: "PT1M", refused: true},
		"nothing pending, 14 minutes":      {in: "PT14M", refused: true},
		"nothing pending, 15 minutes":      {in: "PT15M"},
		"nothing pending, two hours":       {in: "PT2H"},
		"earlier than the run, under 15":   {in: "PT1M", current: dueIn(3 * time.Minute), refused: true},
		"later than the run, under 15":     {in: "PT10M", current: dueIn(3 * time.Minute)},
		"at the run":                       {in: "PT3M", current: dueIn(3*time.Minute - 500*time.Millisecond)},
		"a run already due does not count": {in: "PT1M", current: dueIn(-time.Minute), refused: true},
		"earlier than a far run, over 15":  {in: "PT20M", current: dueIn(2 * time.Hour)},
		"earlier than a far run, under 15": {in: "PT14M", current: dueIn(2 * time.Hour), refused: true},
	} {
		schedule, err := Parse(Input{In: test.in, Timezone: "UTC"}, now)
		require.NoError(t, err, name)
		next, _ := schedule.First()
		err = CheckMove(next, test.current, now)
		if !test.refused {
			require.NoError(t, err, name)
			continue
		}
		failure, ok := err.(*Error)
		require.True(t, ok, "%s: refusal is not a scheduling.Error: %v", name, err)
		require.Equal(t, CodeIntervalTooShort, failure.Code, name)
	}
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

func TestARepeatFromAnAtWithAnOffsetKeepsRecurring(t *testing.T) {
	// The first occurrence is the named instant; the later ones are the
	// same clock reading on later days, not that instant again.
	schedule := parse(t, Input{At: "2026-01-01T09:00:00+01:00", Repeat: "FREQ=DAILY", Timezone: "Europe/Berlin"})
	first, ok := schedule.First()
	require.True(t, ok)
	require.Equal(t, "2026-01-01T08:00:00Z", first.Format(time.RFC3339))
	next, ok := schedule.Next(monday)
	require.True(t, ok)
	require.Equal(t, "2026-10-20T07:00:00Z", next.Format(time.RFC3339))
}

func TestARepeatFromAnInListsDistinctOccurrences(t *testing.T) {
	schedule := parse(t, Input{In: "PT1H", Repeat: "FREQ=DAILY", Timezone: "UTC"})
	var got []string
	for _, at := range schedule.Upcoming(monday, 3) {
		got = append(got, at.Format(time.RFC3339))
	}
	require.Equal(t, []string{"2026-10-19T13:00:00Z", "2026-10-20T13:00:00Z", "2026-10-21T13:00:00Z"}, got)
}

func TestAnInBeyondAYearIsRefusedBeforeItOverflows(t *testing.T) {
	for _, in := range []string{"P367D", "P99999999999D", "P9999999999999999999D", "PT99999999999H", "P53W", "P366DT1S"} {
		for _, repeat := range []string{"", "FREQ=DAILY"} {
			require.Equal(t, CodeScheduleTooFar, codeOf(t, Input{In: in, Repeat: repeat, Timezone: "UTC"}), "%s %q", in, repeat)
		}
	}
	for _, repeat := range []string{"", "FREQ=DAILY"} {
		_, err := Parse(Input{In: "P364D", Repeat: repeat, Timezone: "UTC"}, monday)
		require.NoError(t, err, repeat)
	}
	// A one-shot is also held to a calendar year after the in-bound; a repeat
	// is held only to the in-bound, so exactly 366 days passes with one.
	_, err := Parse(Input{In: "P366D", Repeat: "FREQ=DAILY", Timezone: "UTC"}, monday)
	require.NoError(t, err)
}

// upcomingDays is a schedule's next three occurrences from monday, as dates.
func upcomingDays(schedule *Schedule) []string {
	var days []string
	for _, occurrence := range schedule.Upcoming(monday, 3) {
		days = append(days, occurrence.Format("Mon 2006-01-02"))
	}
	return days
}

func TestARepeatIsDescribedAsItFires(t *testing.T) {
	// Each text and its dates were read on the base, before refusals were
	// added: the repeats still accepted keep both.
	for repeat, want := range map[string]struct {
		text string
		days []string
	}{
		"FREQ=DAILY":                            {"Daily", []string{"Mon 2026-11-02", "Tue 2026-11-03", "Wed 2026-11-04"}},
		"FREQ=DAILY;INTERVAL=2":                 {"Every 2 days", []string{"Mon 2026-11-02", "Wed 2026-11-04", "Fri 2026-11-06"}},
		"FREQ=WEEKLY":                           {"Weekly", []string{"Mon 2026-11-02", "Mon 2026-11-09", "Mon 2026-11-16"}},
		"FREQ=WEEKLY;BYDAY=MO,WE":               {"Weekly on Mon, Wed", []string{"Mon 2026-11-02", "Wed 2026-11-04", "Mon 2026-11-09"}},
		"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR":      {"Every weekday", []string{"Mon 2026-11-02", "Tue 2026-11-03", "Wed 2026-11-04"}},
		"FREQ=WEEKLY;INTERVAL=2;BYDAY=FR":       {"Every 2 weeks on Fri", []string{"Fri 2026-11-06", "Fri 2026-11-20", "Fri 2026-12-04"}},
		"FREQ=MONTHLY":                          {"Monthly", []string{"Mon 2026-11-02", "Wed 2026-12-02", "Sat 2027-01-02"}},
		"FREQ=MONTHLY;BYMONTHDAY=1,15":          {"Monthly on day 1, 15", []string{"Sun 2026-11-15", "Tue 2026-12-01", "Tue 2026-12-15"}},
		"FREQ=MONTHLY;BYMONTHDAY=-1":            {"Monthly on the last day", []string{"Mon 2026-11-30", "Thu 2026-12-31", "Sun 2027-01-31"}},
		"FREQ=MONTHLY;BYMONTHDAY=-2":            {"Monthly on the second-to-last day", []string{"Sun 2026-11-29", "Wed 2026-12-30", "Sat 2027-01-30"}},
		"FREQ=MONTHLY;BYMONTHDAY=1,-1":          {"Monthly on day 1 and the last day", []string{"Mon 2026-11-30", "Tue 2026-12-01", "Thu 2026-12-31"}},
		"FREQ=MONTHLY;BYMONTHDAY=1,15,-1":       {"Monthly on day 1, day 15 and the last day", []string{"Sun 2026-11-15", "Mon 2026-11-30", "Tue 2026-12-01"}},
		"FREQ=MONTHLY;INTERVAL=2;BYMONTHDAY=-1": {"Every 2 months on the last day", []string{"Mon 2026-11-30", "Sun 2027-01-31", "Wed 2027-03-31"}},
		"FREQ=MONTHLY;BYDAY=1MO":                {"Monthly on the first Mon", []string{"Mon 2026-11-02", "Mon 2026-12-07", "Mon 2027-01-04"}},
		"FREQ=MONTHLY;BYDAY=5FR":                {"Monthly on the fifth Fri", []string{"Fri 2027-01-29", "Fri 2027-04-30", "Fri 2027-07-30"}},
		"FREQ=MONTHLY;BYDAY=-2FR":               {"Monthly on the second to last Fri", []string{"Fri 2026-11-20", "Fri 2026-12-18", "Fri 2027-01-22"}},
		"FREQ=MONTHLY;INTERVAL=3;BYDAY=1MO":     {"Every 3 months on the first Mon", []string{"Mon 2026-11-02", "Mon 2027-02-01", "Mon 2027-05-03"}},
		"FREQ=YEARLY":                           {"Yearly on 2 November", []string{"Mon 2026-11-02", "Tue 2027-11-02", "Thu 2028-11-02"}},
	} {
		schedule := parse(t, Input{At: "2026-11-02T09:00", Repeat: repeat, Timezone: "UTC"})
		require.Equal(t, want.text+" at 09:00 (UTC)", schedule.Describe(), repeat)
		require.Equal(t, want.days, upcomingDays(schedule), repeat)
	}
}

func TestARepeatItsDescriptionWouldMisstateIsRefused(t *testing.T) {
	// The comment is what Describe said on the base, and the dates what fired.
	for repeat, refusal := range map[string]string{
		"FREQ=DAILY;BYDAY=MO,WE":               "BYDAY with FREQ=DAILY",             // "Daily"; Mon, Wed, Mon
		"FREQ=DAILY;BYMONTHDAY=1":              "BYMONTHDAY with FREQ=DAILY",        // "Daily"; 1 Dec, 1 Jan
		"FREQ=WEEKLY;BYMONTHDAY=1":             "BYMONTHDAY with FREQ=WEEKLY",       // "Weekly"; 1 Dec, 1 Jan
		"FREQ=WEEKLY;BYDAY=MO,MO":              "BYDAY names MO twice",              // "Weekly on Mon, Mon"
		"FREQ=WEEKLY;BYDAY=1MO":                "only meaningful with FREQ=MONTHLY", // refused on the base too
		"FREQ=MONTHLY;BYDAY=1MO,3WE":           "more than one BYDAY",               // "on the first Mon"; 2 Nov, 18 Nov
		"FREQ=MONTHLY;BYDAY=2TU;BYMONTHDAY=13": "BYDAY with BYMONTHDAY",             // "on day 13"; 13 Apr 2027
		"FREQ=MONTHLY;BYMONTHDAY=-3":           "BYMONTHDAY=-3 is not supported",    // "day -3"
		"FREQ=MONTHLY;BYMONTHDAY=1,-31":        "BYMONTHDAY=-31 is not supported",   // "day 1, -31"
		"FREQ=MONTHLY;BYDAY=MO":                "BYDAY=MO with FREQ=MONTHLY",        // "Monthly"; every Monday
		"FREQ=MONTHLY;BYDAY=-3FR":              "BYDAY=-3FR is not supported",       // "the -3th Fri"
		"FREQ=MONTHLY;BYDAY=6FR":               "BYDAY=6FR is not supported",        // refused as schedule_in_past: never fires
		"FREQ=MONTHLY;BYMONTHDAY=1,1":          "BYMONTHDAY names 1 twice",          // "on day 1, 1"
		"FREQ=YEARLY;INTERVAL=2":               "INTERVAL with FREQ=YEARLY",         // "Yearly on 2 November"; 2026, 2028
		"FREQ=YEARLY;BYDAY=1MO":                "BYDAY with FREQ=YEARLY",            // "first Mon of November"; 4 Jan 2027
		"FREQ=YEARLY;BYMONTHDAY=1":             "BYMONTHDAY with FREQ=YEARLY",       // "Yearly on 2 November"; 1 Dec, 1 Jan
	} {
		_, err := Parse(Input{At: "2026-11-02T09:00", Repeat: repeat, Timezone: "UTC"}, monday)
		failure, ok := err.(*Error)
		require.True(t, ok, "%s: refusal is not a scheduling.Error: %v", repeat, err)
		require.Equal(t, CodeInvalidRepeat, failure.Code, repeat)
		require.Contains(t, failure.Message, refusal, repeat)
		require.Contains(t, failure.Message, supportedRepeats, repeat)
	}
}

func TestAMonthlyRepeatPastTheMonthsEndSaysItSkipsMonths(t *testing.T) {
	// Each text says what fires: the dates are the next four real occurrences.
	for _, want := range []struct {
		at, repeat, text string
		days             []string
	}{
		{"2026-10-31T09:00", "FREQ=MONTHLY", "Monthly on day 31 (day 31 is skipped in months without it)",
			[]string{"Sat 2026-10-31", "Thu 2026-12-31", "Sun 2027-01-31", "Wed 2027-03-31"}},
		{"2026-10-30T09:00", "FREQ=MONTHLY", "Monthly on day 30 (day 30 is skipped in months without it)",
			[]string{"Fri 2026-10-30", "Mon 2026-11-30", "Wed 2026-12-30", "Sat 2027-01-30"}},
		{"2026-10-29T09:00", "FREQ=MONTHLY", "Monthly on day 29 (day 29 is skipped in months without it)",
			[]string{"Thu 2026-10-29", "Sun 2026-11-29", "Tue 2026-12-29", "Fri 2027-01-29"}},
		{"2026-10-31T09:00", "FREQ=MONTHLY;INTERVAL=2", "Every 2 months on day 31 (day 31 is skipped in months without it)",
			[]string{"Sat 2026-10-31", "Thu 2026-12-31", "Tue 2027-08-31", "Sun 2027-10-31"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=31", "Monthly on day 31 (day 31 is skipped in months without it)",
			[]string{"Sat 2026-10-31", "Thu 2026-12-31", "Sun 2027-01-31", "Wed 2027-03-31"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=30", "Monthly on day 30 (day 30 is skipped in months without it)",
			[]string{"Fri 2026-10-30", "Mon 2026-11-30", "Wed 2026-12-30", "Sat 2027-01-30"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=29", "Monthly on day 29 (day 29 is skipped in months without it)",
			[]string{"Thu 2026-10-29", "Sun 2026-11-29", "Tue 2026-12-29", "Fri 2027-01-29"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=30,31", "Monthly on day 30, 31 (days 30 and 31 are skipped in months without them)",
			[]string{"Fri 2026-10-30", "Sat 2026-10-31", "Mon 2026-11-30", "Wed 2026-12-30"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=1,31", "Monthly on day 1, 31 (day 31 is skipped in months without it)",
			[]string{"Sat 2026-10-31", "Sun 2026-11-01", "Tue 2026-12-01", "Thu 2026-12-31"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=15,30", "Monthly on day 15, 30 (day 30 is skipped in months without it)",
			[]string{"Fri 2026-10-30", "Sun 2026-11-15", "Mon 2026-11-30", "Tue 2026-12-15"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=31,-1", "Monthly on day 31 and the last day (day 31 is skipped in months without it)",
			[]string{"Sat 2026-10-31", "Mon 2026-11-30", "Thu 2026-12-31", "Sun 2027-01-31"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY", "Monthly",
			[]string{"Wed 2026-10-28", "Sat 2026-11-28", "Mon 2026-12-28", "Thu 2027-01-28"}},
		{"2026-10-28T09:00", "FREQ=MONTHLY;BYMONTHDAY=28", "Monthly on day 28",
			[]string{"Wed 2026-10-28", "Sat 2026-11-28", "Mon 2026-12-28", "Thu 2027-01-28"}},
	} {
		schedule := parse(t, Input{At: want.at, Repeat: want.repeat, Timezone: "UTC"})
		require.Equal(t, want.text+" at 09:00 (UTC)", schedule.Describe(), want.at+" "+want.repeat)
		var days []string
		for _, occurrence := range schedule.Upcoming(monday, 4) {
			days = append(days, occurrence.Format("Mon 2006-01-02"))
		}
		require.Equal(t, want.days, days, want.at+" "+want.repeat)
	}
}

func TestAYearlyRepeatFromTheLeapDayFiresOnlyInLeapYears(t *testing.T) {
	schedule := parse(t, Input{At: "2028-02-29T09:00", Repeat: "FREQ=YEARLY", Timezone: "UTC"})
	require.Equal(t, "Yearly on 29 February (leap years only) at 09:00 (UTC)", schedule.Describe())
	for at, text := range map[string]string{"2027-02-02T09:00": "Yearly on 2 February", "2027-03-29T09:00": "Yearly on 29 March"} {
		other := parse(t, Input{At: at, Repeat: "FREQ=YEARLY", Timezone: "UTC"})
		require.Equal(t, text+" at 09:00 (UTC)", other.Describe())
	}
	var years []int
	for _, occurrence := range schedule.Upcoming(time.Date(2028, 1, 1, 0, 0, 0, 0, time.UTC), 3) {
		years = append(years, occurrence.Year())
	}
	require.Equal(t, []int{2028, 2032, 2036}, years)
}

func TestARefusalSaysWhichRepeatsAreSupported(t *testing.T) {
	_, err := Parse(Input{At: "2026-11-02T09:00", Repeat: "FREQ=YEARLY;BYDAY=1MO", Timezone: "UTC"}, monday)
	require.ErrorContains(t, err, "FREQ=MONTHLY, alone, with BYMONTHDAY=1,15 or with one BYDAY such as 1MO or -1FR")
}

func TestAStoredRepeatNowRefusedStillRuns(t *testing.T) {
	// A task stored before its shape was refused is read as it was: the
	// worker keeps firing it where it fired on the base.
	start := time.Date(2026, 11, 2, 9, 0, 0, 0, time.UTC)
	schedule, err := FromStored(start, "UTC", "FREQ=DAILY;BYDAY=MO,WE", time.Time{})
	require.NoError(t, err)
	require.Equal(t, []string{"Mon 2026-11-02", "Wed 2026-11-04", "Mon 2026-11-09"}, upcomingDays(schedule))
}
