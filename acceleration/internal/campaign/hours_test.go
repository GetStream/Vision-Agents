package campaign

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type HoursSuite struct {
	suite.Suite
}

func TestHoursSuite(t *testing.T) {
	suite.Run(t, new(HoursSuite))
}

func (s *HoursSuite) zone(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	s.Require().NoError(err)
	return loc
}

func (s *HoursSuite) onlyWindows(days []string, windows ...Window) CallingHours {
	h := DefaultCallingHours()
	h.Windows = map[string][]Window{}
	for _, d := range days {
		h.Windows[d] = windows
	}
	return h
}

var weekdays = []string{"monday", "tuesday", "wednesday", "thursday", "friday"}

func (s *HoursSuite) TestAContactInChicagoIsCalledOnChicagoTime() {
	h := DefaultCallingHours()
	zone := h.Zone("", "+13125550177")
	s.Equal("America/Chicago", zone.String())

	// 2026-11-03 is a Tuesday, CST (UTC-6).
	s.False(h.Open(time.Date(2026, 11, 3, 8, 59, 0, 0, zone), zone))
	s.True(h.Open(time.Date(2026, 11, 3, 9, 0, 0, 0, zone), zone))
	s.True(h.Open(time.Date(2026, 11, 3, 15, 0, 0, 0, time.UTC), zone))
	s.False(h.Open(time.Date(2026, 11, 3, 14, 59, 0, 0, time.UTC), zone))
}

func (s *HoursSuite) TestAContactsOwnZoneWinsOverItsAreaCode() {
	h := DefaultCallingHours()
	s.Equal("Europe/London", h.Zone("Europe/London", "+13125550177").String())
}

func (s *HoursSuite) TestAContactZoneThatDoesNotLoadFallsBackToItsAreaCode() {
	h := DefaultCallingHours()
	s.Equal("America/Chicago", h.Zone("Not/AZone", "+13125550177").String())
}

func (s *HoursSuite) TestANumberWithNoKnownZoneUsesTheFallback() {
	h := DefaultCallingHours()
	h.TimeZone.Fallback = "Asia/Tokyo"
	s.Equal("Asia/Tokyo", h.Zone("", "+18005550100").String())
	s.Equal("Asia/Tokyo", h.Zone("", "garbage").String())
}

func (s *HoursSuite) TestFixedModeIgnoresTheContact() {
	h := DefaultCallingHours()
	h.TimeZone = TimeZone{Mode: ZoneFixed, Fallback: "UTC", Fixed: "Europe/Berlin"}
	s.Equal("Europe/Berlin", h.Zone("America/Chicago", "+13125550177").String())
}

func (s *HoursSuite) TestNoNewCallInTheLastMinuteOfAWindow() {
	h := DefaultCallingHours()
	zone := s.zone("America/Chicago")
	s.True(h.Open(time.Date(2026, 11, 3, 17, 58, 59, 0, zone), zone))
	s.False(h.Open(time.Date(2026, 11, 3, 17, 59, 0, 0, zone), zone))
	s.False(h.Open(time.Date(2026, 11, 3, 18, 0, 0, 0, zone), zone))
}

func (s *HoursSuite) TestAnAttemptDueOutsideHoursWaitsForTheNextOpening() {
	h := DefaultCallingHours()
	zone := s.zone("America/Chicago")

	// Friday 2026-11-06 19:00 local.
	next, ok := h.Next(time.Date(2026, 11, 6, 19, 0, 0, 0, zone), zone)
	s.True(ok)
	s.True(time.Date(2026, 11, 9, 9, 0, 0, 0, zone).Equal(next), next.String())

	// Already open: the same instant.
	now := time.Date(2026, 11, 3, 10, 30, 0, 0, zone)
	next, ok = h.Next(now, zone)
	s.True(ok)
	s.True(now.Equal(next))

	// In the last minute of a window: tomorrow.
	next, ok = h.Next(time.Date(2026, 11, 3, 17, 59, 30, 0, zone), zone)
	s.True(ok)
	s.True(time.Date(2026, 11, 4, 9, 0, 0, 0, zone).Equal(next), next.String())
}

func (s *HoursSuite) TestAnExcludedDateIsSkippedInTheContactsLocalDate() {
	h := DefaultCallingHours()
	h.ExcludedDates = []string{"2026-11-26"}
	zone := s.zone("America/Los_Angeles")

	// 2026-11-26 17:30 in Los Angeles is 01:30 UTC on the 27th.
	at := time.Date(2026, 11, 27, 1, 30, 0, 0, time.UTC)
	s.Equal(26, at.In(zone).Day())
	s.False(h.Open(at, zone))

	// The same UTC instant seen from Tokyo is already the 27th, a Friday.
	tokyo := s.zone("Asia/Tokyo")
	s.True(h.Open(time.Date(2026, 11, 27, 0, 30, 0, 0, time.UTC), tokyo))

	next, ok := h.Next(at, zone)
	s.True(ok)
	s.True(time.Date(2026, 11, 27, 9, 0, 0, 0, zone).Equal(next), next.String())
}

func (s *HoursSuite) TestTwoWindowsInADaySkipLunch() {
	h := s.onlyWindows(weekdays, Window{"09:00", "12:00"}, Window{"13:00", "17:00"})
	s.Require().NoError(h.Validate())
	zone := s.zone("UTC")

	s.True(h.Open(time.Date(2026, 11, 3, 11, 58, 0, 0, zone), zone))
	s.False(h.Open(time.Date(2026, 11, 3, 12, 30, 0, 0, zone), zone))
	s.True(h.Open(time.Date(2026, 11, 3, 13, 0, 0, 0, zone), zone))

	next, ok := h.Next(time.Date(2026, 11, 3, 12, 30, 0, 0, zone), zone)
	s.True(ok)
	s.True(time.Date(2026, 11, 3, 13, 0, 0, 0, zone).Equal(next))
}

func (s *HoursSuite) TestWindowsOutOfOrderAreStillFoundInOrder() {
	h := s.onlyWindows(weekdays, Window{"13:00", "17:00"}, Window{"09:00", "12:00"})
	s.Require().NoError(h.Validate())
	zone := s.zone("UTC")

	next, ok := h.Next(time.Date(2026, 11, 3, 6, 0, 0, 0, zone), zone)
	s.True(ok)
	s.True(time.Date(2026, 11, 3, 9, 0, 0, 0, zone).Equal(next))
}

func (s *HoursSuite) TestAWindowCrossingMidnightIsRefused() {
	h := s.onlyWindows(weekdays, Window{"22:00", "02:00"})
	s.ErrorContains(h.Validate(), "before it ends")
}

func (s *HoursSuite) TestAQuarterPastIsAllowedAndTenPastIsNot() {
	s.NoError(s.onlyWindows(weekdays, Window{"09:15", "17:45"}).Validate())
	s.ErrorContains(s.onlyWindows(weekdays, Window{"09:10", "17:00"}).Validate(), "15 minutes")
	s.ErrorContains(s.onlyWindows(weekdays, Window{"09:00", "17:10"}).Validate(), "15 minutes")
}

func (s *HoursSuite) TestWindowsEndingAtMidnightAreAllowed() {
	h := s.onlyWindows([]string{"monday"}, Window{"00:00", "24:00"})
	s.NoError(h.Validate())
	zone := s.zone("UTC")
	s.True(h.Open(time.Date(2026, 11, 2, 23, 58, 0, 0, zone), zone))
	s.False(h.Open(time.Date(2026, 11, 2, 23, 59, 0, 0, zone), zone))
	s.True(h.Open(time.Date(2026, 11, 2, 0, 0, 0, 0, zone), zone))
	s.False(h.Open(time.Date(2026, 11, 3, 12, 0, 0, 0, zone), zone))

	s.ErrorContains(s.onlyWindows(weekdays, Window{"09:00", "24:15"}).Validate(), "24:00")
	s.ErrorContains(s.onlyWindows(weekdays, Window{"24:00", "24:00"}).Validate(), "before it ends")
}

func (s *HoursSuite) TestLegacyHoursAreAlwaysOpen() {
	h := LegacyCallingHours()
	s.NoError(h.Validate())
	zone := h.Zone("America/Chicago", "+13125550177")
	s.Equal("UTC", zone.String())

	for _, at := range []time.Time{
		time.Date(2026, 11, 3, 3, 0, 0, 0, time.UTC),
		time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC),
		time.Date(2026, 12, 25, 23, 58, 0, 0, time.UTC),
	} {
		s.True(h.Open(at, zone), at.String())
	}
}

func (s *HoursSuite) TestTheLegacyHoursAreWhatTheMigrationWrote() {
	raw, err := json.Marshal(LegacyCallingHours())
	s.Require().NoError(err)

	day := `[{"start":"00:00","end":"24:00"}]`
	s.JSONEq(`{"windows":{"monday":`+day+`,"tuesday":`+day+`,"wednesday":`+day+`,"thursday":`+day+`,"friday":`+day+`,"saturday":`+day+`,"sunday":`+day+`},"time_zone":{"mode":"fixed","fallback":"UTC","fixed":"UTC"},"excluded_dates":[]}`, string(raw))
}

func (s *HoursSuite) TestTheDefaultHoursAreWeekdaysNineToSixInTheContactsZone() {
	h := DefaultCallingHours()
	s.NoError(h.Validate())
	s.Equal(ZoneContact, h.TimeZone.Mode)
	s.Equal("America/Los_Angeles", h.TimeZone.Fallback)
	s.Equal([]Window{{"09:00", "18:00"}}, h.Windows["monday"])
	s.Empty(h.Windows["saturday"])
	s.Empty(h.Windows["sunday"])

	raw, err := json.Marshal(h)
	s.Require().NoError(err)
	s.Contains(string(raw), `"excluded_dates":[]`)
}

func (s *HoursSuite) TestAWeekWithNoWindowIsRefused() {
	h := DefaultCallingHours()
	h.Windows = map[string][]Window{"monday": {}}
	s.ErrorContains(h.Validate(), "at least one window")
	h.Windows = nil
	s.ErrorContains(h.Validate(), "at least one window")
}

func (s *HoursSuite) TestOverlappingWindowsAreRefusedButTouchingOnesAreNot() {
	s.ErrorContains(s.onlyWindows(weekdays, Window{"09:00", "12:00"}, Window{"11:45", "17:00"}).Validate(), "overlap")
	s.NoError(s.onlyWindows(weekdays, Window{"09:00", "12:00"}, Window{"12:00", "17:00"}).Validate())
}

func (s *HoursSuite) TestBadDaysTimesZonesAndDatesAreRefused() {
	h := DefaultCallingHours()
	h.Windows["funday"] = []Window{{"09:00", "10:00"}}
	s.ErrorContains(h.Validate(), "funday")

	for _, bad := range []Window{{"9:00", "10:00"}, {"09:00", "25:00"}, {"09:60", "10:00"}, {"", "10:00"}, {"09:00", "9am"}} {
		s.Error(s.onlyWindows(weekdays, bad).Validate(), "%v", bad)
	}

	h = DefaultCallingHours()
	h.TimeZone.Fallback = "Mars/Olympus"
	s.ErrorContains(h.Validate(), "Mars/Olympus")

	h = DefaultCallingHours()
	h.TimeZone = TimeZone{Mode: ZoneFixed, Fallback: "UTC"}
	s.ErrorContains(h.Validate(), "fixed")

	h = DefaultCallingHours()
	h.TimeZone.Mode = "sometimes"
	s.ErrorContains(h.Validate(), "mode")

	h = DefaultCallingHours()
	h.ExcludedDates = []string{"2026-02-30"}
	s.ErrorContains(h.Validate(), "2026-02-30")
	h.ExcludedDates = []string{"26-11-26"}
	s.Error(h.Validate())
}

func (s *HoursSuite) TestParseRefusesUnknownFieldsAndTrailingData() {
	raw, err := json.Marshal(DefaultCallingHours())
	s.Require().NoError(err)

	got, err := ParseCallingHours(raw)
	s.Require().NoError(err)
	s.Equal(DefaultCallingHours(), got)

	_, err = ParseCallingHours([]byte(`{"windows":{},"time_zone":{"mode":"fixed","fixed":"UTC"},"excluded_dates":[],"extra":1}`))
	s.Error(err)
	_, err = ParseCallingHours(append(raw, []byte(` {}`)...))
	s.Error(err)
	_, err = ParseCallingHours([]byte(`nope`))
	s.Error(err)
}

func (s *HoursSuite) TestNextLandsOnTheRightLocalNineAcrossTheFallBack() {
	h := DefaultCallingHours()
	zone := s.zone("America/New_York")

	// Fri 2026-10-30 19:00 EDT. Monday 2026-11-02 09:00 is EST, after the 11-01 fall-back.
	next, ok := h.Next(time.Date(2026, 10, 30, 19, 0, 0, 0, zone), zone)
	s.True(ok)
	s.Equal(time.Date(2026, 11, 2, 14, 0, 0, 0, time.UTC), next.UTC())
	s.Equal(9, next.In(zone).Hour())

	// Sunday 2026-11-01 is 25 hours long; Monday's 09:00 is still the answer from late Sunday.
	next, ok = h.Next(time.Date(2026, 11, 1, 23, 0, 0, 0, zone), zone)
	s.True(ok)
	s.Equal(time.Date(2026, 11, 2, 14, 0, 0, 0, time.UTC), next.UTC())
}

func (s *HoursSuite) TestNextLandsOnTheRightLocalNineAcrossTheSpringForward() {
	h := DefaultCallingHours()
	zone := s.zone("America/New_York")

	// Fri 2026-03-06 19:00 EST. Monday 2026-03-09 09:00 is EDT, after the 03-08 spring-forward.
	next, ok := h.Next(time.Date(2026, 3, 6, 19, 0, 0, 0, zone), zone)
	s.True(ok)
	s.Equal(time.Date(2026, 3, 9, 13, 0, 0, 0, time.UTC), next.UTC())
	s.Equal(9, next.In(zone).Hour())
}

func (s *HoursSuite) TestAWindowOnTheDayOfTheChangeUsesTheClockOnTheWall() {
	zone := s.zone("America/New_York")
	h := s.onlyWindows([]string{"sunday"}, Window{"09:00", "18:00"})

	// Sunday 2026-11-01, after the fall-back: 09:00 is EST (14:00 UTC).
	s.False(h.Open(time.Date(2026, 11, 1, 13, 59, 0, 0, time.UTC), zone))
	s.True(h.Open(time.Date(2026, 11, 1, 14, 0, 0, 0, time.UTC), zone))

	// Sunday 2026-03-08, after the spring-forward: 09:00 is EDT (13:00 UTC).
	s.False(h.Open(time.Date(2026, 3, 8, 12, 59, 0, 0, time.UTC), zone))
	s.True(h.Open(time.Date(2026, 3, 8, 13, 0, 0, 0, time.UTC), zone))

	next, ok := h.Next(time.Date(2026, 3, 7, 12, 0, 0, 0, time.UTC), zone)
	s.True(ok)
	s.Equal(time.Date(2026, 3, 8, 13, 0, 0, 0, time.UTC), next.UTC())
}

func (s *HoursSuite) TestAWindowInsideTheRepeatedHourOpensAtTheFirstPass() {
	zone := s.zone("America/New_York")
	h := s.onlyWindows([]string{"sunday"}, Window{"01:00", "02:00"})

	// 01:00 happens twice on 2026-11-01 (EDT, then EST); both passes are open.
	s.True(h.Open(time.Date(2026, 11, 1, 5, 0, 0, 0, time.UTC), zone))
	s.True(h.Open(time.Date(2026, 11, 1, 6, 0, 0, 0, time.UTC), zone))

	next, ok := h.Next(time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC), zone)
	s.True(ok)
	s.Equal(time.Date(2026, 11, 1, 5, 0, 0, 0, time.UTC), next.UTC())
}

func (s *HoursSuite) TestNextGivesUpAfterAYearAndAHalf() {
	h := s.onlyWindows([]string{"monday"}, Window{"09:00", "10:00"})
	h.ExcludedDates = nil
	zone := s.zone("UTC")
	after := time.Date(2026, 11, 2, 12, 0, 0, 0, zone)
	for d := after; d.Before(after.AddDate(0, 0, 420)); d = d.AddDate(0, 0, 7) {
		h.ExcludedDates = append(h.ExcludedDates, d.AddDate(0, 0, 7).Format("2006-01-02"))
	}
	_, ok := h.Next(after, zone)
	s.False(ok)
}
