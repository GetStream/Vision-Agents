package phone

import "time"

func (s *PhoneSuite) TestANumbersMonthsBeginOnTheDayItWasBought() {
	bought := time.Date(2026, time.January, 31, 9, 30, 0, 0, time.UTC)
	day := func(month time.Month, d int) time.Time { return time.Date(2026, month, d, 9, 30, 0, 0, time.UTC) }

	s.Equal([]time.Time{bought}, MonthsBegun(bought, bought.Add(-time.Hour), bought), "the month it was bought in")
	s.Equal([]time.Time{day(time.February, 28)}, MonthsBegun(bought, day(time.February, 27), day(time.March, 1)),
		"a short month begins on its last day")
	s.Equal([]time.Time{day(time.March, 31)}, MonthsBegun(bought, day(time.March, 30), day(time.March, 31)))
	s.Equal([]time.Time{day(time.April, 30), day(time.May, 31)}, MonthsBegun(bought, day(time.April, 29), day(time.June, 1)),
		"a router down for weeks bills each month it missed")
	s.Empty(MonthsBegun(bought, day(time.March, 31), day(time.April, 29)), "between two months nothing begins")
}
