package scheduling

import (
	"strconv"
	"strings"
	"time"
)

// ClockMinutes reads a time of day as the minutes after midnight it names,
// "22:00" as 1320. It is the shape quiet hours are written in, and all that
// is checked: an hour of the day and a minute of the hour.
func ClockMinutes(text string) (int, bool) {
	hour, minute, found := strings.Cut(strings.TrimSpace(text), ":")
	if !found {
		return 0, false
	}
	hours, err := strconv.Atoi(hour)
	if err != nil || hours < 0 || hours > 23 {
		return 0, false
	}
	minutes, err := strconv.Atoi(minute)
	if err != nil || minutes < 0 || minutes > 59 || len(minute) != 2 {
		return 0, false
	}
	return hours*60 + minutes, true
}

// QuietHours reads a quiet-hours pair as minutes after midnight. Not ok is
// a boundary that does not read, or both reading the same, which would
// hold the whole day or nothing.
func QuietHours(start, end string) (from, to int, ok bool) {
	from, okFrom := ClockMinutes(start)
	to, okTo := ClockMinutes(end)
	if !okFrom || !okTo || from == to {
		return 0, 0, false
	}
	return from, to, true
}

// QuietEnd is when the quiet hours covering now end, read in the task's
// zone, and whether now is in them at all. The span from start to end wraps
// midnight when the end does not read after the start: "22:00" to "07:00"
// is the night. A span that does not read is no quiet hours. The end is a
// wall-clock reading first and an instant second, so a night the clocks
// move still ends when the user's clock says it does.
func QuietEnd(start, end string, loc *time.Location, now time.Time) (time.Time, bool) {
	from, to, ok := QuietHours(start, end)
	if !ok {
		return time.Time{}, false
	}
	local := now.In(loc)
	minutes := local.Hour()*60 + local.Minute()
	endDate := dateOf(local)
	switch {
	case from < to:
		if minutes < from || minutes >= to {
			return time.Time{}, false
		}
	case minutes >= from:
		// The night began today and ends tomorrow.
		endDate = endDate.Add(24 * time.Hour)
	case minutes >= to:
		return time.Time{}, false
	}
	return ResolveLocal(endDate.Add(time.Duration(to)*time.Minute), loc), true
}
