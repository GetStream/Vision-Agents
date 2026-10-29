package agent

import "time"

// clockLine tells the model the date and the hour, in the user's zone when it is known and
// in UTC always, so "tomorrow" or "next Wednesday at 8" resolves without asking the user what
// day it is. It is accurate to the hour rather than the minute because it is part of every
// turn's instructions, ahead of the history: a line that changed every minute would cost the
// provider's prompt cache that history on most turns.
func clockLine(now time.Time, zone *time.Location) string {
	hour := now.Truncate(time.Hour)
	utc := "between " + hour.UTC().Format("15:04") + " and " + hour.Add(time.Hour).UTC().Format("15:04") + " UTC"
	if zone == nil {
		return "Today is " + now.UTC().Format("Monday, 2 January 2006") + " in UTC, " + utc + ". " +
			"The user's timezone is not known: ask for it when a time of day depends on it."
	}
	local := hour.In(zone)
	return "Today is " + now.In(zone).Format("Monday, 2 January 2006") + " for the user (" + zone.String() + "), " +
		"between " + local.Format("15:04") + " and " + local.Add(time.Hour).Format("15:04") + " there, " + utc + "."
}
