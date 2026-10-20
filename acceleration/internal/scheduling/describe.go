package scheduling

import (
	"slices"
	"strconv"
	"strings"
	"time"
)

// Describe says the schedule in one sentence a person, or the model reading a
// task back, understands: "Every weekday at 09:00 (Europe/Amsterdam) until
// Fri 18 Dec 2026", or "Once, Wed 21 Oct 2026 at 18:00 (Europe/Amsterdam)".
func (s *Schedule) Describe() string {
	at := s.StartLocal.Format("15:04")
	if s.rule == nil {
		return "Once, " + s.StartLocal.Format("Mon 2 Jan 2006") + " at " + at + " (" + s.Timezone + ")"
	}
	text := s.rule.phrase(s.StartLocal) + " at " + at + " (" + s.Timezone + ")"
	if !s.Until.IsZero() {
		text += " until " + s.Until.Format("Mon 2 Jan 2006")
	}
	if s.rule.count > 0 {
		text += ", " + strconv.Itoa(s.rule.count) + " times in all"
	}
	return text
}

// supportedRepeats ends every refusal sayable makes, so the caller learns what
// to send instead.
const supportedRepeats = "supported: FREQ=DAILY; FREQ=WEEKLY, alone or with BYDAY=MO,WE,FR; " +
	"FREQ=MONTHLY, alone, with BYMONTHDAY=1,15 or with one BYDAY such as 1MO or -1FR; " +
	"FREQ=YEARLY alone, on the start's date; INTERVAL with any but YEARLY; COUNT with any"

// sayable refuses a repeat that phrase would misstate, so what a task says it
// does is what fires. Parse holds new rules to it; FromStored does not, so a
// task stored before a shape was refused keeps running.
func (r *repeatRule) sayable() error {
	if r == nil {
		return nil
	}
	switch {
	case len(r.byMonthDay) > 0 && r.freq != "MONTHLY":
		return fail(CodeInvalidRepeat, "BYMONTHDAY with FREQ=%s is not supported; %s", r.freq, supportedRepeats)
	case len(r.byDay) > 0 && (r.freq == "DAILY" || r.freq == "YEARLY"):
		return fail(CodeInvalidRepeat, "BYDAY with FREQ=%s is not supported; %s", r.freq, supportedRepeats)
	case r.freq == "YEARLY" && r.interval > 1:
		return fail(CodeInvalidRepeat, "INTERVAL with FREQ=YEARLY is not supported; %s", supportedRepeats)
	case len(r.byDay) > 0 && len(r.byMonthDay) > 0:
		return fail(CodeInvalidRepeat, "BYDAY with BYMONTHDAY is not supported; %s", supportedRepeats)
	case r.freq == "MONTHLY" && len(r.byDay) > 1:
		return fail(CodeInvalidRepeat, "FREQ=MONTHLY with more than one BYDAY is not supported; %s", supportedRepeats)
	}
	for i, spec := range r.byDay {
		switch {
		case r.freq == "MONTHLY" && spec.nth == 0:
			return fail(CodeInvalidRepeat, "BYDAY=%s with FREQ=MONTHLY would mean every %s in the month; give an ordinal such as 1%s for the first one; %s", spec.day, weekdayNames[spec.day], spec.day, supportedRepeats)
		case r.freq == "WEEKLY" && spec.nth != 0:
			return fail(CodeInvalidRepeat, "BYDAY=%d%s is only meaningful with FREQ=MONTHLY; %s", spec.nth, spec.day, supportedRepeats)
		// ordinalName names only these.
		case spec.nth < -2 || spec.nth > 5:
			return fail(CodeInvalidRepeat, "BYDAY=%d%s is not supported; the ordinal is 1 to 5, -1 or -2; %s", spec.nth, spec.day, supportedRepeats)
		case slices.ContainsFunc(r.byDay[:i], func(earlier weekdaySpec) bool { return earlier.day == spec.day }):
			return fail(CodeInvalidRepeat, "BYDAY names %s twice; %s", spec.day, supportedRepeats)
		}
	}
	for i, day := range r.byMonthDay {
		switch {
		// monthDayName names only -1 and -2 among the negative days.
		case day < -2:
			return fail(CodeInvalidRepeat, "BYMONTHDAY=%d is not supported; the days counted from the end are -1 and -2; %s", day, supportedRepeats)
		case slices.Contains(r.byMonthDay[:i], day):
			return fail(CodeInvalidRepeat, "BYMONTHDAY names %d twice; %s", day, supportedRepeats)
		}
	}
	return nil
}

func (r *repeatRule) phrase(start time.Time) string {
	every := ""
	if r.interval > 1 {
		every = "Every " + strconv.Itoa(r.interval) + " "
	}
	switch r.freq {
	case "DAILY":
		if r.interval == 1 {
			return "Daily"
		}
		return every + "days"
	case "WEEKLY":
		days := r.dayNames()
		if r.interval == 1 {
			if days == "Mon, Tue, Wed, Thu, Fri" {
				return "Every weekday"
			}
			if days == "" {
				return "Weekly"
			}
			return "Weekly on " + days
		}
		if days == "" {
			return every + "weeks"
		}
		return every + "weeks on " + days
	case "MONTHLY":
		if len(r.byMonthDay) > 0 {
			days := monthDays(r.byMonthDay)
			if r.interval == 1 {
				return "Monthly on " + days
			}
			return every + "months on " + days
		}
		if len(r.byDay) > 0 {
			if r.interval == 1 {
				return "Monthly on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day]
			}
			return every + "months on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day]
		}
		if start.Day() > 28 {
			days := monthDays([]int{start.Day()})
			if r.interval == 1 {
				return "Monthly on " + days
			}
			return every + "months on " + days
		}
		if r.interval == 1 {
			return "Monthly"
		}
		return every + "months"
	case "YEARLY":
		if len(r.byDay) > 0 && r.byDay[0].nth != 0 {
			return "Yearly on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day] + " of " + start.Format("January")
		}
		if start.Month() == time.February && start.Day() == 29 {
			return "Yearly on 29 February (leap years only)"
		}
		return "Yearly on " + start.Format("2 January")
	}
	return "Repeatedly"
}

// monthDays says days of the month: "day 1, 15" when all count from the start,
// "day 1 and the last day" when some count from the end. A day past the 28th
// adds that it is skipped in months without it.
func monthDays(days []int) string {
	fromEnd := slices.ContainsFunc(days, func(day int) bool { return day < 0 })
	names := make([]string, 0, len(days))
	var short []string
	for _, day := range days {
		switch {
		case day == -1:
			names = append(names, "the last day")
		case day == -2:
			names = append(names, "the second-to-last day")
		case fromEnd:
			names = append(names, "day "+strconv.Itoa(day))
		default:
			names = append(names, strconv.Itoa(day))
		}
		if day > 28 {
			short = append(short, strconv.Itoa(day))
		}
	}
	text := ""
	switch {
	case !fromEnd:
		text = "day " + strings.Join(names, ", ")
	case len(names) == 1:
		text = names[0]
	default:
		text = strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
	}
	switch len(short) {
	case 0:
	case 1:
		text += " (day " + short[0] + " is skipped in months without it)"
	default:
		text += " (days " + strings.Join(short[:len(short)-1], ", ") + " and " + short[len(short)-1] + " are skipped in months without them)"
	}
	return text
}

func (r *repeatRule) dayNames() string {
	if len(r.byDay) == 0 {
		return ""
	}
	codes := make([]string, 0, len(r.byDay))
	for _, spec := range r.byDay {
		codes = append(codes, spec.day)
	}
	names := make([]string, 0, len(codes))
	for _, code := range SortedDays(codes) {
		names = append(names, weekdayNames[code])
	}
	return strings.Join(names, ", ")
}

func ordinalName(nth int) string {
	switch nth {
	case 1:
		return "first"
	case 2:
		return "second"
	case 3:
		return "third"
	case 4:
		return "fourth"
	case 5:
		return "fifth"
	case -1:
		return "last"
	case -2:
		return "second to last"
	}
	return strconv.Itoa(nth) + "th"
}
