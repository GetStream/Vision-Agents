package scheduling

import (
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
			days := make([]string, 0, len(r.byMonthDay))
			for _, day := range r.byMonthDay {
				days = append(days, strconv.Itoa(day))
			}
			if r.interval == 1 {
				return "Monthly on day " + strings.Join(days, ", ")
			}
			return every + "months on day " + strings.Join(days, ", ")
		}
		if len(r.byDay) > 0 {
			if r.interval == 1 {
				return "Monthly on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day]
			}
			return every + "months on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day]
		}
		if r.interval == 1 {
			return "Monthly"
		}
		return every + "months"
	case "YEARLY":
		if len(r.byDay) > 0 && r.byDay[0].nth != 0 {
			return "Yearly on the " + ordinalName(r.byDay[0].nth) + " " + weekdayNames[r.byDay[0].day] + " of " + start.Format("January")
		}
		return "Yearly on " + start.Format("2 January")
	}
	return "Repeatedly"
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
