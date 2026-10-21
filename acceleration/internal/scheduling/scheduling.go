// Package scheduling parses, validates and expands the schedules of scheduled
// tasks: a wall-clock time in an IANA zone, optionally repeated by an RRULE
// subset. It knows nothing of the store or of sessions; callers hand it a
// resolved zone and a clock.
package scheduling

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	// Zone data is embedded because the router's image cannot be relied on to
	// carry it: a zone the OS has never heard of must still resolve.
	_ "time/tzdata"

	"github.com/teambition/rrule-go"
)

// Error is a refusal a caller can act on: its Code is answered over the API
// verbatim and told to the model in a tool result.
type Error struct {
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Message }

func fail(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// The codes a schedule can be refused with.
const (
	CodeScheduleInPast   = "schedule_in_past"
	CodeScheduleTooFar   = "schedule_too_far"
	CodeInvalidTimezone  = "invalid_timezone"
	CodeTimezoneUnknown  = "timezone_unknown"
	CodeInvalidRepeat    = "invalid_repeat"
	CodeIntervalTooShort = "interval_too_short"
	CodeInvalidSchedule  = "invalid_schedule"
)

// MinMove is the soonest a moved run (a snooze, or a next set by hand) may
// land from now, unless it lands no earlier than the run it moves. Without it
// a caller re-arming a task a minute ahead, again and again, buys a turn a
// minute per task: 25 tasks are 25 turns a minute for one user. It is its own
// constant; the doc of the next field of UpdateScheduledTaskRequest
// (api/scheduled_tasks.go) says the same 15 minutes in words.
const MinMove = 15 * time.Minute

// CheckMove refuses a moved run that is both earlier than the run it moves
// (current, nil when none is pending) and sooner than MinMove from now. A move
// to or past the current run cannot raise how often the task runs. A current
// run already due does not count: moving it is a re-arm from now. Now is read
// to the second, as a schedule holds its times, so in PT15M passes.
func CheckMove(next time.Time, current *time.Time, now time.Time) error {
	now = now.Truncate(time.Second)
	if current != nil && current.After(now) && !next.Before(*current) {
		return nil
	}
	if next.Sub(now) < MinMove {
		return fail(CodeIntervalTooShort, "a run can be moved to %d minutes from now at the soonest, unless it moves to or past the run it replaces", int(MinMove/time.Minute))
	}
	return nil
}

// MaxActivePerUser is the default cap on a user's active tasks, used when the
// agent config does not set one.
const MaxActivePerUser = 25

const (
	maxCount = 1000
	// maxInHours is a leap year: the most a year ahead can be.
	maxInHours = 366 * 24
	maxSteps   = 20000
)

// Input is a schedule as a caller wrote it, before anything is resolved. The
// zone is the one already chosen for the task (see ResolveZone); Parse refuses
// an empty or unknown one rather than guessing.
type Input struct {
	At       string
	In       string
	Repeat   string
	Until    string
	Timezone string
}

// Schedule is a validated schedule. StartLocal is the first occurrence as the
// user's clock reads it, held as a time in UTC so its fields are the wall
// clock's; Timezone says which clock. Repeat is the normalised RRULE, empty
// for a task that runs once. Until is the last local date it repeats on, in
// the same UTC-held form, or the zero time.
type Schedule struct {
	StartLocal time.Time
	Timezone   string
	Repeat     string
	Until      time.Time

	location *time.Location
	rule     *repeatRule
	// start is the instant StartLocal names when the caller named one (an in,
	// or an at with an offset), so a reading in a fall-back hour's second
	// pass is not resolved to its first; zero for a plain wall-clock reading.
	start time.Time
}

type repeatRule struct {
	freq       string
	interval   int
	byDay      []weekdaySpec
	byMonthDay []int
	count      int
	rrule      *rrule.RRule
}

type weekdaySpec struct {
	day string
	nth int
}

var weekdayByName = map[string]rrule.Weekday{
	"MO": rrule.MO, "TU": rrule.TU, "WE": rrule.WE, "TH": rrule.TH,
	"FR": rrule.FR, "SA": rrule.SA, "SU": rrule.SU,
}

var weekdayNames = map[string]string{
	"MO": "Mon", "TU": "Tue", "WE": "Wed", "TH": "Thu",
	"FR": "Fri", "SA": "Sat", "SU": "Sun",
}

var weekdayOrder = []string{"MO", "TU", "WE", "TH", "FR", "SA", "SU"}

// Location returns the schedule's zone.
func (s *Schedule) Location() *time.Location { return s.location }

// LoadZone resolves an IANA zone name, or refuses it. "Local" is refused: Go
// reads it as the router's own zone, which is nobody's.
func LoadZone(name string) (*time.Location, error) {
	loc, err := time.LoadLocation(strings.TrimSpace(name))
	if err != nil || loc == time.Local {
		return nil, fail(CodeInvalidTimezone, "%q is not a timezone I know; use an IANA name such as Europe/Amsterdam", name)
	}
	return loc, nil
}

// ResolveZone picks the zone for a task, first match wins: what the caller
// named for this task, the session's, the user's stored one, then the agent
// config's default. None of them is a refusal to ask rather than a guess.
func ResolveZone(argument, session, stored, configDefault string) (string, error) {
	for _, candidate := range []string{argument, session, stored, configDefault} {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		if _, err := LoadZone(candidate); err != nil {
			return "", err
		}
		return candidate, nil
	}
	return "", fail(CodeTimezoneUnknown, "the user's timezone is not known; ask which city or timezone they are in, then try again naming it")
}

// Parse validates input against now and returns the schedule it describes.
func Parse(input Input, now time.Time) (*Schedule, error) {
	zone := strings.TrimSpace(input.Timezone)
	if zone == "" {
		return nil, fail(CodeTimezoneUnknown, "no timezone was given and none is known for the user; ask which city or timezone they are in")
	}
	loc, err := LoadZone(zone)
	if err != nil {
		return nil, err
	}

	at, in := strings.TrimSpace(input.At), strings.TrimSpace(input.In)
	if (at == "") == (in == "") {
		return nil, fail(CodeInvalidSchedule, "give exactly one of at (a local date and time) and in (how long from now)")
	}

	rule, err := parseRepeat(input.Repeat)
	if err != nil {
		return nil, err
	}
	if err := rule.sayable(); err != nil {
		return nil, err
	}

	var startLocal, start time.Time
	if in != "" {
		days, clock, err := parseDuration(in)
		if err != nil {
			return nil, err
		}
		// Days are calendar days (RFC 5545 3.3.6 calls them nominal): P1D is
		// the same clock reading tomorrow, 23 or 25 hours away across a
		// daylight-saving change. Hours and less are exact.
		from := now
		if days > 0 {
			from = ResolveLocal(wallOf(now.In(loc)).AddDate(0, 0, days), loc)
		}
		start = from.Add(clock).Truncate(time.Second).UTC()
		startLocal = wallOf(start.In(loc))
	} else {
		startLocal, start, err = parseAt(at, loc, now, rule == nil)
		if err != nil {
			return nil, err
		}
	}

	until, err := parseUntil(input.Until)
	if err != nil {
		return nil, err
	}
	if !until.IsZero() && rule == nil {
		return nil, fail(CodeInvalidSchedule, "until only makes sense with repeat; a task that runs once runs at at")
	}
	if !until.IsZero() && dateOf(startLocal).After(until) {
		return nil, fail(CodeInvalidSchedule, "until %s is before the first occurrence %s", until.Format("2006-01-02"), startLocal.Format("2006-01-02"))
	}

	schedule := &Schedule{
		StartLocal: startLocal,
		Timezone:   zone,
		Until:      until,
		location:   loc,
		rule:       rule,
		start:      start,
	}
	if rule != nil {
		built, err := rule.build(startLocal)
		if err != nil {
			return nil, err
		}
		rule.rrule = built
		schedule.Repeat = rule.String()
	}

	first, ok := schedule.First()
	if !ok {
		return nil, fail(CodeScheduleInPast, "that schedule has no occurrence left; its last one has passed")
	}
	if rule == nil {
		if !first.After(now) {
			return nil, fail(CodeScheduleInPast, "that time has passed; give a time in the future")
		}
		if first.After(now.AddDate(1, 0, 0)) {
			return nil, fail(CodeScheduleTooFar, "that is more than a year away; a task can be at most a year ahead")
		}
	}
	return schedule, nil
}

// FromStored rebuilds a schedule from what the store holds, without the
// future checks Parse makes: a stored task may legitimately be due.
func FromStored(startLocal time.Time, zone, repeat string, until time.Time) (*Schedule, error) {
	loc, err := LoadZone(zone)
	if err != nil {
		return nil, err
	}
	schedule := &Schedule{
		StartLocal: wallOf(startLocal),
		Timezone:   zone,
		Until:      until,
		location:   loc,
	}
	if strings.TrimSpace(repeat) != "" {
		rule, err := parseRepeat(repeat)
		if err != nil {
			return nil, err
		}
		built, err := rule.build(schedule.StartLocal)
		if err != nil {
			return nil, err
		}
		rule.rrule = built
		schedule.rule = rule
		schedule.Repeat = rule.String()
	}
	return schedule, nil
}

// First is the schedule's first occurrence as an instant.
func (s *Schedule) First() (time.Time, bool) {
	next := s.occurrences()
	first, ok := next()
	return first, ok
}

// Next is the first occurrence after the given instant.
func (s *Schedule) Next(after time.Time) (time.Time, bool) {
	next := s.occurrences()
	for {
		occurrence, ok := next()
		if !ok {
			return time.Time{}, false
		}
		if occurrence.After(after) {
			return occurrence, true
		}
	}
}

// Upcoming lists the next n occurrences after the given instant.
func (s *Schedule) Upcoming(after time.Time, n int) []time.Time {
	out := []time.Time{}
	next := s.occurrences()
	for len(out) < n {
		occurrence, ok := next()
		if !ok {
			return out
		}
		if occurrence.After(after) {
			out = append(out, occurrence)
		}
	}
	return out
}

// occurrences returns a pull iterator over the schedule's instants, earliest
// first. Recurrences expand in local time: each generated date carries
// StartLocal's time of day and is then resolved in the zone, so a 09:00 task
// stays at 09:00 across a daylight-saving change.
func (s *Schedule) occurrences() func() (time.Time, bool) {
	if s.rule == nil {
		done := false
		return func() (time.Time, bool) {
			if done {
				return time.Time{}, false
			}
			done = true
			return s.resolve(s.StartLocal), true
		}
	}
	iterator := s.rule.rrule.Iterator()
	steps := 0
	return func() (time.Time, bool) {
		for {
			steps++
			if steps > maxSteps {
				return time.Time{}, false
			}
			wall, ok := iterator()
			if !ok {
				return time.Time{}, false
			}
			if wall.Before(s.StartLocal) {
				continue
			}
			if !s.Until.IsZero() && dateOf(wall).After(s.Until) {
				return time.Time{}, false
			}
			return s.resolve(wall), true
		}
	}
}

// resolve is the instant a wall-clock occurrence names: the one the caller
// named for the start, else ResolveLocal's.
func (s *Schedule) resolve(wall time.Time) time.Time {
	if !s.start.IsZero() && wall.Equal(s.StartLocal) {
		return s.start
	}
	return ResolveLocal(wall, s.location)
}

// parseAt reads at as a wall-clock reading and, when it carries an offset,
// the instant it names.
func parseAt(at string, loc *time.Location, now time.Time, oneShot bool) (time.Time, time.Time, error) {
	if instant, err := time.Parse(time.RFC3339, at); err == nil {
		instant = instant.Truncate(time.Second).UTC()
		return wallOf(instant.In(loc)), instant, nil
	}
	for _, layout := range []string{"2006-01-02T15:04:05", "2006-01-02T15:04", "2006-01-02 15:04:05", "2006-01-02 15:04", "2006-01-02"} {
		if parsed, err := time.ParseInLocation(layout, at, time.UTC); err == nil {
			return wallOf(parsed), time.Time{}, nil
		}
	}
	for _, layout := range []string{"15:04:05", "15:04"} {
		if parsed, err := time.ParseInLocation(layout, at, time.UTC); err == nil {
			local := now.In(loc)
			start := time.Date(local.Year(), local.Month(), local.Day(),
				parsed.Hour(), parsed.Minute(), parsed.Second(), 0, time.UTC)
			if oneShot && !ResolveLocal(start, loc).After(now) {
				start = start.AddDate(0, 0, 1)
			}
			return start, time.Time{}, nil
		}
	}
	return time.Time{}, time.Time{}, fail(CodeInvalidSchedule, "at %q is not a date and time I can read; use 2026-10-20T09:00, 09:00 for the next one, or a time with an offset", at)
}

func parseUntil(until string) (time.Time, error) {
	until = strings.TrimSpace(until)
	if until == "" {
		return time.Time{}, nil
	}
	parsed, err := time.ParseInLocation("2006-01-02", until, time.UTC)
	if err != nil {
		return time.Time{}, fail(CodeInvalidSchedule, "until %q is not a date; use 2026-12-31", until)
	}
	return wallOf(parsed), nil
}

var durationPattern = regexp.MustCompile(`^P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$`)

// parseDuration splits an ISO 8601 duration into its calendar days (weeks
// count as seven) and the exact time after them.
func parseDuration(in string) (int, time.Duration, error) {
	match := durationPattern.FindStringSubmatch(strings.TrimSpace(in))
	if match == nil {
		return 0, 0, fail(CodeInvalidSchedule, "in %q is not an ISO 8601 duration; use PT20M, PT2H or P1D", in)
	}
	part := func(text string) float64 {
		value, _ := strconv.ParseFloat(text, 64)
		return value
	}
	// A start past the one-year horizon Parse holds a one-shot to is refused
	// before the conversion below can overflow.
	hours := (part(match[1])*7+part(match[2]))*24 + part(match[3]) + part(match[4])/60 + part(match[5])/3600
	if hours > maxInHours {
		return 0, 0, fail(CodeScheduleTooFar, "in %q is more than a year away; a task can be at most a year ahead", in)
	}
	days := int(part(match[1]))*7 + int(part(match[2]))
	clock := time.Duration(part(match[3]))*time.Hour +
		time.Duration(part(match[4]))*time.Minute +
		time.Duration(part(match[5])*float64(time.Second))
	if days <= 0 && clock <= 0 {
		return 0, 0, fail(CodeInvalidSchedule, "in must be in the future; %q is not", in)
	}
	return days, clock, nil
}

func parseRepeat(repeat string) (*repeatRule, error) {
	repeat = strings.TrimSpace(repeat)
	if repeat == "" {
		return nil, nil
	}
	rule := &repeatRule{interval: 1}
	seen := map[string]bool{}
	for part := range strings.SplitSeq(repeat, ";") {
		key, value, found := strings.Cut(part, "=")
		if !found {
			return nil, fail(CodeInvalidRepeat, "repeat part %q is not KEY=value", part)
		}
		key = strings.ToUpper(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		if seen[key] {
			return nil, fail(CodeInvalidRepeat, "repeat names %s twice", key)
		}
		seen[key] = true
		switch key {
		case "FREQ":
			rule.freq = strings.ToUpper(value)
			switch rule.freq {
			case "DAILY", "WEEKLY", "MONTHLY", "YEARLY":
			default:
				return nil, fail(CodeInvalidRepeat, "FREQ=%s is not one of DAILY, WEEKLY, MONTHLY or YEARLY", value)
			}
		case "INTERVAL":
			interval, err := strconv.Atoi(value)
			if err != nil || interval < 1 {
				return nil, fail(CodeInvalidRepeat, "INTERVAL=%s is not a whole number from 1", value)
			}
			rule.interval = interval
		case "COUNT":
			count, err := strconv.Atoi(value)
			if err != nil || count < 1 || count > maxCount {
				return nil, fail(CodeInvalidRepeat, "COUNT=%s is not a whole number from 1 to %d", value, maxCount)
			}
			rule.count = count
		case "BYDAY":
			for token := range strings.SplitSeq(value, ",") {
				spec, err := parseWeekday(token)
				if err != nil {
					return nil, err
				}
				rule.byDay = append(rule.byDay, spec)
			}
		case "BYMONTHDAY":
			for token := range strings.SplitSeq(value, ",") {
				day, err := strconv.Atoi(strings.TrimSpace(token))
				if err != nil || day == 0 || day < -31 || day > 31 {
					return nil, fail(CodeInvalidRepeat, "BYMONTHDAY=%s is not a day of the month from -31 to 31", token)
				}
				rule.byMonthDay = append(rule.byMonthDay, day)
			}
		case "DTSTART", "UNTIL":
			return nil, fail(CodeInvalidRepeat, "repeat must not carry %s; the start is at and the end is until", key)
		default:
			return nil, fail(CodeInvalidRepeat, "repeat part %s is not supported; use FREQ, INTERVAL, BYDAY, BYMONTHDAY and COUNT", key)
		}
	}
	if rule.freq == "" {
		return nil, fail(CodeInvalidRepeat, "repeat has no FREQ; use FREQ=DAILY, FREQ=WEEKLY;BYDAY=MO,WE,FR or FREQ=MONTHLY;BYMONTHDAY=1")
	}
	return rule, nil
}

var weekdayPattern = regexp.MustCompile(`^([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)$`)

func parseWeekday(token string) (weekdaySpec, error) {
	match := weekdayPattern.FindStringSubmatch(strings.ToUpper(strings.TrimSpace(token)))
	if match == nil {
		return weekdaySpec{}, fail(CodeInvalidRepeat, "BYDAY entry %q is not a weekday such as MO or 1FR", token)
	}
	spec := weekdaySpec{day: match[2]}
	if match[1] != "" {
		nth, _ := strconv.Atoi(match[1])
		if nth == 0 || nth < -53 || nth > 53 {
			return weekdaySpec{}, fail(CodeInvalidRepeat, "BYDAY entry %q has an ordinal out of range", token)
		}
		spec.nth = nth
	}
	return spec, nil
}

func (r *repeatRule) build(startLocal time.Time) (*rrule.RRule, error) {
	option := rrule.ROption{
		Dtstart:    startLocal,
		Interval:   r.interval,
		Count:      r.count,
		Bymonthday: r.byMonthDay,
	}
	switch r.freq {
	case "DAILY":
		option.Freq = rrule.DAILY
	case "WEEKLY":
		option.Freq = rrule.WEEKLY
	case "MONTHLY":
		option.Freq = rrule.MONTHLY
	case "YEARLY":
		option.Freq = rrule.YEARLY
	}
	for _, spec := range r.byDay {
		day := weekdayByName[spec.day]
		if spec.nth != 0 {
			day = day.Nth(spec.nth)
		}
		option.Byweekday = append(option.Byweekday, day)
	}
	built, err := rrule.NewRRule(option)
	if err != nil {
		return nil, fail(CodeInvalidRepeat, "that repeat cannot be expanded: %s", err)
	}
	return built, nil
}

// String is the normalised RRULE the store keeps.
func (r *repeatRule) String() string {
	parts := []string{"FREQ=" + r.freq}
	if r.interval != 1 {
		parts = append(parts, "INTERVAL="+strconv.Itoa(r.interval))
	}
	if len(r.byDay) > 0 {
		days := make([]string, 0, len(r.byDay))
		for _, spec := range r.byDay {
			if spec.nth != 0 {
				days = append(days, strconv.Itoa(spec.nth)+spec.day)
			} else {
				days = append(days, spec.day)
			}
		}
		parts = append(parts, "BYDAY="+strings.Join(days, ","))
	}
	if len(r.byMonthDay) > 0 {
		days := make([]string, 0, len(r.byMonthDay))
		for _, day := range r.byMonthDay {
			days = append(days, strconv.Itoa(day))
		}
		parts = append(parts, "BYMONTHDAY="+strings.Join(days, ","))
	}
	if r.count > 0 {
		parts = append(parts, "COUNT="+strconv.Itoa(r.count))
	}
	return strings.Join(parts, ";")
}

// wallOf holds a time's clock reading as a time in UTC, so a wall time can be
// stored, compared and expanded without a zone attached.
func wallOf(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second(), 0, time.UTC)
}

func dateOf(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// FormatLocal renders an instant as the user reads it: weekday, date, time
// and the zone's own abbreviation.
func FormatLocal(t time.Time, loc *time.Location) string {
	return t.In(loc).Format("Mon 2 Jan 2006 15:04 MST")
}

// SortedDays orders weekday codes the way a week reads.
func SortedDays(days []string) []string {
	out := slices.Clone(days)
	slices.SortFunc(out, func(a, b string) int {
		return slices.Index(weekdayOrder, a) - slices.Index(weekdayOrder, b)
	})
	return out
}
