package campaign

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strconv"
	"time"
	_ "time/tzdata"

	"github.com/nyaruka/phonenumbers"
)

const (
	ZoneContact = "contact"
	ZoneFixed   = "fixed"
)

const (
	windowStepMinutes  = 15
	dayMinutes         = 24 * 60
	lookAheadDays      = 400
	unknownZone        = "Etc/Unknown"
	excludedDateLayout = "2006-01-02"
)

var (
	weekdayNames = [...]string{"sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"}
	clockTime    = regexp.MustCompile(`^(\d{2}):(\d{2})$`)
)

// CallingHours says when a campaign may start a call, in the contact's local time.
type CallingHours struct {
	Windows       map[string][]Window `json:"windows"` // keys monday..sunday; missing or empty = no calls that day
	TimeZone      TimeZone            `json:"time_zone"`
	ExcludedDates []string            `json:"excluded_dates"` // YYYY-MM-DD, the contact's local date
}

// Window is one stretch of a day, "HH:MM" on a 15-minute step. It cannot cross midnight.
type Window struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type TimeZone struct {
	Mode     string `json:"mode"`     // ZoneContact or ZoneFixed
	Fallback string `json:"fallback"` // IANA, used by contact mode
	Fixed    string `json:"fixed,omitempty"`
}

// DefaultCallingHours is what a new draft starts with: weekdays 09:00 to 18:00 where the contact is.
func DefaultCallingHours() CallingHours {
	day := []Window{{Start: "09:00", End: "18:00"}}
	return CallingHours{
		Windows: map[string][]Window{
			"monday": day, "tuesday": day, "wednesday": day, "thursday": day, "friday": day,
		},
		TimeZone:      TimeZone{Mode: ZoneContact, Fallback: "America/Los_Angeles"},
		ExcludedDates: []string{},
	}
}

// LegacyCallingHours is what campaigns made before calling hours got: always open.
// It must marshal as the migration wrote it, with "excluded_dates": [] and not null.
func LegacyCallingHours() CallingHours {
	windows := map[string][]Window{}
	for _, d := range weekdayNames {
		windows[d] = []Window{{Start: "00:00", End: "24:00"}}
	}
	return CallingHours{
		Windows:       windows,
		TimeZone:      TimeZone{Mode: ZoneFixed, Fallback: "UTC", Fixed: "UTC"},
		ExcludedDates: []string{},
	}
}

func ParseCallingHours(raw []byte) (CallingHours, error) {
	var h CallingHours
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&h); err != nil {
		return CallingHours{}, fmt.Errorf("campaign: calling hours: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return CallingHours{}, errors.New("campaign: calling hours: unexpected data after the JSON value")
	}
	return h, nil
}

// Validate checks the shape, which is what a PATCH may store.
func (h CallingHours) Validate() error {
	windows := 0
	for day, list := range h.Windows {
		if !slices.Contains(weekdayNames[:], day) {
			return fmt.Errorf("%q is not a day of the week: use monday to sunday in lowercase", day)
		}
		spans, err := parseWindows(day, list)
		if err != nil {
			return err
		}
		windows += len(spans)
	}
	if windows == 0 {
		return errors.New("calling hours need at least one window in the week")
	}
	if err := h.TimeZone.validate(); err != nil {
		return err
	}
	for _, d := range h.ExcludedDates {
		if t, err := time.Parse(excludedDateLayout, d); err != nil || t.Format(excludedDateLayout) != d {
			return fmt.Errorf("the excluded date %q must be a real date written YYYY-MM-DD", d)
		}
	}
	return nil
}

// Zone picks the time zone a contact is called in.
func (h CallingHours) Zone(contactZone, number string) *time.Location {
	if h.TimeZone.Mode == ZoneFixed {
		return loadZone(h.TimeZone.Fixed)
	}
	if loc, err := time.LoadLocation(contactZone); err == nil && contactZone != "" {
		return loc
	}
	if num, err := phonenumbers.Parse(number, ""); err == nil && !isNotWhereThePersonIs(num) {
		names, _ := phonenumbers.GetTimezonesForNumber(num)
		for _, name := range names {
			if name == unknownZone {
				continue
			}
			if loc, err := time.LoadLocation(name); err == nil {
				return loc
			}
		}
	}
	return loadZone(h.TimeZone.Fallback)
}

// Open reports whether a call may start at the instant: the contact's local date is not
// excluded and the local time lies in a window, except its last minute.
func (h CallingHours) Open(at time.Time, zone *time.Location) bool {
	local := at.In(zone)
	if h.excluded(local.Format(excludedDateLayout)) {
		return false
	}
	sinceMidnight := time.Duration(local.Hour())*time.Hour + time.Duration(local.Minute())*time.Minute +
		time.Duration(local.Second())*time.Second + time.Duration(local.Nanosecond())
	spans, err := parseWindows("", h.Windows[weekdayNames[local.Weekday()]])
	if err != nil {
		return false
	}
	for _, w := range spans {
		start := time.Duration(w.start) * time.Minute
		end := time.Duration(w.end-1) * time.Minute
		if sinceMidnight >= start && sinceMidnight < end {
			return true
		}
	}
	return false
}

// Next is the earliest instant at or after the given one where Open is true. It looks 400 days
// ahead and reports false when it finds nothing.
func (h CallingHours) Next(after time.Time, zone *time.Location) (time.Time, bool) {
	if h.Open(after, zone) {
		return after, true
	}
	local := after.In(zone)
	for i := 0; i <= lookAheadDays; i++ {
		day := time.Date(local.Year(), local.Month(), local.Day()+i, 0, 0, 0, 0, zone)
		spans, err := parseWindows("", h.Windows[weekdayNames[day.Weekday()]])
		if err != nil {
			return time.Time{}, false
		}
		for _, w := range spans {
			start := time.Date(day.Year(), day.Month(), day.Day(), w.start/60, w.start%60, 0, 0, zone)
			if !start.Before(after) && h.Open(start, zone) {
				return start, true
			}
		}
	}
	return time.Time{}, false
}

func (h CallingHours) excluded(date string) bool {
	return slices.Contains(h.ExcludedDates, date)
}

func (tz TimeZone) validate() error {
	switch tz.Mode {
	case ZoneContact:
		return checkZone("fallback", tz.Fallback)
	case ZoneFixed:
		return checkZone("fixed", tz.Fixed)
	}
	return fmt.Errorf("the time zone mode must be %q or %q", ZoneContact, ZoneFixed)
}

func checkZone(field, name string) error {
	if name == "" {
		return fmt.Errorf("the %s time zone is required", field)
	}
	if _, err := time.LoadLocation(name); err != nil {
		return fmt.Errorf("the %s time zone %q is not an IANA time zone", field, name)
	}
	return nil
}

// isNotWhereThePersonIs is true for numbers the library lists every zone of the country for,
// so the first zone would be a guess that can ring someone in the middle of the night.
func isNotWhereThePersonIs(num *phonenumbers.PhoneNumber) bool {
	switch phonenumbers.GetNumberType(num) {
	case phonenumbers.TOLL_FREE, phonenumbers.PREMIUM_RATE, phonenumbers.SHARED_COST:
		return true
	}
	return false
}

func loadZone(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	if err != nil || name == "" {
		return time.UTC
	}
	return loc
}

type span struct{ start, end int } // minutes since midnight

// parseWindows reads and sorts one day's windows. day only names the day in errors.
func parseWindows(day string, list []Window) ([]span, error) {
	spans := make([]span, 0, len(list))
	for _, w := range list {
		start, err := parseClock(w.Start)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", day, err)
		}
		end, err := parseClock(w.End)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", day, err)
		}
		if start >= end {
			return nil, fmt.Errorf("%s: a window must start before it ends and cannot cross midnight (%s to %s)", day, w.Start, w.End)
		}
		spans = append(spans, span{start, end})
	}
	slices.SortFunc(spans, func(a, b span) int { return a.start - b.start })
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return nil, fmt.Errorf("%s: windows overlap", day)
		}
	}
	return spans, nil
}

func parseClock(s string) (int, error) {
	m := clockTime.FindStringSubmatch(s)
	if m == nil {
		return 0, fmt.Errorf("the time %q must be written HH:MM", s)
	}
	h, _ := strconv.Atoi(m[1])
	min, _ := strconv.Atoi(m[2])
	if min%windowStepMinutes != 0 || min >= 60 {
		return 0, fmt.Errorf("the time %q must be on a step of %d minutes", s, windowStepMinutes)
	}
	total := h*60 + min
	if total > dayMinutes {
		return 0, fmt.Errorf("the time %q is past 24:00", s)
	}
	return total, nil
}
