package usage

import (
	"strconv"
	"strings"
	"time"
)

// Interval is how long a window is. Windows are UTC: a day, a week from Monday, 14 days from
// BiweeklyEpoch, a calendar month.
type Interval string

const (
	Daily    Interval = "daily"
	Weekly   Interval = "weekly"
	Biweekly Interval = "biweekly"
	Monthly  Interval = "monthly"
)

// Intervals is every interval a subject's spend is kept in.
var Intervals = []Interval{Daily, Weekly, Biweekly, Monthly}

// BiweeklyEpoch is the Monday the first bi-weekly window started on.
var BiweeklyEpoch = time.Date(2024, time.January, 1, 0, 0, 0, 0, time.UTC)

// Valid reports whether the interval is one spend is kept in.
func (i Interval) Valid() bool {
	switch i {
	case Daily, Weekly, Biweekly, Monthly:
		return true
	}
	return false
}

// Window is the window of interval holding t, as [start, end). An interval that is not
// Valid is taken as Daily.
func Window(interval Interval, t time.Time) (start, end time.Time) {
	t = t.UTC()
	day := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
	switch interval {
	case Weekly:
		start = day.AddDate(0, 0, -((int(day.Weekday()) + 6) % 7))
		return start, start.AddDate(0, 0, 7)
	case Biweekly:
		days := int(day.Sub(BiweeklyEpoch) / (24 * time.Hour))
		// Floored, so a day before the epoch falls in the window that ends on it.
		offset := ((days % 14) + 14) % 14
		start = day.AddDate(0, 0, -offset)
		return start, start.AddDate(0, 0, 14)
	case Monthly:
		start = time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, time.UTC)
		return start, start.AddDate(0, 1, 0)
	default:
		return day, day.AddDate(0, 0, 1)
	}
}

// Field is the name of one window's spend in a subject's Redis hash:
// "<interval>:<window start as YYYYMMDD>".
func Field(interval Interval, start time.Time) string {
	return string(interval) + ":" + start.UTC().Format("20060102")
}

// Subject is what a counter, a limit, a reset or a hold is about.
type Subject string

const (
	SubjectOrganization Subject = "organization"
	SubjectApp          Subject = "app"
	SubjectAPIKey       Subject = "api_key"
	SubjectUser         Subject = "user"         // one end user of one app
	SubjectCategory     Subject = "category"     // one usage category across the organization
	SubjectAppCategory  Subject = "app_category" // one usage category within one app
)

// SubjectRef names one subject. Ids the subject does not have are empty.
type SubjectRef struct {
	OrganizationID string  `json:"organization_id"`
	Subject        Subject `json:"subject"`
	AppID          string  `json:"app_id,omitempty"`
	APIKey         string  `json:"api_key,omitempty"`
	UserID         string  `json:"user_id,omitempty"`
	Category       string  `json:"category,omitempty"`
}

// separator joins a subject's ids in its Redis key, and the braces around the organization
// pick its cluster slot, so no id may contain any of them: "o}:app:x" would be another
// organization's app.
const separator = "|"

// keyCharacter is the first character of id that delimits a Redis key, if any.
func keyCharacter(id string) (string, bool) {
	if i := strings.IndexAny(id, separator+"{}"); i >= 0 {
		return id[i : i+1], true
	}
	return "", false
}

// Subjects is every subject an event counts for: its organization, app and key, its user
// when it has one, and each category both in the organization and in the app. A category
// named twice counts once.
func Subjects(e Event) []SubjectRef {
	categories := normalCategories(e.UsageCategories)
	refs := make([]SubjectRef, 0, 4+2*len(categories))
	org := e.OrganizationID
	refs = append(refs,
		SubjectRef{OrganizationID: org, Subject: SubjectOrganization},
		SubjectRef{OrganizationID: org, Subject: SubjectApp, AppID: e.AppID},
		SubjectRef{OrganizationID: org, Subject: SubjectAPIKey, AppID: e.AppID, APIKey: e.APIKey},
	)
	if e.UserID != "" {
		refs = append(refs, SubjectRef{OrganizationID: org, Subject: SubjectUser, AppID: e.AppID, UserID: e.UserID})
	}
	for _, category := range categories {
		refs = append(refs,
			SubjectRef{OrganizationID: org, Subject: SubjectCategory, Category: category},
			SubjectRef{OrganizationID: org, Subject: SubjectAppCategory, AppID: e.AppID, Category: category},
		)
	}
	return refs
}

// ids are the subject's own ids, in the order its key joins them.
func (r SubjectRef) ids() []string {
	switch r.Subject {
	case SubjectApp:
		return []string{r.AppID}
	case SubjectAPIKey:
		return []string{r.AppID, r.APIKey}
	case SubjectUser:
		return []string{r.AppID, r.UserID}
	case SubjectCategory:
		return []string{r.Category}
	case SubjectAppCategory:
		return []string{r.AppID, r.Category}
	}
	return nil
}

// CounterPrefix starts the key of a subject's spend counters, which the usage service writes
// and admission reads.
const CounterPrefix = "usage"

// Key is the subject's Redis key after prefix (usage or hold):
// "<prefix>:{<org>}:<subject>:<ids joined by |>", and "<prefix>:{<org>}:organization" for
// the organization, which has no other id. The braces put an organization's keys in one
// cluster slot.
func (r SubjectRef) Key(prefix string) string {
	key := prefix + ":{" + r.OrganizationID + "}:" + string(r.Subject)
	if ids := r.ids(); len(ids) > 0 {
		key += ":" + strings.Join(ids, separator)
	}
	return key
}

// Problem says what is wrong with the ref, or nothing: a known subject, an organization, and
// exactly the ids the subject has, none too long or holding a character of the key.
func (r SubjectRef) Problem() string {
	if r.OrganizationID == "" {
		return "organization_id is required"
	}
	want := map[string]bool{}
	switch r.Subject {
	case SubjectOrganization:
	case SubjectApp:
		want["app_id"] = true
	case SubjectAPIKey:
		want["app_id"], want["api_key"] = true, true
	case SubjectUser:
		want["app_id"], want["user_id"] = true, true
	case SubjectCategory:
		want["category"] = true
	case SubjectAppCategory:
		want["app_id"], want["category"] = true, true
	default:
		return "subject " + string(r.Subject) + " is not one of organization, app, api_key, user, category, app_category"
	}
	for name, value := range map[string]string{
		"organization_id": r.OrganizationID, "app_id": r.AppID, "api_key": r.APIKey, "user_id": r.UserID, "category": r.Category,
	} {
		if name != "organization_id" && want[name] != (value != "") {
			if want[name] {
				return name + " is required for subject " + string(r.Subject)
			}
			return name + " is not an id of subject " + string(r.Subject)
		}
		if len(value) > MaxIDLength {
			return name + " is longer than " + strconv.Itoa(MaxIDLength) + " bytes"
		}
		if found, ok := keyCharacter(value); ok {
			return name + " contains " + found
		}
	}
	return ""
}
