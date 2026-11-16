package policy

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// ErrUserLimit is work past a limit on one end user, which stops that user and no one else.
var ErrUserLimit = errors.New("policy: the user's spend limit is reached")

// ResetsAt is when the window a refusal of a spend limit was made in ends, after which the
// subject may spend again; false for any other error.
func ResetsAt(err error) (time.Time, bool) {
	var refusal *refused
	if errors.As(err, &refusal) && !refusal.resets.IsZero() {
		return refusal.resets, true
	}
	return time.Time{}, false
}

// teamPrefix starts the category a team is kept as.
const teamPrefix = "team:"

// subjectsOf is every subject one owner's work counts for, as usage.Subjects names them for
// the event it will become, in the order admission checks them: the organization, the app,
// each category in the organization and then in the app, the key, and the user when there is
// one. organization is whose limits they are: the app's own id for an app seen in no
// organization. A key that is an origin (hook:chat) is no key a limit names, and a user the
// usage records would leave off is no user.
func subjectsOf(owner routing.Owner, organization string) []usage.SubjectRef {
	app := owner.CustomerID
	categories := owner.UsageCategories
	refs := make([]usage.SubjectRef, 0, 4+2*len(categories))
	refs = append(refs,
		usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectOrganization},
		usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectApp, AppID: app},
	)
	// The categories were checked where the request came in (usage.CheckCategories).
	for _, category := range categories {
		refs = append(refs, usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectCategory, Category: category})
	}
	for _, category := range categories {
		refs = append(refs, usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectAppCategory, AppID: app, Category: category})
	}
	if key := owner.APIKey; key != "" && !strings.Contains(key, ":") && usage.IDProblem(key) == "" {
		refs = append(refs, usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectAPIKey, AppID: app, APIKey: key})
	}
	if user := routing.RecordedUser(owner.UserID); user != "" {
		refs = append(refs, usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectUser, AppID: app, UserID: user})
	}
	return refs
}

// limitKey is one row's place in limits_one_per_level.
type limitKey struct {
	subject  usage.Subject
	level    store.LimitLevel
	interval usage.Interval
	app      string
	key      string
	user     string
	category string
}

func keyOfLimit(l store.Limit) limitKey {
	return limitKey{subject: l.Subject, level: l.Level, interval: l.Interval, app: l.AppID, key: l.APIKey, user: l.UserID, category: l.Category}
}

// limitIndex is an organization's limits by their place, so a subject's is found by lookup.
type limitIndex map[limitKey]store.Limit

func indexLimits(rows []store.Limit) limitIndex {
	index := make(limitIndex, len(rows))
	for _, row := range rows {
		index[keyOfLimit(row)] = row
	}
	return index
}

// at is the key of the row that would limit ref at level over interval.
func at(ref usage.SubjectRef, level store.LimitLevel, interval usage.Interval) limitKey {
	key := limitKey{subject: ref.Subject, level: level, interval: interval, category: ref.Category}
	switch level {
	case store.LevelApp:
		key.app = ref.AppID
	case store.LevelAPIKey:
		key.app, key.key = ref.AppID, ref.APIKey
	case store.LevelUser:
		key.app, key.user = ref.AppID, ref.UserID
	}
	return key
}

// resolve is the limit one subject has for one interval: the row of the most specific level
// that covers it (api_key and user, then app, then organization), or false when none does.
func resolve(index limitIndex, ref usage.SubjectRef, interval usage.Interval) (store.Limit, bool) {
	for _, level := range store.LimitLevels(ref.Subject) {
		if row, ok := index[at(ref, level, interval)]; ok {
			return row, true
		}
	}
	return store.Limit{}, false
}

// limited reports whether the subject has a limit in any interval.
func limited(index limitIndex, ref usage.SubjectRef) bool {
	for _, interval := range usage.Intervals {
		if _, ok := resolve(index, ref, interval); ok {
			return true
		}
	}
	return false
}

// room is what the subject may still spend under one limit in the current window: the limit,
// less what it spent since its newest reset there, less what is held against it.
func room(limit store.Limit, baseline, spent, holds int64) int64 {
	return limit.LimitMicros - max(spent-baseline, 0) - holds
}

// baselineOf is what the subject had spent in the window that starts at start when it was
// last reset there, or 0 when it was not.
func baselineOf(resets []store.UsageReset, ref usage.SubjectRef, interval usage.Interval, start time.Time) int64 {
	var newest *store.UsageReset
	for i := range resets {
		reset := &resets[i]
		if reset.Interval == interval && reset.WindowStart.Equal(start) && reset.Ref() == ref && (newest == nil || reset.ID > newest.ID) {
			newest = reset
		}
	}
	if newest == nil {
		return 0
	}
	return newest.BaselineMicros
}

// limitRefusal is the refusal of work past a spent limit: ErrUserLimit for an end user, which
// stops that user alone, and ErrBudgetSpent for every other subject. It names the limit and,
// for a default, the subject it stopped.
func limitRefusal(limit store.Limit, ref usage.SubjectRef, resets time.Time) error {
	kind := ErrBudgetSpent
	if ref.Subject == usage.SubjectUser {
		kind = ErrUserLimit
	}
	whose := ""
	if limit.Level != store.LevelUser && limit.Level != store.LevelAPIKey {
		switch ref.Subject {
		case usage.SubjectAPIKey:
			whose = " for key " + ref.APIKey
		case usage.SubjectUser:
			whose = " for user " + ref.UserID
		}
	}
	return &refused{kind: kind, resets: resets, message: fmt.Sprintf("the %s limit of %s on %s is spent%s; it resets at %s",
		limit.Interval, dollars(limit.LimitMicros), whatLimits(limit), whose, resets.UTC().Format(time.RFC3339))}
}

// whatLimits says what a row limits, as a refusal names it.
func whatLimits(l store.Limit) string {
	team, isTeam := strings.CutPrefix(l.Category, teamPrefix)
	switch {
	case l.Subject == usage.SubjectOrganization:
		return "the organization"
	case l.Subject == usage.SubjectApp:
		return "app " + l.AppID
	case l.Subject == usage.SubjectCategory && isTeam:
		return "team " + team
	case l.Subject == usage.SubjectCategory:
		return "category " + l.Category
	case l.Subject == usage.SubjectAppCategory && isTeam:
		return "team " + team + " in app " + l.AppID
	case l.Subject == usage.SubjectAppCategory:
		return "category " + l.Category + " in app " + l.AppID
	}
	noun := "key"
	if l.Subject == usage.SubjectUser {
		noun = "user"
	}
	switch l.Level {
	case store.LevelOrganization:
		return "every " + noun + " of the organization"
	case store.LevelApp:
		return "every " + noun + " of app " + l.AppID
	case store.LevelAPIKey:
		return "key " + l.APIKey + " of app " + l.AppID
	}
	return "user " + l.UserID + " of app " + l.AppID
}
