package limitnotice

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// digestSettle is how long past midnight UTC the day before's digest waits, so what every
	// router counted in that day's last minute is written first: a Recorder writes every
	// activity.DefaultInterval and keeps what a failed write held for its next, so five
	// intervals let a router miss four writes in a row.
	digestSettle = 5 * activity.DefaultInterval
	// digestListMax is how many organizations one list of the digest names; the rest are
	// counted. Slack truncates a message's text past 40,000 characters
	// (https://docs.slack.dev/reference/methods/chat.postMessage), and a line is at most about
	// 1,000: an id of up to 255, every area of the 14 with its clients, and a Staff link. Two
	// lists of 15 stay inside it.
	digestListMax = 15
)

// DayDigest is what organizations did with Agents on one UTC day.
type DayDigest struct {
	Day time.Time
	// Organizations are those whose apps made a request on Day, by id.
	Organizations []OrganizationDay
	// ActiveBefore is how many organizations' apps made a request the day before Day.
	ActiveBefore int
}

// OrganizationDay is what one organization's apps did on the digest's day.
type OrganizationDay struct {
	ID string
	// New says the organization had no activity before the day.
	New   bool
	Areas []store.ActivityArea
}

// organizationDays groups areas, which come ordered by organization, by organization.
func organizationDays(areas []store.OrganizationDayArea) []OrganizationDay {
	var days []OrganizationDay
	for _, area := range areas {
		if last := len(days) - 1; last >= 0 && days[last].ID == area.OrganizationID {
			days[last].Areas = append(days[last].Areas, area.ActivityArea)
			continue
		}
		days = append(days, OrganizationDay{
			ID: area.OrganizationID, New: area.FirstDay, Areas: []store.ActivityArea{area.ActivityArea},
		})
	}
	return days
}

// counts is how many requests the organization's apps made and how many of them were refused.
func (o OrganizationDay) counts() (requests, refused int64) {
	for _, area := range o.Areas {
		requests += area.Requests
		refused += area.Refused
	}
	return requests, refused
}

// allRefused says every request the organization's apps made was refused.
func (o OrganizationDay) allRefused() bool {
	requests, refused := o.counts()
	return refused == requests
}

// refusalCode is the code that ended the most of the organization's refused requests, as far
// as the rows say: each keeps only the code of its latest refusal, which stands for all of its
// refusals here. A tie goes to the code first by name.
func (o OrganizationDay) refusalCode() string {
	refusals := map[string]int64{}
	for _, area := range o.Areas {
		refusals[area.LastRefusalCode] += area.Refused
	}
	var code string
	var most int64
	for named, refused := range refusals {
		if refused > most || refused == most && named < code {
			code, most = named, refused
		}
	}
	return code
}

// path is the areas the organization's apps had a request answered in, each with its clients:
// the steps (activity.Steps) in their order, then the other areas by name.
func (o OrganizationDay) path() string {
	clients := map[string][]string{}
	for _, area := range o.Areas {
		if area.Requests > area.Refused {
			clients[area.Area] = append(clients[area.Area], area.Client)
		}
	}
	areas := slices.Collect(maps.Keys(clients))
	slices.SortFunc(areas, func(a, b string) int {
		return cmp.Or(cmp.Compare(stepRank(a), stepRank(b)), cmp.Compare(a, b))
	})
	var steps, others []string
	for _, area := range areas {
		named := area + " (" + strings.Join(clients[area], ", ") + ")"
		if stepRank(area) < len(activity.Steps) {
			steps = append(steps, named)
		} else {
			others = append(others, named)
		}
	}
	path := strings.Join(steps, " → ")
	if len(others) > 0 && len(steps) > 0 {
		path += " + "
	}
	return path + strings.Join(others, ", ")
}

// stepRank is where area is in activity.Steps, and past them for an area that is no step.
func stepRank(area string) int {
	if rank := slices.Index(activity.Steps, area); rank >= 0 {
		return rank
	}
	return len(activity.Steps)
}

// digestMessage is the digest as it is posted, in Slack's mrkdwn, tagged with its environment
// as Message is.
func digestMessage(environment string, links Links, digest DayDigest) string {
	var b strings.Builder
	b.WriteString(EnvironmentTag(environment))
	name := strings.ToLower(strings.TrimSpace(environment))
	if name == "" {
		name = "development"
	}
	fmt.Fprintf(&b, "Agents yesterday (%s, %s)", digest.Day.UTC().Format("2 Jan"), name)

	var fresh, refused []OrganizationDay
	for _, organization := range digest.Organizations {
		if organization.New {
			fresh = append(fresh, organization)
		}
		if organization.allRefused() {
			refused = append(refused, organization)
		}
	}
	fmt.Fprintf(&b, "\nNew orgs (first request yesterday): %d", len(fresh))
	writeList(&b, fresh, func(organization OrganizationDay) string {
		reached := activity.FurthestStep(organization.Areas)
		if reached == "" {
			reached = "—"
		}
		detail := organization.path()
		if organization.allRefused() {
			requests, _ := organization.counts()
			detail = fmt.Sprintf("all %s refused%s", countOf(requests), inParentheses(organization.refusalCode()))
		}
		return fmt.Sprintf("%s   reached: %s   %s", organizationName(links, organization.ID), reached, detail)
	})
	fmt.Fprintf(&b, "\nAll requests refused: %d", len(refused))
	writeList(&b, refused, func(organization OrganizationDay) string {
		requests, _ := organization.counts()
		return fmt.Sprintf("%s   %s%s", organizationName(links, organization.ID), countOf(requests),
			inParentheses(organization.refusalCode()))
	})
	fmt.Fprintf(&b, "\nActive orgs: %d (day before: %d)", len(digest.Organizations), digest.ActiveBefore)
	return b.String()
}

// writeList writes a line for each of the first digestListMax organizations, and how many more
// there are.
func writeList(b *strings.Builder, organizations []OrganizationDay, line func(OrganizationDay) string) {
	for _, organization := range organizations[:min(len(organizations), digestListMax)] {
		b.WriteString("\n  • " + line(organization))
	}
	if more := len(organizations) - digestListMax; more > 0 {
		fmt.Fprintf(b, "\n  … and %d more", more)
	}
}

// organizationName is the organization's id, linked to its Staff page when there is one.
func organizationName(links Links, organization string) string {
	if staff := link(links.Staff, store.LimitNotice{OrganizationID: organization}); staff != "" {
		return "<" + staff + "|" + organization + ">"
	}
	return "`" + organization + "`"
}

// countOf is how many requests there were, in words.
func countOf(requests int64) string {
	if requests == 1 {
		return "1 request"
	}
	return fmt.Sprintf("%d requests", requests)
}

// inParentheses is " (text)", or nothing for no text.
func inParentheses(text string) string {
	if text == "" {
		return ""
	}
	return " (" + text + ")"
}

// postDigest posts the activity digest of the UTC day before now's, once per day whatever the
// routers and restarts: the first check digestSettle past midnight to claim the day posts it,
// and a claim whose post failed is taken again once stale.
func (a *Anomalies) postDigest(ctx context.Context, now time.Time) {
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	if now.Before(today.Add(digestSettle)) {
		return
	}
	day := today.AddDate(0, 0, -1)
	claimed, err := a.store.ClaimActivityDigest(ctx, day, stale)
	if err != nil {
		if ctx.Err() == nil {
			a.logger.Error("could not claim the activity digest, trying again", "in", a.every, "error", err)
		}
		return
	}
	if !claimed {
		return
	}
	digest, err := a.dayDigest(ctx, day)
	if err != nil {
		if ctx.Err() == nil {
			a.logger.Error("could not read the activity digest; it is retried", "day", day.Format(time.DateOnly), "error", err)
		}
		return
	}
	if err := a.poster.Post(ctx, digestMessage(a.environment, a.links, digest)); err != nil {
		a.logger.Error("could not post the activity digest; it is retried", "day", day.Format(time.DateOnly), "error", err)
		return
	}
	if err := a.store.MarkActivityDigestPosted(ctx, day); err != nil {
		a.logger.Error("posted the activity digest, and could not record it; it may be posted again",
			"day", day.Format(time.DateOnly), "error", err)
		return
	}
	a.logger.Info("posted the activity digest", "day", day.Format(time.DateOnly))
}

// dayDigest reads what organizations did on day, a UTC day.
func (a *Anomalies) dayDigest(ctx context.Context, day time.Time) (DayDigest, error) {
	areas, err := a.store.DayActivity(ctx, day)
	if err != nil {
		return DayDigest{}, err
	}
	before, err := a.store.ActiveOrganizations(ctx, day.AddDate(0, 0, -1))
	if err != nil {
		return DayDigest{}, err
	}
	return DayDigest{Day: day, Organizations: organizationDays(areas), ActiveBefore: before}, nil
}
