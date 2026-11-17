package client

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// What a limit, a grant, a reset or a usage read is about.
const (
	Organization = acceleration.LimitSubjectOrganization
	App          = acceleration.LimitSubjectApp
	APIKey       = acceleration.LimitSubjectApiKey
	User         = acceleration.LimitSubjectUser
	// Team is a team across the organization, or within one app when AppID is set.
	Team = acceleration.LimitSubjectTeam
	// AppTeam is a team within one app. Team with an AppID is the same subject.
	AppTeam = acceleration.LimitSubjectAppTeam
	// Category is a usage category across the organization, or within one app when AppID is
	// set. A team is not a category: use Team.
	Category = acceleration.LimitSubjectCategory
	// AppCategory is a usage category within one app. Category with an AppID is the same.
	AppCategory = acceleration.LimitSubjectAppCategory
)

// The windows a limit counts in.
const (
	Daily    = acceleration.LimitIntervalDaily
	Weekly   = acceleration.LimitIntervalWeekly
	Biweekly = acceleration.LimitIntervalBiweekly
	Monthly  = acceleration.LimitIntervalMonthly
)

const (
	// maxUsageSubjects is the most subjects one usage read names, as the router counts them.
	maxUsageSubjects = 100
	// limitsPage is the page size lists are read in, the most the router answers.
	limitsPage = 200
	teamPrefix = "team:"
)

var (
	teamName     = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,59}$`)
	categoryName = regexp.MustCompile(`^[A-Za-z0-9_.:-]{1,64}$`)
)

// SpendLimits is the customer's spend limits: on the organization, its apps, keys, end
// users, teams and usage categories. Server side only: the router refuses a user's token,
// 403 server_side_only.
//
// Every limit here is the customer's own. Stream's spend limit for the organization is never
// among them, and nothing here lifts it.
//
// Work a limit refuses fails with a *stream.RouterError of Status 402 and Code
// spend_limit_reached, or 429 user_limit_reached for one end user's limit. Stream's own limit
// refuses with 402 system_limit_reached.
type SpendLimits struct {
	client *Client
}

// SpendLimits is the customer's spend limits.
func (c *Client) SpendLimits() *SpendLimits { return &SpendLimits{client: c} }

// Limit is a limit to create: what one subject may spend in each window of Interval.
//
// The level is the most specific one the ids name, which the router works out: User with a
// UserID is that user's own limit, User with only an AppID the default for every user of the
// app, and User alone the default for every user of the organization. Keys go the same way.
// Level overrides it.
type Limit struct {
	Subject  acceleration.LimitSubject
	Interval acceleration.LimitInterval
	Amount   USD

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string

	Level acceleration.LimitLevel
	Note  string
}

// LimitFilter is which limits List reads. The zero value is every limit.
type LimitFilter struct {
	Subject  acceleration.LimitSubject
	Level    acceleration.LimitLevel
	Interval acceleration.LimitInterval

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string
}

// Target is one subject, for what one subject spent.
type Target struct {
	Subject acceleration.LimitSubject

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string
}

// UsageIDs names several subjects of one kind. Each id is one value for every subject, or one
// per subject: at most 100, and every list of more than one the same length.
type UsageIDs struct {
	AppID    []string
	APIKey   []string
	UserID   []string
	Team     []string
	Category []string
}

// Grant is a one-time amount one subject may spend past its own spent limits.
//
// It is drawn while any of the subject's limits is spent, across windows, until it is used
// up, ExpiresAt passes or it is revoked. It never lifts a parent's limit or Stream's. "$10
// more today" is a grant that expires at the next midnight UTC. A zero ExpiresAt is the
// router's default.
type Grant struct {
	Subject   acceleration.LimitSubject
	Amount    USD
	Reason    string
	ExpiresAt time.Time

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string
}

// GrantFilter is which grants Grants reads. The zero value is every grant.
type GrantFilter struct {
	Subject acceleration.LimitSubject
	State   acceleration.LimitGrantState

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string
}

// Reset counts a subject's limits from now until each window ends: every interval's, or only
// those in Intervals. Nothing is refunded: what it spent stays spent.
type Reset struct {
	Subject   acceleration.LimitSubject
	Reason    string
	Intervals []acceleration.LimitInterval

	AppID    string
	APIKey   string
	UserID   string
	Team     string
	Category string
}

// List is every limit the filter matches, read page by page.
func (s *SpendLimits) List(ctx context.Context, filter LimitFilter) ([]acceleration.Limit, error) {
	ids := subjectIDs{filter.AppID, filter.APIKey, filter.UserID, filter.Team, filter.Category}
	params := acceleration.ListLimitsParams{
		AppId: pointer(ids.appID), ApiKeyId: pointer(ids.apiKey), UserId: pointer(ids.userID),
		Team: pointer(ids.team), Category: pointer(ids.category),
		Level: pointer(filter.Level), Interval: pointer(filter.Interval), Limit: pointer(int64(limitsPage)),
	}
	if err := ids.filtering(filter.Subject, &params.Subject); err != nil {
		return nil, err
	}
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}
	var limits []acceleration.Limit
	for {
		page, err := api.ListLimitsWithResponse(ctx, &params)
		if err != nil {
			return nil, fmt.Errorf("client: listing spend limits: %w", err)
		}
		if page.JSON200 == nil {
			return nil, failure("listing spend limits", page.HTTPResponse, page.Body)
		}
		if page.JSON200.Items != nil {
			limits = append(limits, *page.JSON200.Items...)
		}
		if !page.JSON200.HasMore || page.JSON200.NextCursor == nil || *page.JSON200.NextCursor == "" {
			return limits, nil
		}
		params.Cursor = page.JSON200.NextCursor
	}
}

// Get is one limit by its id.
func (s *SpendLimits) Get(ctx context.Context, limitID int64) (acceleration.Limit, error) {
	api, err := s.client.api()
	if err != nil {
		return acceleration.Limit{}, err
	}
	what := fmt.Sprintf("reading the spend limit %d", limitID)
	got, err := api.GetLimitWithResponse(ctx, limitID)
	if err != nil {
		return acceleration.Limit{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if got.JSON200 == nil {
		return acceleration.Limit{}, failure(what, got.HTTPResponse, got.Body)
	}
	return *got.JSON200, nil
}

// Create creates a limit.
//
// The router refuses with Code limit_above_parent (the limit is above one of a subject that
// contains it), limit_exists (the subject has one at that level and interval: change it with
// SetAmount) or use_team.
func (s *SpendLimits) Create(ctx context.Context, limit Limit) (acceleration.Limit, error) {
	ids := subjectIDs{limit.AppID, limit.APIKey, limit.UserID, limit.Team, limit.Category}
	kind, err := ids.target(limit.Subject)
	if err != nil {
		return acceleration.Limit{}, err
	}
	if limit.Interval == "" {
		return acceleration.Limit{}, errors.New("client: a spend limit needs an interval")
	}
	if limit.Amount < 0 {
		return acceleration.Limit{}, fmt.Errorf("client: a spend limit of %s is below zero", limit.Amount)
	}
	api, err := s.client.api()
	if err != nil {
		return acceleration.Limit{}, err
	}
	what := fmt.Sprintf("creating a %s spend limit on %s", limit.Interval, ids.named(kind))
	created, err := api.CreateLimitWithResponse(ctx, acceleration.LimitInput{
		Subject: kind, Interval: limit.Interval, LimitMicros: limit.Amount.Micros(),
		AppId: pointer(ids.appID), ApiKey: pointer(ids.apiKey), UserId: pointer(ids.userID),
		Team: pointer(ids.team), Category: pointer(ids.category),
		Level: pointer(limit.Level), Note: pointer(limit.Note),
	})
	if err != nil {
		return acceleration.Limit{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if created.JSON201 == nil {
		return acceleration.Limit{}, failure(what, created.HTTPResponse, created.Body)
	}
	return *created.JSON201, nil
}

// SetAmount changes what a limit allows. What it is on never changes.
func (s *SpendLimits) SetAmount(ctx context.Context, limitID int64, amount USD) (acceleration.Limit, error) {
	if amount < 0 {
		return acceleration.Limit{}, fmt.Errorf("client: a spend limit of %s is below zero", amount)
	}
	micros := amount.Micros()
	return s.update(ctx, limitID, acceleration.LimitChange{LimitMicros: &micros})
}

// SetNote changes a limit's note. An empty one clears it.
func (s *SpendLimits) SetNote(ctx context.Context, limitID int64, note string) (acceleration.Limit, error) {
	return s.update(ctx, limitID, acceleration.LimitChange{Note: &note})
}

func (s *SpendLimits) update(ctx context.Context, limitID int64, change acceleration.LimitChange) (acceleration.Limit, error) {
	api, err := s.client.api()
	if err != nil {
		return acceleration.Limit{}, err
	}
	what := fmt.Sprintf("updating the spend limit %d", limitID)
	updated, err := api.UpdateLimitWithResponse(ctx, limitID, change)
	if err != nil {
		return acceleration.Limit{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if updated.JSON200 == nil {
		return acceleration.Limit{}, failure(what, updated.HTTPResponse, updated.Body)
	}
	return *updated.JSON200, nil
}

// Delete deletes a limit. Its subject's spend from then on is held only to its parents'.
func (s *SpendLimits) Delete(ctx context.Context, limitID int64) error {
	api, err := s.client.api()
	if err != nil {
		return err
	}
	what := fmt.Sprintf("deleting the spend limit %d", limitID)
	deleted, err := api.DeleteLimitWithResponse(ctx, limitID)
	if err != nil {
		return fmt.Errorf("client: %s: %w", what, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure(what, deleted.HTTPResponse, deleted.Body)
	}
	return nil
}

// Grant gives a subject a one-time amount past its spent limits.
//
// The router refuses with Code no_limit (404: the subject has no limit of the customer's to
// grant past), invalid_expiry or too_many_grants.
func (s *SpendLimits) Grant(ctx context.Context, grant Grant) (acceleration.LimitGrant, error) {
	if grant.Reason == "" {
		return acceleration.LimitGrant{}, errors.New("client: a grant needs a reason")
	}
	if grant.Amount < 0 {
		return acceleration.LimitGrant{}, fmt.Errorf("client: a grant of %s is below zero", grant.Amount)
	}
	ids := subjectIDs{grant.AppID, grant.APIKey, grant.UserID, grant.Team, grant.Category}
	kind, err := ids.target(grant.Subject)
	if err != nil {
		return acceleration.LimitGrant{}, err
	}
	body := acceleration.LimitGrantInput{
		Subject: kind, AmountMicros: grant.Amount.Micros(), Reason: grant.Reason,
		AppId: pointer(ids.appID), ApiKey: pointer(ids.apiKey), UserId: pointer(ids.userID),
		Team: pointer(ids.team), Category: pointer(ids.category),
	}
	if !grant.ExpiresAt.IsZero() {
		expires := grant.ExpiresAt.UTC()
		body.ExpiresAt = &expires
	}
	api, err := s.client.api()
	if err != nil {
		return acceleration.LimitGrant{}, err
	}
	what := fmt.Sprintf("granting %s past its spend limits", ids.named(kind))
	created, err := api.CreateLimitGrantWithResponse(ctx, body)
	if err != nil {
		return acceleration.LimitGrant{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if created.JSON201 == nil {
		return acceleration.LimitGrant{}, failure(what, created.HTTPResponse, created.Body)
	}
	return *created.JSON201, nil
}

// Grants is every grant the filter matches, read page by page.
func (s *SpendLimits) Grants(ctx context.Context, filter GrantFilter) ([]acceleration.LimitGrant, error) {
	ids := subjectIDs{filter.AppID, filter.APIKey, filter.UserID, filter.Team, filter.Category}
	params := acceleration.ListLimitGrantsParams{
		AppId: pointer(ids.appID), ApiKeyId: pointer(ids.apiKey), UserId: pointer(ids.userID),
		Team: pointer(ids.team), Category: pointer(ids.category),
		State: pointer(filter.State), Limit: pointer(int64(limitsPage)),
	}
	if err := ids.filtering(filter.Subject, &params.Subject); err != nil {
		return nil, err
	}
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}
	var grants []acceleration.LimitGrant
	for {
		page, err := api.ListLimitGrantsWithResponse(ctx, &params)
		if err != nil {
			return nil, fmt.Errorf("client: listing spend limit grants: %w", err)
		}
		if page.JSON200 == nil {
			return nil, failure("listing spend limit grants", page.HTTPResponse, page.Body)
		}
		if page.JSON200.Items != nil {
			grants = append(grants, *page.JSON200.Items...)
		}
		if !page.JSON200.HasMore || page.JSON200.NextCursor == nil || *page.JSON200.NextCursor == "" {
			return grants, nil
		}
		params.Cursor = page.JSON200.NextCursor
	}
}

// RevokeGrant ends a grant before it is used up or expires.
func (s *SpendLimits) RevokeGrant(ctx context.Context, grantID int64, reason string) (acceleration.LimitGrant, error) {
	if reason == "" {
		return acceleration.LimitGrant{}, errors.New("client: revoking a grant needs a reason")
	}
	api, err := s.client.api()
	if err != nil {
		return acceleration.LimitGrant{}, err
	}
	what := fmt.Sprintf("revoking the grant %d", grantID)
	revoked, err := api.RevokeLimitGrantWithResponse(ctx, grantID, acceleration.LimitGrantRevocation{Reason: reason})
	if err != nil {
		return acceleration.LimitGrant{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if revoked.JSON200 == nil {
		return acceleration.LimitGrant{}, failure(what, revoked.HTTPResponse, revoked.Body)
	}
	return *revoked.JSON200, nil
}

// UnrevokeGrant gives back a revoked grant what it had left.
func (s *SpendLimits) UnrevokeGrant(ctx context.Context, grantID int64, reason string) (acceleration.LimitGrant, error) {
	if reason == "" {
		return acceleration.LimitGrant{}, errors.New("client: unrevoking a grant needs a reason")
	}
	api, err := s.client.api()
	if err != nil {
		return acceleration.LimitGrant{}, err
	}
	what := fmt.Sprintf("unrevoking the grant %d", grantID)
	restored, err := api.UnrevokeLimitGrantWithResponse(ctx, grantID, acceleration.LimitGrantUnrevocation{Reason: reason})
	if err != nil {
		return acceleration.LimitGrant{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if restored.JSON200 == nil {
		return acceleration.LimitGrant{}, failure(what, restored.HTTPResponse, restored.Body)
	}
	return *restored.JSON200, nil
}

// Usage is what one subject spent in its current windows, against its limits and grants.
func (s *SpendLimits) Usage(ctx context.Context, target Target) (acceleration.SubjectUsage, error) {
	answered, err := s.UsageMany(ctx, target.Subject, UsageIDs{
		AppID: listed(target.AppID), APIKey: listed(target.APIKey), UserID: listed(target.UserID),
		Team: listed(target.Team), Category: listed(target.Category),
	})
	if err != nil {
		return acceleration.SubjectUsage{}, err
	}
	if len(answered) != 1 {
		return acceleration.SubjectUsage{}, fmt.Errorf("client: reading the spend of one %s: the router answered %d", target.Subject, len(answered))
	}
	return answered[0], nil
}

// UsageMany is what several subjects of one kind spent, one SubjectUsage each, in the order
// asked.
func (s *SpendLimits) UsageMany(ctx context.Context, subject acceleration.LimitSubject, ids UsageIDs) ([]acceleration.SubjectUsage, error) {
	lists := []struct {
		name   string
		values []string
	}{{"AppID", ids.AppID}, {"APIKey", ids.APIKey}, {"UserID", ids.UserID}, {"Team", ids.Team}, {"Category", ids.Category}}
	count := 1
	for _, list := range lists {
		count = max(count, len(list.values))
	}
	if count > maxUsageSubjects {
		return nil, fmt.Errorf("client: a usage read names at most %d subjects, this one %d", maxUsageSubjects, count)
	}
	for _, list := range lists {
		if len(list.values) > 1 && len(list.values) != count {
			return nil, fmt.Errorf("client: %s names %d subjects and another id %d: give one value for all of them, or one per subject", list.name, len(list.values), count)
		}
		for _, value := range list.values {
			if value == "" {
				return nil, fmt.Errorf("client: %s names an empty subject", list.name)
			}
		}
	}
	kind, err := kindOf(subject, len(ids.AppID) > 0, ids.Team, ids.Category)
	if err != nil {
		return nil, err
	}
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}
	what := fmt.Sprintf("reading the spend of %s", kind)
	read, err := api.GetLimitUsageWithResponse(ctx, &acceleration.GetLimitUsageParams{
		Subject: kind, AppId: some(ids.AppID), ApiKeyId: some(ids.APIKey), UserId: some(ids.UserID),
		Team: some(ids.Team), Category: some(ids.Category),
	})
	if err != nil {
		return nil, fmt.Errorf("client: %s: %w", what, err)
	}
	if read.JSON200 == nil {
		return nil, failure(what, read.HTTPResponse, read.Body)
	}
	return *read.JSON200, nil
}

// Reset counts a subject's limits from now. The router refuses with Code no_limit a subject
// with no limit of the customer's.
func (s *SpendLimits) Reset(ctx context.Context, reset Reset) (acceleration.SubjectUsage, error) {
	if reset.Reason == "" {
		return acceleration.SubjectUsage{}, errors.New("client: a reset needs a reason")
	}
	ids := subjectIDs{reset.AppID, reset.APIKey, reset.UserID, reset.Team, reset.Category}
	kind, err := ids.target(reset.Subject)
	if err != nil {
		return acceleration.SubjectUsage{}, err
	}
	body := acceleration.LimitResetInput{
		Subject: kind, Reason: reset.Reason,
		AppId: pointer(ids.appID), ApiKey: pointer(ids.apiKey), UserId: pointer(ids.userID),
		Team: pointer(ids.team), Category: pointer(ids.category),
	}
	if len(reset.Intervals) > 0 {
		body.Intervals = &reset.Intervals
	}
	api, err := s.client.api()
	if err != nil {
		return acceleration.SubjectUsage{}, err
	}
	what := fmt.Sprintf("resetting the spend of %s", ids.named(kind))
	done, err := api.ResetLimitUsageWithResponse(ctx, body)
	if err != nil {
		return acceleration.SubjectUsage{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if done.JSON200 == nil {
		return acceleration.SubjectUsage{}, failure(what, done.HTTPResponse, done.Body)
	}
	return *done.JSON200, nil
}

// subjectIDs are the ids that name one subject, which every limit body and filter has.
type subjectIDs struct {
	appID, apiKey, userID, team, category string
}

// target is the subject a write is about, as the router names it.
func (ids subjectIDs) target(subject acceleration.LimitSubject) (acceleration.LimitSubject, error) {
	if subject == "" {
		return "", errors.New("client: name the subject: Organization, App, APIKey, User, Team or Category")
	}
	return kindOf(subject, ids.appID != "", listed(ids.team), listed(ids.category))
}

// filtering sets a list's subject, as the router names it, and checks its names.
func (ids subjectIDs) filtering(subject acceleration.LimitSubject, set **acceleration.LimitSubject) error {
	if subject != "" {
		kind, err := ids.target(subject)
		if err != nil {
			return err
		}
		*set = &kind
		return nil
	}
	if ids.team != "" {
		if err := checkTeam(ids.team); err != nil {
			return err
		}
	}
	if ids.category != "" {
		if err := checkCategory(ids.category); err != nil {
			return err
		}
	}
	return nil
}

// named is the subject as a message names it, such as "team (team finance)".
func (ids subjectIDs) named(kind acceleration.LimitSubject) string {
	var named []string
	for _, id := range []struct{ name, value string }{
		{"app", ids.appID}, {"key", ids.apiKey}, {"user", ids.userID}, {"team", ids.team}, {"category", ids.category},
	} {
		if id.value != "" {
			named = append(named, id.name+" "+id.value)
		}
	}
	if len(named) == 0 {
		return string(kind)
	}
	return fmt.Sprintf("%s (%s)", kind, strings.Join(named, ", "))
}

// kindOf is the subject as the router names it: a team or category within an app is AppTeam
// or AppCategory. It refuses what the router would, before anything is sent.
func kindOf(subject acceleration.LimitSubject, inApp bool, teams, categories []string) (acceleration.LimitSubject, error) {
	for _, category := range categories {
		if err := checkCategory(category); err != nil {
			return "", err
		}
	}
	for _, team := range teams {
		if err := checkTeam(team); err != nil {
			return "", err
		}
	}
	switch subject {
	case Team, AppTeam:
		if len(teams) == 0 {
			return "", fmt.Errorf("client: a %s subject names its team in Team", subject)
		}
		if subject == Team && inApp {
			subject = AppTeam
		}
	case Category, AppCategory:
		if len(categories) == 0 {
			return "", fmt.Errorf("client: a %s subject names its category in Category", subject)
		}
		if subject == Category && inApp {
			subject = AppCategory
		}
	}
	if len(teams) > 0 && subject != Team && subject != AppTeam {
		return "", fmt.Errorf("client: Team is for the Team subject, not %s", subject)
	}
	if len(categories) > 0 && subject != Category && subject != AppCategory {
		return "", fmt.Errorf("client: Category is for the Category subject, not %s", subject)
	}
	if (subject == AppTeam || subject == AppCategory) && !inApp {
		return "", fmt.Errorf("client: an %s subject names its app in AppID", subject)
	}
	return subject, nil
}

// checkTeam refuses a team name the router would.
func checkTeam(team string) error {
	if !teamName.MatchString(team) {
		return fmt.Errorf("client: team %q is not 1 to 59 characters of A-Z, a-z, 0-9, _ . and -", team)
	}
	return nil
}

// checkCategory refuses a usage category the router would, and a team written as one.
func checkCategory(category string) error {
	// In any case, as the proxy reads it: Team:finance would be a second team.
	if strings.HasPrefix(strings.ToLower(category), teamPrefix) {
		return fmt.Errorf("client: category %q names a team: use the Team subject with Team", category)
	}
	if !categoryName.MatchString(category) {
		return fmt.Errorf("client: usage category %q is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -", category)
	}
	return nil
}

func listed(value string) []string {
	if value == "" {
		return nil
	}
	return []string{value}
}

func some(values []string) *[]string {
	if len(values) == 0 {
		return nil
	}
	return &values
}
