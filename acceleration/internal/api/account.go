package api

import (
	"cmp"
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// Where an account's balance warning threshold comes from.
const (
	warningFromTier               = "tier"
	warningFromAccount            = "account"
	warningFromOrganizationPolicy = "organization_policy"
)

// defaultCredits is how many credit entries a page holds when the caller names no limit.
const defaultCredits = 50

var (
	errBillingNotEnabled = APIError{Type: ErrorTypeNotFound, Code: codeBillingNotEnabled,
		Message: "billing is not enabled here: this deployment has no usage service, or the app is in no organization it bills"}
	errUsageUnavailable = unavailable("the usage service did not answer; try again shortly")
	errCreditsCursor    = invalidRequest("cursor is not one this list answered")
)

// Account is the calling organization's money, as its dashboard shows it.
type Account struct {
	OrganizationId       string               `json:"organization_id"`
	Status               string               `json:"status" enum:"active,suspended" doc:"suspended: Stream refuses every request of the organization, free work included, until it restores it."`
	Tier                 string               `json:"tier" doc:"The tier the organization is billed on."`
	BalanceMicros        int64                `json:"balance_micros" doc:"What the organization has to spend now, in millionths of a dollar: its credit less what it spent and what work in flight holds, the figure every answer's X-Stream-Balance-Micros says."`
	GrantedMicros        int64                `json:"granted_micros" doc:"Every credit entry summed: top-ups, grants, refunds and adjustments."`
	SpentMicros          int64                `json:"spent_micros" doc:"Every billable unit of work recorded, summed."`
	BalanceWarningMicros int64                `json:"balance_warning_micros" doc:"The balance at or below which the organization is warned: X-Stream-Balance-Warning says low, and its stream event destinations get account.balance_low. 0 when nothing sets one."`
	BalanceWarningSource string               `json:"balance_warning_source" enum:"tier,account,organization_policy" doc:"Whose threshold it is: the tier's, the one Stream set for this organization, or the organization policy's balance_warning_micros, which wins over both."`
	BalanceState         string               `json:"balance_state" enum:"ok,low,exhausted" doc:"Where balance_micros stands: exhausted at or below 0, low at or below the threshold."`
	Limits               usage.Limits         `json:"limits" doc:"What Stream allows the organization: its tier's settings, with those Stream set for this organization laid over them."`
	OwnLimits            []string             `json:"own_limits,omitempty" doc:"The settings in limits Stream set for this organization rather than leaving its tier's, as dotted paths such as channels.voice.concurrent."`
	SpendHardLimit       *SpendHardLimitUsage `json:"spend_hard_limit,omitempty" doc:"The most the organization may spend in the UTC day and the calendar month, whatever its credit and its own limits, with what it spent and what is left. Absent when Stream sets none."`
}

func (*Account) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The calling organization's account: its balance, its credit and spend, its " +
		"status and tier, the threshold it is warned at, and what Stream allows it. Why an " +
		"organization is suspended is Stream's, and is not here."
	return schema
}

// CreditEntry is one change to the organization's credit.
type CreditEntry struct {
	Id           int64     `json:"id"`
	Kind         string    `json:"kind" enum:"top_up,grant,refund,dispute,dispute_reversal,adjustment"`
	AmountMicros int64     `json:"amount_micros" doc:"Signed: what the entry added to the credit, in millionths of a dollar."`
	Reference    string    `json:"reference,omitempty" doc:"The payment's own reference, such as a top-up's payment id. Absent on entries Stream's staff wrote."`
	CreatedAt    time.Time `json:"created_at"`
}

// CreditPage is one page of the organization's credit entries, newest first.
type CreditPage struct {
	Items      []CreditEntry `json:"items" nullable:"false"`
	HasMore    bool          `json:"has_more"`
	NextCursor *string       `json:"next_cursor,omitempty"`
}

// WindowSpend is what a subject spent in its current window of one interval.
type WindowSpend struct {
	Interval    LimitInterval `json:"interval"`
	WindowStart time.Time     `json:"window_start"`
	WindowEnd   time.Time     `json:"window_end"`
	SpentMicros int64         `json:"spent_micros"`
}

// Team is one of the organization's teams: one a limit names, or one this month's work
// carried. It has no members: each user's token names its team.
type Team struct {
	Name   string        `json:"name"`
	Apps   []string      `json:"apps" nullable:"false" doc:"The apps whose work carried the team this calendar month (UTC), by id."`
	Spent  []WindowSpend `json:"spent" nullable:"false" doc:"What the team spent in the current daily, weekly, biweekly and monthly windows."`
	Limits []Limit       `json:"limits" nullable:"false" doc:"The team's spend limits, across the organization and within each app, by id."`
}

func (*Team) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One of the organization's teams: one a spend limit names, or one this month's " +
		"work carried. A team has no members: each user's token names its team."
	return schema
}

type accountResponse struct{ Body Account }

type listCreditsRequest struct {
	Limit  int    `query:"limit" minimum:"1" maximum:"100" doc:"Up to 100. Omitted is 50."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page."`
}

type creditPageResponse struct{ Body CreditPage }

// ListTeamsParams narrow a list of teams to one app.
type ListTeamsParams struct {
	AppId string `query:"app_id" doc:"Only what happened within this app: each team's spend and limits there, and only the teams it saw."`
}

type manageListTeamsRequest struct {
	ListTeamsParams
	OrganizationId string `query:"organization_id" required:"true"`
}

type teamsResponse struct {
	Body []Team `nullable:"false"`
}

func (s *Server) registerAccount(api huma.API) {
	serverSide := "\n\nServer-side only: a request that is not server-side answers 403 server_side_only."
	notEnabled := " With no usage service, or for an app in no organization, it answers 404 billing_not_enabled. " +
		"A suspended organization reads it all the same."
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound, http.StatusServiceUnavailable}
	huma.Register(api, huma.Operation{
		OperationID: "getAccount", Method: http.MethodGet, Path: "/v1/account",
		Summary: "Read the organization's account",
		Description: "The caller's organization's balance, credit, spend, status, tier, the threshold it is warned " +
			"at and where that comes from, what Stream allows it, and Stream's spend hard limit on it with this " +
			"window's spend. The balance is what admission counts, less the work in flight, so it agrees with " +
			"X-Stream-Balance-Micros." + notEnabled + serverSide,
		Errors: errs,
	}, s.getAccount)
	huma.Register(api, huma.Operation{
		OperationID: "listCredits", Method: http.MethodGet, Path: "/v1/account/credits",
		Summary:     "List the organization's credit entries",
		Description: "Newest first: top-ups, grants, refunds, disputes and adjustments." + notEnabled + serverSide,
		Errors:      errs,
	}, s.listCredits)
	list := huma.Operation{
		OperationID: "listTeams", Method: http.MethodGet, Path: "/v1/teams",
		Summary: "List the organization's teams",
		Description: "Every team a spend limit names or this month's work carried, by name, with the apps that " +
			"carried it this month, what it spent in each current window and its limits, each with what it can " +
			"spend once its parents are counted. A team has no members: a user's token names its team. With " +
			"app_id, only what happened within that app. Spend is the usage service's, or with none the request " +
			"rows summed." + serverSide,
		Errors: errs,
	}
	huma.Register(api, list, func(ctx context.Context, request *ListTeamsParams) (*teamsResponse, error) {
		organization, _, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.listTeams(ctx, organization, s.billedOrganization(ctx), request.AppId)
	})
	list.OperationID, list.Path = "manageListTeams", "/v1/manage/teams"
	list.Description = strings.TrimSuffix(list.Description, serverSide) + " The organization is the one named." +
		"\n\nStream staff only, for support: it needs the manage key."
	list.Tags, list.Security = []string{staffTag}, manageSecurity
	huma.Register(api, list, func(ctx context.Context, request *manageListTeamsRequest) (*teamsResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.listTeams(ctx, request.OrganizationId, true, request.AppId)
	})
}

// accountCaller is the organization a customer's account read is about, refused when nothing
// bills it.
func (s *Server) accountCaller(ctx context.Context) (string, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return "", errMissingCustomer
	}
	organization := OrganizationFrom(ctx)
	if organization == "" || s.policies == nil {
		return "", errBillingNotEnabled
	}
	return organization, nil
}

func (s *Server) getAccount(ctx context.Context, _ *struct{}) (*accountResponse, error) {
	organization, err := s.accountCaller(ctx)
	if err != nil {
		return nil, err
	}
	read, ok, err := s.policies.Account(ctx, organization)
	if !ok {
		return nil, s.usageFailure(err, "account", organization)
	}
	held := s.policies.Held(organization)
	threshold, source := read.LowBalanceMicros, read.LowBalanceSource
	if source == "" {
		source = warningFromTier
	}
	if s.configs != nil {
		own, err := s.configs.Policy(ctx, store.ScopeOrganization, organization)
		if err != nil {
			return nil, err
		}
		if own.BalanceWarningMicros != nil {
			threshold, source = own.BalanceWarningMicros, warningFromOrganizationPolicy
		}
	}
	balance := read.BalanceMicros - held.BalanceMicros
	account := Account{
		OrganizationId: organization, Status: string(read.Status), Tier: cmp.Or(read.OrganizationTier, read.Tier.Name),
		BalanceMicros: balance, GrantedMicros: read.GrantedMicros, SpentMicros: read.SpentMicros,
		BalanceWarningSource: source, BalanceState: string(policy.StateOf(balance, threshold)),
		Limits: read.Limits, OwnLimits: read.OwnLimits,
		SpendHardLimit: hardLimitUsage(read.Limits, read.SpendWindows, held, time.Now()),
	}
	if threshold != nil {
		account.BalanceWarningMicros = *threshold
	}
	if len(account.OwnLimits) == 0 {
		account.OwnLimits = nil
	}
	return &accountResponse{Body: account}, nil
}

func (s *Server) listCredits(ctx context.Context, request *listCreditsRequest) (*creditPageResponse, error) {
	organization, err := s.accountCaller(ctx)
	if err != nil {
		return nil, err
	}
	limit := request.Limit
	if limit == 0 {
		limit = defaultCredits
	}
	read, ok, err := s.policies.Credits(ctx, organization, limit, request.Cursor)
	var refused *usage.StatusError
	if errors.As(err, &refused) && refused.Status == http.StatusBadRequest && request.Cursor != "" {
		return nil, errCreditsCursor
	}
	if !ok {
		return nil, s.usageFailure(err, "credits", organization)
	}
	page := CreditPage{Items: make([]CreditEntry, 0, len(read.Items)), HasMore: read.HasMore}
	for _, entry := range read.Items {
		page.Items = append(page.Items, CreditEntry{
			Id: entry.ID, Kind: string(entry.Kind), AmountMicros: entry.AmountMicros, Reference: entry.Reference, CreatedAt: entry.CreatedAt,
		})
	}
	if read.NextCursor != "" {
		page.NextCursor = &read.NextCursor
	}
	return &creditPageResponse{Body: page}, nil
}

// usageFailure is the answer to a read of the usage service that did not answer, logged with
// what was read: 404 billing_not_enabled with no usage service, else 503.
func (s *Server) usageFailure(err error, read, organization string) error {
	if err == nil {
		return errBillingNotEnabled
	}
	s.logger.Error("could not read an organization's "+read+" from the usage service", "organization", organization, "error", err)
	return errUsageUnavailable
}

// teamSpend is what one team spent, across the organization or within one app, in each
// interval's current window, and the apps whose work carried it this month.
type teamSpend struct {
	spent map[usage.Interval]int64
	apps  map[string]map[usage.Interval]int64
	// carried are the apps whose work carried it this month.
	carried map[string]bool
}

// listTeams is every team of the organization a limit names or this month's work carried, by
// name: within app alone when it is set.
func (s *Server) listTeams(ctx context.Context, organization string, billed bool, app string) (*teamsResponse, error) {
	if s.store == nil {
		return nil, errNoLimits
	}
	now := time.Now()
	limits, err := s.store.LimitsFor(ctx, []string{organization})
	if err != nil {
		return nil, err
	}
	spent, err := s.teamSpend(ctx, organization, billed, now)
	if err != nil {
		return nil, err
	}
	limited := map[string][]store.Limit{}
	for _, row := range limits {
		name, ok := usage.TeamOf(row.Category)
		if !ok || (app != "" && (row.Subject != usage.SubjectAppCategory || row.AppID != app)) {
			continue
		}
		limited[name] = append(limited[name], row)
	}
	names := map[string]bool{}
	for name := range limited {
		names[name] = true
	}
	for name, team := range spent {
		if (app == "" && len(team.carried) > 0) || team.carried[app] {
			names[name] = true
		}
	}
	teams := make([]Team, 0, len(names))
	for _, name := range slices.Sorted(func(yield func(string) bool) {
		for name := range names {
			if !yield(name) {
				return
			}
		}
	}) {
		team := Team{Name: name, Apps: []string{}, Limits: []Limit{}}
		found := spent[name]
		windows := map[usage.Interval]int64{}
		if found != nil {
			windows = found.spent
			if app != "" {
				windows = found.apps[app]
			}
			for carried := range found.carried {
				if app == "" || carried == app {
					team.Apps = append(team.Apps, carried)
				}
			}
			slices.Sort(team.Apps)
		}
		for _, interval := range usage.Intervals {
			start, end := usage.Window(interval, now)
			team.Spent = append(team.Spent, WindowSpend{Interval: LimitInterval(interval), WindowStart: start, WindowEnd: end, SpentMicros: windows[interval]})
		}
		rows := limited[name]
		slices.SortFunc(rows, func(a, b store.Limit) int { return cmp.Compare(a.ID, b.ID) })
		for _, row := range rows {
			team.Limits = append(team.Limits, limitOf(row, limits))
		}
		teams = append(teams, team)
	}
	return &teamsResponse{Body: teams}, nil
}

// teamSpend is what each team of the organization spent in the current windows, by name: from
// the usage service when it bills the organization, else summed from the request rows.
func (s *Server) teamSpend(ctx context.Context, organization string, billed bool, now time.Time) (map[string]*teamSpend, error) {
	teams := map[string]*teamSpend{}
	team := func(name string) *teamSpend {
		if teams[name] == nil {
			teams[name] = &teamSpend{spent: map[usage.Interval]int64{}, apps: map[string]map[usage.Interval]int64{}, carried: map[string]bool{}}
		}
		return teams[name]
	}
	current := func(window usage.WindowSpend) bool {
		start, _ := usage.Window(window.Interval, now)
		return window.WindowStart.Equal(start)
	}
	if billed && s.policies != nil {
		read, ok, err := s.policies.Teams(ctx, organization)
		if err != nil {
			return nil, s.usageFailure(err, "teams", organization)
		}
		if ok {
			for _, spent := range read.Teams {
				found := team(spent.Team)
				for _, window := range spent.Windows {
					if current(window) {
						found.spent[window.Interval] = window.SpentMicros
					}
				}
				for _, app := range spent.Apps {
					found.carried[app.AppID] = true
					found.apps[app.AppID] = map[usage.Interval]int64{}
					for _, window := range app.Windows {
						if current(window) {
							found.apps[app.AppID][window.Interval] = window.SpentMicros
						}
					}
				}
			}
			return teams, nil
		}
	}
	since := make([]time.Time, len(usage.Intervals))
	for i, interval := range usage.Intervals {
		since[i], _ = usage.Window(interval, now)
	}
	month, _ := usage.Window(usage.Monthly, now)
	rows, err := s.store.TeamSpendSince(ctx, organization, since, month)
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		name, _ := usage.TeamOf(row.Category)
		found := team(name)
		if found.apps[row.AppID] == nil {
			found.apps[row.AppID] = map[usage.Interval]int64{}
		}
		for i, interval := range usage.Intervals {
			found.spent[interval] += row.Spent[i]
			found.apps[row.AppID][interval] += row.Spent[i]
		}
		if row.ThisMonth {
			found.carried[row.AppID] = true
		}
	}
	return teams, nil
}
