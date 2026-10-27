package server

import (
	"context"
	"net/http"
	"slices"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

// customerNote ends every operation of the customer view.
const customerNote = "\n\nThe customer view: what an organization's own dashboard reads through its router, with the " +
	"ingest token, never the manage key. It writes nothing, and leaves out what only the operator reads: why an " +
	"organization is suspended, who wrote each credit entry and why, and each usage event's tier, cost at the " +
	"vendor's rates, price version and producer."

// registerAccounts declares the customer view of the accounts.
func (s *Server) registerAccounts(api huma.API) {
	security := []map[string][]string{{ingestScheme: {}}}
	failures := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusServiceUnavailable}
	huma.Register(api, huma.Operation{
		OperationID: "readAccount", Method: http.MethodGet, Path: usage.PathAccounts + "/{organization_id}",
		Summary: "Read an organization's account, as the organization may",
		Description: "The organization's standing without `status_reason`, with the low-balance threshold in force " +
			"(the account's own, else its tier's) and whose it is, and `own_limits`, every setting the operator " +
			"set for the organization alone. An organization with no account is active with nothing granted or " +
			"spent." + customerNote,
		Tags: []string{tag}, Security: security, Errors: failures,
	}, s.readAccount)
	huma.Register(api, huma.Operation{
		OperationID: "readAccountCredits", Method: http.MethodGet, Path: usage.PathAccounts + "/{organization_id}/credits",
		Summary: "List an organization's credit entries, as the organization may",
		Description: "Newest first, without `actor`, `note` or `idempotency_key`; `reference` only on the entries " +
			"the payment processor wrote (actor `" + usage.StripeActor + "`), so no staff reference reaches the " +
			"customer." + customerNote,
		Tags: []string{tag}, Security: security, Errors: failures,
	}, s.readAccountCredits)
	huma.Register(api, huma.Operation{
		OperationID: "readAccountTeams", Method: http.MethodGet, Path: usage.PathAccounts + "/{organization_id}/teams",
		Summary: "Read what an organization's teams spent",
		Description: "Every team whose work was recorded this calendar month (UTC), by name: its category " +
			"`team:<name>` across the organization in the current window of every interval, and the same within " +
			"each app whose work carried it this month. A window nobody spent in is at zero." + customerNote,
		Tags: []string{tag}, Security: security, Errors: failures,
	}, s.readAccountTeams)
	huma.Register(api, huma.Operation{
		OperationID: "readAccountUsage", Method: http.MethodGet, Path: usage.PathAccounts + "/{organization_id}/usage",
		Summary: "List an organization's usage events, as the organization may",
		Description: "The events `listUsage` answers, with its filters, order (newest first) and cursor, without " +
			"`organization_id`, `organization_tier`, `base_cost_micros`, `price_version` or `source`. `team` is the " +
			"category `team:<name>`, so it cannot be given with `category`; `user_id` needs `app_id`, since a user id " +
			"is only unique within its app." + customerNote,
		Tags: []string{tag}, Security: security, Errors: failures,
	}, s.readAccountUsage)
}

type readAccountUsageOutput struct {
	Body usage.CustomerUsagePage
}

func (s *Server) readAccountUsage(ctx context.Context, input *listUsageInput) (*readAccountUsageOutput, error) {
	listed, err := s.listUsage(ctx, input)
	if err != nil {
		return nil, err
	}
	page := usage.CustomerUsagePage{Items: make([]usage.CustomerUsageEvent, 0, len(listed.Body.Items)),
		HasMore: listed.Body.HasMore, NextCursor: listed.Body.NextCursor}
	for _, event := range listed.Body.Items {
		page.Items = append(page.Items, usage.CustomerUsageEventOf(event))
	}
	return &readAccountUsageOutput{Body: page}, nil
}

type readAccountInput struct {
	OrganizationPath
	OrganizationTier string `query:"organization_tier" doc:"The tier to answer on; absent is the tier the organization's newest event carried."`
}

type readAccountOutput struct {
	Body usage.Account
}

func (s *Server) readAccount(ctx context.Context, input *readAccountInput) (*readAccountOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	standing, err := s.standing(ctx, input.OrganizationID, input.OrganizationTier)
	if err != nil {
		return nil, err
	}
	return &readAccountOutput{Body: usage.CustomerAccount(standing.Body)}, nil
}

type readAccountCreditsOutput struct {
	Body usage.CustomerCreditPage
}

func (s *Server) readAccountCredits(ctx context.Context, input *listCreditsInput) (*readAccountCreditsOutput, error) {
	listed, err := s.listCredits(ctx, input)
	if err != nil {
		return nil, err
	}
	page := usage.CustomerCreditPage{Items: make([]usage.CustomerCredit, 0, len(listed.Body.Items)),
		HasMore: listed.Body.HasMore, NextCursor: listed.Body.NextCursor}
	for _, entry := range listed.Body.Items {
		page.Items = append(page.Items, usage.CustomerCreditOf(entry))
	}
	return &readAccountCreditsOutput{Body: page}, nil
}

type readAccountTeamsOutput struct {
	Body usage.Teams
}

func (s *Server) readAccountTeams(ctx context.Context, input *OrganizationPath) (*readAccountTeamsOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	asOf := s.now().UTC()
	windows, err := s.ledger.TeamWindows(ctx, input.OrganizationID)
	if err != nil {
		return nil, s.failed(err, "read team windows")
	}
	return &readAccountTeamsOutput{Body: usage.Teams{AsOf: asOf, Teams: teamsOf(windows)}}, nil
}

// teamsOf groups the windows of team subjects by team, and within a team by app, by name.
func teamsOf(windows []ledger.UsageWindow) []usage.TeamSpend {
	byName := map[string]*usage.TeamSpend{}
	apps := map[string]map[string]*usage.TeamAppSpend{}
	for _, window := range windows {
		name, _ := usage.TeamOf(window.Category)
		team := byName[name]
		if team == nil {
			team = &usage.TeamSpend{Team: name, Windows: []usage.WindowSpend{}, Apps: []usage.TeamAppSpend{}}
			byName[name] = team
			apps[name] = map[string]*usage.TeamAppSpend{}
		}
		if window.Subject == usage.SubjectCategory {
			team.Windows = append(team.Windows, window.Spend())
			continue
		}
		app := apps[name][window.AppID]
		if app == nil {
			app = &usage.TeamAppSpend{AppID: window.AppID}
			apps[name][window.AppID] = app
		}
		app.Windows = append(app.Windows, window.Spend())
	}
	teams := make([]usage.TeamSpend, 0, len(byName))
	for name, team := range byName {
		for _, app := range apps[name] {
			team.Apps = append(team.Apps, *app)
		}
		slices.SortFunc(team.Apps, func(a, b usage.TeamAppSpend) int { return strings.Compare(a.AppID, b.AppID) })
		teams = append(teams, *team)
	}
	slices.SortFunc(teams, func(a, b usage.TeamSpend) int { return strings.Compare(a.Team, b.Team) })
	return teams
}
