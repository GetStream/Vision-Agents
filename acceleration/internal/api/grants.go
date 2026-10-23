package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// The codes the grant operations answer beside the limits' own.
const (
	codeInvalidExpiry  = "invalid_expiry"
	codeTooManyGrants  = "too_many_grants"
	codeGrantNotFound  = "grant_not_found"
	longestGrant       = 366 * 24 * time.Hour
	grantReasonMaximum = 500
)

var errNoSuchGrant = APIError{Type: ErrorTypeNotFound, Code: codeGrantNotFound, Message: "there is no such grant in the organization"}

// LimitGrantState is where a grant stands.
type LimitGrantState string

const (
	GrantLive    LimitGrantState = "live"
	GrantUsed    LimitGrantState = "used"
	GrantExpired LimitGrantState = "expired"
	GrantRevoked LimitGrantState = "revoked"
)

func (LimitGrantState) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "LimitGrantState", namedEnum(registry, "LimitGrantState",
		"Where a grant stands: live (it is drawn once a limit of its subject is spent), used (it paid "+
			"for its amount), expired (past expires_at) or revoked.",
		string(GrantLive), string(GrantUsed), string(GrantExpired), string(GrantRevoked)))
}

// LimitGrant is a one-time amount one subject may spend past its own spent limits.
type LimitGrant struct {
	Id              int64           `json:"id"`
	Subject         LimitSubject    `json:"subject"`
	AppId           string          `json:"app_id,omitempty"`
	ApiKey          string          `json:"api_key,omitempty"`
	UserId          string          `json:"user_id,omitempty"`
	Team            string          `json:"team,omitempty"`
	Category        string          `json:"category,omitempty"`
	AmountMicros    int64           `json:"amount_micros"`
	UsedMicros      int64           `json:"used_micros" doc:"What the work admitted on it has cost: the whole billable cost of every row drawn from it, as the usage records have them."`
	RemainingMicros int64           `json:"remaining_micros" doc:"amount_micros less used_micros, never below 0."`
	State           LimitGrantState `json:"state"`
	ExpiresAt       *time.Time      `json:"expires_at,omitempty"`
	RevokedAt       *time.Time      `json:"revoked_at,omitempty"`
	RevokedBy       string          `json:"revoked_by,omitempty"`
	Reason          string          `json:"reason"`
	CreatedBy       string          `json:"created_by" doc:"Who made it, as the audit log names them."`
	CreatedAt       time.Time       `json:"created_at"`
}

func (*LimitGrant) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A one-time amount one subject may spend past its own limits once one of " +
		"them is spent, until it is used up, expires or is revoked. It is not credit: the " +
		"organization's balance still pays, and credit, the operator's spend hard limit and the " +
		"limits of every subject that contains this one still refuse. Work admitted on it draws its " +
		"whole cost from it, so the last request can take it a little past its amount."
	return schema
}

// LimitGrantInput is a grant to make.
type LimitGrantInput struct {
	LimitTarget
	AmountMicros int64      `json:"amount_micros" minimum:"1" maximum:"1000000000000000"`
	ExpiresAt    *time.Time `json:"expires_at,omitempty" doc:"When it ends, in the future and at most a year away. Left out, it ends when used up or revoked. The next midnight UTC makes it more for today."`
	Reason       string     `json:"reason" minLength:"1" maxLength:"500"`
}

func (*LimitGrantInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A one-time grant to one subject, which names the subject's own ids, never a default's.")
}

// ManageLimitGrantInput is a grant staff make for an organization.
type ManageLimitGrantInput struct {
	LimitGrantInput
	OrganizationId string `json:"organization_id" minLength:"1" maxLength:"255"`
}

func (*ManageLimitGrantInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A one-time grant staff make for an organization, for support. It is the customer's like any other.")
}

// LimitGrantRevocation says why a grant ends.
type LimitGrantRevocation struct {
	Reason string `json:"reason" minLength:"1" maxLength:"500"`
}

func (*LimitGrantRevocation) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "Why a grant ends now.")
}

// LimitGrantPage is one page of grants, newest first.
type LimitGrantPage struct {
	Items      []LimitGrant `json:"items"`
	HasMore    bool         `json:"has_more"`
	NextCursor *string      `json:"next_cursor,omitempty"`
}

// ListGrantsParams pick the grants a list answers.
type ListGrantsParams struct {
	Subject  LimitSubject    `query:"subject"`
	AppId    string          `query:"app_id"`
	ApiKey   string          `query:"api_key"`
	UserId   string          `query:"user_id"`
	Team     string          `query:"team"`
	Category string          `query:"category"`
	State    LimitGrantState `query:"state" doc:"Only grants in this state. Omitted is every one."`
	Limit    int             `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor   string          `query:"cursor" doc:"The next_cursor of the previous page, sent with the same filters."`
}

type manageListGrantsRequest struct {
	ListGrantsParams
	OrganizationId string `query:"organization_id" required:"true"`
}

type grantIDRequest struct {
	GrantId int64 `path:"grant_id"`
	Body    LimitGrantRevocation
}

type createGrantRequest struct{ Body LimitGrantInput }
type manageCreateGrantRequest struct{ Body ManageLimitGrantInput }

type grantResponse struct{ Body LimitGrant }
type createdGrantResponse struct {
	Status int
	Body   LimitGrant
}
type grantPageResponse struct{ Body LimitGrantPage }

// grantPosition is where a page of grants ends.
type grantPosition struct {
	ID int64 `json:"id"`
}

func (s *Server) registerLimitGrants(api huma.API) {
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound}
	serverSide := "\n\nServer-side only: one server-side credential of any app grants for any subject of " +
		"its organization."
	staff := "\n\nStream staff only, for support: it needs the manage key, and is audited with source " +
		"manage. A grant staff make is the customer's like any other."
	op := func(id, method, path, summary, description string, extra ...int) huma.Operation {
		return huma.Operation{
			OperationID: id, Method: method, Path: path, Summary: summary, Description: description + serverSide,
			Errors: append(slices.Clone(errs), extra...),
		}
	}
	manage := func(operation huma.Operation) huma.Operation {
		operation.OperationID = "manage" + strings.ToUpper(operation.OperationID[:1]) + operation.OperationID[1:]
		operation.Path = "/v1/manage" + strings.TrimPrefix(operation.Path, "/v1")
		operation.Description = strings.TrimSuffix(operation.Description, serverSide) + staff
		operation.Tags, operation.Security = []string{staffTag}, manageSecurity
		return operation
	}
	created := map[string]*huma.Response{"201": {Description: "The grant was made"}}

	create := op("createLimitGrant", http.MethodPost, "/v1/spend/limits/grants", "Grant a subject a one-time amount past its limits",
		"Once one of the subject's own limits is spent, in any interval, its work is admitted on the grant "+
			"until the grant is used up, expires or is revoked; work that fits under every limit draws nothing. "+
			"It lifts the subject's own limits only: a parent's limit, credit and the operator's spend hard "+
			"limit still refuse. The subject must have a limit (404 no_limit); expires_at must be in the "+
			"future and at most a year away (400 invalid_expiry); an organization has at most 20 live "+
			"grants (409 too_many_grants). The router that made it draws on it at once, the others within 5 s.",
		http.StatusConflict)
	create.DefaultStatus, create.Responses = http.StatusCreated, created
	huma.Register(api, create, func(ctx context.Context, request *createGrantRequest) (*createdGrantResponse, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.createGrant(ctx, organization, request.Body, audit)
	})
	manageCreate := manage(create)
	manageCreate.Responses = created
	huma.Register(api, manageCreate, func(ctx context.Context, request *manageCreateGrantRequest) (*createdGrantResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		organization := strings.TrimSpace(request.Body.OrganizationId)
		if organization == "" {
			return nil, errManageOrganization
		}
		return s.createGrant(ctx, organization, request.Body.LimitGrantInput, s.manageAudit(ctx, organization))
	})

	list := op("listLimitGrants", http.MethodGet, "/v1/spend/limits/grants", "List the organization's grants",
		"Newest first, revoked, expired and used ones included, each with what it paid for and what it has left.")
	huma.Register(api, list, func(ctx context.Context, request *ListGrantsParams) (*grantPageResponse, error) {
		organization, _, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.listGrants(ctx, organization, request)
	})
	huma.Register(api, manage(list), func(ctx context.Context, request *manageListGrantsRequest) (*grantPageResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.listGrants(ctx, request.OrganizationId, &request.ListGrantsParams)
	})

	revoke := op("revokeLimitGrant", http.MethodPost, "/v1/spend/limits/grants/{grant_id}/revoke", "Revoke a grant",
		"The next admission no longer draws on it; work already admitted finishes and still counts. The "+
			"grant and what it paid for stay. Revoking it again changes nothing.")
	huma.Register(api, revoke, func(ctx context.Context, request *grantIDRequest) (*grantResponse, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.revokeGrant(ctx, organization, request, audit)
	})
	huma.Register(api, manage(revoke), func(ctx context.Context, request *grantIDRequest) (*grantResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.revokeGrant(ctx, "", request, LimitAuditOf(managedBy(ctx), RequestIDFrom(ctx), ""))
	})
}

// grantOf is a stored grant as the API reads it at now.
func grantOf(grant store.LimitGrant, now time.Time) LimitGrant {
	used := grant.UsedMicros
	subject, team, category := apiSubject(grant.Subject, grant.Category)
	answer := LimitGrant{
		Id: grant.ID, Subject: subject, AppId: grant.AppID, ApiKey: grant.APIKey, UserId: grant.UserID, Team: team,
		Category: category, AmountMicros: grant.AmountMicros, UsedMicros: used, RemainingMicros: max(grant.AmountMicros-used, 0),
		ExpiresAt: grant.ExpiresAt, RevokedAt: grant.RevokedAt, RevokedBy: grant.RevokedBy, Reason: grant.Reason,
		CreatedBy: grant.Actor, CreatedAt: grant.CreatedAt,
	}
	switch {
	case grant.RevokedAt != nil:
		answer.State = GrantRevoked
	case grant.ExpiresAt != nil && !now.Before(*grant.ExpiresAt):
		answer.State = GrantExpired
	case used >= grant.AmountMicros:
		answer.State = GrantUsed
	default:
		answer.State = GrantLive
	}
	return answer
}

func (s *Server) createGrant(ctx context.Context, organization string, sent LimitGrantInput, audit store.LimitAudit) (*createdGrantResponse, error) {
	ref, err := usageTarget(sent.LimitTarget, organization)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	if sent.ExpiresAt != nil && (!sent.ExpiresAt.After(now) || sent.ExpiresAt.Sub(now) > longestGrant) {
		return nil, APIError{Type: ErrorTypeInvalidRequest, Code: codeInvalidExpiry,
			Message: "expires_at is in the future and at most a year away"}
	}
	limits, err := s.store.LimitsFor(ctx, []string{organization})
	if err != nil {
		return nil, err
	}
	if !slices.ContainsFunc(usage.Intervals, func(interval usage.Interval) bool {
		_, ok := policy.Resolve(limits, ref, interval)
		return ok
	}) {
		return nil, APIError{Type: ErrorTypeNotFound, Code: codeNoLimit,
			Message: "the subject has no limit, so a grant to it would never be drawn"}
	}
	grant := store.LimitGrant{
		OrganizationID: organization, Subject: ref.Subject, AppID: ref.AppID, APIKey: ref.APIKey, UserID: ref.UserID,
		Category: ref.Category, AmountMicros: sent.AmountMicros, ExpiresAt: sent.ExpiresAt, Reason: sent.Reason,
		Actor: actorName(audit),
	}
	err = s.store.AddGrant(ctx, &grant, audit)
	if errors.Is(err, store.ErrTooManyGrants) {
		return nil, APIError{Type: ErrorTypeConflict, Code: codeTooManyGrants, Message: fmt.Sprintf(
			"the organization has %d live grants, as many as it may: revoke one, or wait for one to end", store.MaxLiveGrants)}
	}
	if err != nil {
		return nil, err
	}
	s.policies.LimitsChanged(ctx, organization)
	return &createdGrantResponse{Status: http.StatusCreated, Body: grantOf(grant, now)}, nil
}

func (s *Server) listGrants(ctx context.Context, organization string, request *ListGrantsParams) (*grantPageResponse, error) {
	before, err := decodeCursor[grantPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	now := time.Now()
	filter := store.GrantFilter{
		OrganizationID: organization, AppID: request.AppId, APIKey: request.ApiKey, UserID: request.UserId,
		Category: request.Category, Now: now,
	}
	switch request.Subject {
	case LimitSubjectTeam, LimitSubjectAppTeam:
		filter.Subject = usage.SubjectCategory
		if request.Subject == LimitSubjectAppTeam {
			filter.Subject = usage.SubjectAppCategory
		}
		filter.CategoryPrefix = usage.TeamPrefix
	case LimitSubjectCategory, LimitSubjectAppCategory:
		filter.Subject, filter.NotCategoryPrefix = usage.Subject(request.Subject), usage.TeamPrefix
	default:
		filter.Subject = usage.Subject(request.Subject)
	}
	if request.Team != "" {
		filter.Category, filter.CategoryPrefix = usage.TeamPrefix+request.Team, ""
	}
	yes, no := true, false
	switch request.State {
	case GrantRevoked:
		filter.Revoked = &yes
	case GrantExpired:
		filter.Revoked, filter.Expired = &no, &yes
	case GrantUsed:
		filter.Revoked, filter.Expired, filter.Used = &no, &no, &yes
	case GrantLive:
		filter.Revoked, filter.Expired, filter.Used = &no, &no, &no
	}
	var position int64
	if before != nil {
		position = before.ID
	}
	rows, err := s.store.ListGrants(ctx, filter, position, request.Limit)
	if err != nil {
		return nil, err
	}
	batch, more := page(rows, store.LimitPage(request.Limit))
	answer := LimitGrantPage{Items: make([]LimitGrant, 0, len(batch)), HasMore: more}
	for _, row := range batch {
		answer.Items = append(answer.Items, grantOf(row, now))
	}
	if more {
		answer.NextCursor = encodeCursor(grantPosition{ID: batch[len(batch)-1].ID})
	}
	return &grantPageResponse{Body: answer}, nil
}

func (s *Server) revokeGrant(ctx context.Context, organization string, request *grantIDRequest, audit store.LimitAudit) (*grantResponse, error) {
	revoked, err := s.store.RevokeGrant(ctx, organization, request.GrantId, actorName(audit), request.Body.Reason, audit)
	if errors.Is(err, store.ErrNoGrant) {
		return nil, errNoSuchGrant
	}
	if err != nil {
		return nil, err
	}
	s.policies.LimitsChanged(ctx, revoked.OrganizationID)
	return &grantResponse{Body: grantOf(revoked, time.Now())}, nil
}

// subjectGrants are the subject's grants live at now, each with what it paid for.
func (s *Server) subjectGrants(ctx context.Context, ref usage.SubjectRef, now time.Time) ([]LimitGrant, error) {
	live, err := s.store.LiveGrants(ctx, []string{ref.OrganizationID}, now)
	if err != nil {
		return nil, err
	}
	live = slices.DeleteFunc(live, func(grant store.LimitGrant) bool { return grant.Ref() != ref })
	grants := make([]LimitGrant, 0, len(live))
	for _, grant := range live {
		grants = append(grants, grantOf(grant, now))
	}
	return grants, nil
}
