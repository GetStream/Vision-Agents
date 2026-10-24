package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// The codes the limits operations answer beside the shared ones.
const (
	codeLimitAboveParent = "limit_above_parent"
	codeLimitExists      = "limit_exists"
	codeNoLimit          = "no_limit"
	codeUseTeam          = "use_team"
	codeLimitNotFound    = "limit_not_found"
)

var (
	errNoLimits    = notConfigured("spend limits are not available: no database configured")
	errNoSuchLimit = APIError{Type: ErrorTypeNotFound, Code: codeLimitNotFound,
		Message: "there is no such limit in the organization"}
	errManageOrganization = invalidRequest("a management call on limits names its organization in organization_id")
)

// What a limit's ids may hold. A key is never an origin (hook:chat), which holds a ':'. A user
// id is held to Stream Chat's rule for user ids: 1 to 255 bytes of ASCII letters, digits, @, _,
// ., - and space.
var (
	limitKeyID  = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)
	limitUserID = regexp.MustCompile(`^[@\w .-]{1,255}$`)
)

// LimitSubject is what a limit is on: whose spend it counts.
type LimitSubject string

const (
	LimitSubjectOrganization LimitSubject = "organization"
	LimitSubjectApp          LimitSubject = "app"
	LimitSubjectAPIKey       LimitSubject = "api_key"
	LimitSubjectUser         LimitSubject = "user"
	LimitSubjectCategory     LimitSubject = "category"
	LimitSubjectAppCategory  LimitSubject = "app_category"
	LimitSubjectTeam         LimitSubject = "team"
	LimitSubjectAppTeam      LimitSubject = "app_team"
)

// namedConstants names an enum's constants in the generated clients: two enums sharing a value
// make oapi-codegen prefix both, renaming constants callers already use (as ConnectionOwnerType
// explains).
func namedConstants(registry huma.Registry, enum string, ref *huma.Schema) *huma.Schema {
	schema := registry.Map()[enum]
	names := make([]any, len(schema.Enum))
	for i, value := range schema.Enum {
		words := strings.Split(value.(string), "_")
		for j, word := range words {
			words[j] = strings.ToUpper(word[:1]) + word[1:]
		}
		names[i] = enum + strings.Join(words, "")
	}
	schema.Extensions = map[string]any{"x-enum-varnames": names}
	return ref
}

func (LimitSubject) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "LimitSubject", namedEnum(registry, "LimitSubject",
		"Whose spend a limit counts: the organization, one app, a key, an end user of an app, a "+
			"usage category across the organization or within one app, or a team across the "+
			"organization or within one app. A team is the user's team, as its token names it.",
		string(LimitSubjectOrganization), string(LimitSubjectApp), string(LimitSubjectAPIKey), string(LimitSubjectUser),
		string(LimitSubjectCategory), string(LimitSubjectAppCategory), string(LimitSubjectTeam), string(LimitSubjectAppTeam)))
}

// LimitLevel is whose ids a limit names.
type LimitLevel string

func (LimitLevel) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "LimitLevel", namedEnum(registry, "LimitLevel",
		"Whose ids a limit names. A key's or a user's limit at organization or app level is a "+
			"default for every key or user there, each counted on its own; at api_key or user level "+
			"it names one. The most specific level wins. Every other subject has one level: the "+
			"organization, a category and a team are organization level; an app, and a category "+
			"or team within one, app level.",
		string(store.LevelOrganization), string(store.LevelApp), string(store.LevelAPIKey), string(store.LevelUser)))
}

// LimitInterval is the window a limit counts over.
type LimitInterval string

func (LimitInterval) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "LimitInterval", namedEnum(registry, "LimitInterval",
		"The window a limit counts spend over, in UTC: the day, the week from Monday, 14 days "+
			"counted from Monday 2024-01-01, or the calendar month.",
		string(usage.Daily), string(usage.Weekly), string(usage.Biweekly), string(usage.Monthly)))
}

// Limit is one of the organization's spend limits.
type Limit struct {
	Id                   int64         `json:"id"`
	Subject              LimitSubject  `json:"subject"`
	Level                LimitLevel    `json:"level"`
	AppId                string        `json:"app_id,omitempty" doc:"The app, for a limit at app, api_key or user level, or on an app's category or team."`
	ApiKey               string        `json:"api_key,omitempty" doc:"The key, for a limit at api_key level."`
	UserId               string        `json:"user_id,omitempty" doc:"The end user, for a limit at user level."`
	Team                 string        `json:"team,omitempty" doc:"The team, for a team's limit."`
	Category             string        `json:"category,omitempty" doc:"The usage category, for a category's limit."`
	Interval             LimitInterval `json:"interval"`
	LimitMicros          int64         `json:"limit_micros" doc:"What the subject may spend in each window, in millionths of a dollar."`
	EffectiveLimitMicros int64         `json:"effective_limit_micros" doc:"The lowest of limit_micros and the limits, in the same interval, of every subject that contains this one: what the subject can spend before something refuses it."`
	Note                 string        `json:"note,omitempty"`
	UpdatedBy            string        `json:"updated_by" doc:"Who last wrote it, as the audit log names them."`
	CreatedAt            time.Time     `json:"created_at"`
	UpdatedAt            time.Time     `json:"updated_at"`
}

func (*Limit) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A spend limit: what one subject may spend in each window of its interval. " +
		"Billable work for a subject with no room left under any of its limits is refused, 402 " +
		"spend_limit_reached, or 429 user_limit_reached with Retry-After for an end user's. A " +
		"subject is also held to every subject that contains it, so an app's work is refused " +
		"once its organization's limit is spent."
	return schema
}

// LimitTarget names the subject a write is about.
type LimitTarget struct {
	Subject  LimitSubject `json:"subject"`
	AppId    string       `json:"app_id,omitempty" maxLength:"255" doc:"The app. Required at app, api_key and user level, and for an app's category or team."`
	ApiKey   string       `json:"api_key,omitempty" maxLength:"128" doc:"The key: 1 to 128 characters of A-Z, a-z, 0-9, _ and -."`
	UserId   string       `json:"user_id,omitempty" maxLength:"255" doc:"The end user, as the app's tokens name them, with Stream Chat's rule for user ids: 1 to 255 characters of A-Z, a-z, 0-9, @, _, ., - and space."`
	Team     string       `json:"team,omitempty" maxLength:"59" doc:"The team, for subject team or app_team: 1 to 59 characters of A-Z, a-z, 0-9, _ . and -."`
	Category string       `json:"category,omitempty" maxLength:"64" doc:"The usage category, for subject category or app_category: 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -. A team is subject team, never a category starting team:."`
}

// LimitInput is a limit to create.
type LimitInput struct {
	LimitTarget
	Level       LimitLevel    `json:"level,omitempty" doc:"Left out, the level of the most specific id given: api_key or user when it is named, else app when app_id is, else organization."`
	Interval    LimitInterval `json:"interval"`
	LimitMicros int64         `json:"limit_micros" minimum:"0" maximum:"1000000000000000" doc:"What the subject may spend in each window, in millionths of a dollar. 0 refuses every billable request."`
	Note        string        `json:"note,omitempty" maxLength:"500"`
}

// closed refuses a field the schema does not define, so a limit is never written from a field
// a caller thought meant something.
func closed(schema *huma.Schema, description string) *huma.Schema {
	schema.Description = description
	schema.AdditionalProperties = false
	return schema
}

func (*LimitInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A spend limit to create. Every limit is the customer's; an unknown field is refused.")
}

// ManageLimitInput is a limit staff create for an organization.
type ManageLimitInput struct {
	LimitInput
	OrganizationId string `json:"organization_id" minLength:"1" maxLength:"255" doc:"The organization the limit is for."`
}

func (*ManageLimitInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A spend limit staff create for an organization, for support. It is the customer's like any other.")
}

// LimitChange is what an update changes: the amount and the note. What a limit is on never
// changes; delete it and create another.
type LimitChange struct {
	LimitMicros *int64  `json:"limit_micros,omitempty" minimum:"0" maximum:"1000000000000000"`
	Note        *string `json:"note,omitempty" maxLength:"500"`
}

func (*LimitChange) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "What changes about a spend limit: its amount and its note. A field left out keeps what is stored.")
}

// LimitPage is one page of limits, by id.
type LimitPage struct {
	Items      []Limit `json:"items"`
	HasMore    bool    `json:"has_more"`
	NextCursor *string `json:"next_cursor,omitempty"`
}

// limitPosition is where a page of limits ends.
type limitPosition struct {
	ID int64 `json:"id"`
}

// LimitInForce is the limit a subject has in one interval.
type LimitInForce struct {
	Id                   int64      `json:"id"`
	Level                LimitLevel `json:"level"`
	LimitMicros          int64      `json:"limit_micros"`
	EffectiveLimitMicros int64      `json:"effective_limit_micros"`
}

// LimitWindow is one subject's current window of one interval.
type LimitWindow struct {
	Interval       LimitInterval `json:"interval"`
	WindowStart    time.Time     `json:"window_start"`
	WindowEnd      time.Time     `json:"window_end"`
	SpentMicros    int64         `json:"spent_micros" doc:"What the subject spent in the window, as the usage records count it."`
	Limit          *LimitInForce `json:"limit,omitempty" doc:"The limit in force, absent when the subject has none in this interval."`
	BaselineMicros int64         `json:"baseline_micros" doc:"What the subject had spent in the window when it was last reset; its limit counts spend past this."`
	RoomMicros     *int64        `json:"room_micros,omitempty" doc:"What the subject may still spend under its own limit in the window: the limit less what it spent since its baseline. Absent with no limit."`
	OnGrant        bool          `json:"on_grant" doc:"The limit is spent and the subject's work is admitted on one of its grants."`
}

// SpendHardLimitWindow is the operator's limit on the organization in one window.
type SpendHardLimitWindow struct {
	LimitMicros int64     `json:"limit_micros"`
	SpentMicros int64     `json:"spent_micros"`
	RoomMicros  int64     `json:"room_micros"`
	WindowEnd   time.Time `json:"window_end"`
}

// SpendHardLimitUsage is the operator's spend hard limit on the organization.
type SpendHardLimitUsage struct {
	Daily   *SpendHardLimitWindow `json:"daily,omitempty"`
	Monthly *SpendHardLimitWindow `json:"monthly,omitempty"`
}

func (*SpendHardLimitUsage) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The most this organization may spend in the UTC day and in the calendar " +
		"month, whatever its credit and its own limits. Set by the operator, never by the " +
		"customer. A missing interval has no limit."
	return schema
}

// SubjectUsage is what one subject spent in its current windows, against its limits.
type SubjectUsage struct {
	Subject        LimitSubject         `json:"subject"`
	AppId          string               `json:"app_id,omitempty"`
	ApiKey         string               `json:"api_key,omitempty"`
	UserId         string               `json:"user_id,omitempty"`
	Team           string               `json:"team,omitempty"`
	Category       string               `json:"category,omitempty"`
	Windows        []LimitWindow        `json:"windows" doc:"The current window of every interval, daily first."`
	Grants         []LimitGrant         `json:"grants" doc:"The subject's live grants, which its work is admitted on once one of its limits is spent."`
	SpendHardLimit *SpendHardLimitUsage `json:"spend_hard_limit,omitempty"`
}

// LimitResetInput resets a subject's spend for its limits.
type LimitResetInput struct {
	LimitTarget
	Intervals []LimitInterval `json:"intervals,omitempty" maxItems:"4" doc:"The windows to reset. Left out, every one."`
	Reason    string          `json:"reason" minLength:"1" maxLength:"500"`
}

func (*LimitResetInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A reset of one subject's spend for its limits, in the current windows of the intervals named.")
}

// ManageLimitResetInput is a reset staff make for an organization.
type ManageLimitResetInput struct {
	LimitResetInput
	OrganizationId string `json:"organization_id" minLength:"1" maxLength:"255"`
}

func (*ManageLimitResetInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	return closed(schema, "A reset staff make of one subject's spend for its limits, for support.")
}

// ListLimitsParams pick the limits a list answers.
type ListLimitsParams struct {
	Subject  LimitSubject  `query:"subject" doc:"Only limits on this subject. team lists the teams' limits, category every other category's."`
	Level    LimitLevel    `query:"level"`
	AppId    string        `query:"app_id"`
	ApiKey   string        `query:"api_key"`
	UserId   string        `query:"user_id"`
	Team     string        `query:"team"`
	Category string        `query:"category"`
	Interval LimitInterval `query:"interval"`
	Limit    int           `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor   string        `query:"cursor" doc:"The next_cursor of the previous page, sent with the same filters."`
}

type manageListLimitsRequest struct {
	ListLimitsParams
	OrganizationId string `query:"organization_id" required:"true"`
}

// LimitUsageParams name the one subject a usage read is of.
type LimitUsageParams struct {
	Subject  LimitSubject `query:"subject" required:"true"`
	AppId    string       `query:"app_id"`
	ApiKey   string       `query:"api_key"`
	UserId   string       `query:"user_id"`
	Team     string       `query:"team"`
	Category string       `query:"category"`
}

type manageLimitUsageRequest struct {
	LimitUsageParams
	OrganizationId string `query:"organization_id" required:"true"`
}

type limitIDRequest struct {
	LimitId int64 `path:"limit_id"`
}

type createLimitRequest struct{ Body LimitInput }
type manageCreateLimitRequest struct{ Body ManageLimitInput }

type updateLimitRequest struct {
	LimitId int64 `path:"limit_id"`
	Body    LimitChange
}

type resetLimitUsageRequest struct{ Body LimitResetInput }
type manageResetLimitUsageRequest struct{ Body ManageLimitResetInput }

type limitResponse struct{ Body Limit }
type createdLimitResponse struct {
	Status int
	Body   Limit
}
type limitPageResponse struct{ Body LimitPage }
type subjectUsageResponse struct{ Body SubjectUsage }

func (s *Server) registerLimits(api huma.API) {
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound}
	serverSide := "\n\nServer-side only: one server-side credential of any app changes every limit of its " +
		"organization, and a request that is not server-side, reads included, answers 403 " +
		"server_side_only."
	staff := "\n\nStream staff only, for support: it needs the manage key, and is audited with source " +
		"manage. A limit staff write is the customer's like any other, and the customer can change it."
	created := map[string]*huma.Response{"201": {Description: "The limit was created"}}
	op := func(id, method, path, summary, description string, extra ...int) huma.Operation {
		return huma.Operation{
			OperationID: id, Method: method, Path: path, Summary: summary, Description: description + serverSide,
			Errors: append(slices.Clone(errs), extra...),
		}
	}
	manage := func(operation huma.Operation) huma.Operation {
		operation.OperationID = "manage" + strings.ToUpper(operation.OperationID[:1]) + operation.OperationID[1:]
		operation.Path = "/v1/manage" + strings.TrimPrefix(operation.Path, "/v1")
		operation.Description = strings.ReplaceAll(strings.TrimSuffix(operation.Description, serverSide),
			"the caller's organization", "the organization named, or any organization's by id") + staff
		operation.Tags, operation.Security = []string{staffTag}, manageSecurity
		return operation
	}

	list := op("listLimits", http.MethodGet, "/v1/spend/limits", "List the organization's spend limits",
		"Every limit of the caller's organization, by id, with what each can spend once its parents are counted.")
	huma.Register(api, list, func(ctx context.Context, request *ListLimitsParams) (*limitPageResponse, error) {
		organization, _, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.listLimits(ctx, organization, request)
	})
	huma.Register(api, manage(list), func(ctx context.Context, request *manageListLimitsRequest) (*limitPageResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.listLimits(ctx, request.OrganizationId, &request.ListLimitsParams)
	})

	create := op("createLimit", http.MethodPost, "/v1/spend/limits", "Create a spend limit",
		"A limit on one subject, at one level, over one interval; a subject has one per level and interval, "+
			"so a second answers 409 limit_exists naming the first. A limit above the limit, in the same "+
			"interval, of a subject that contains it (the organization contains its apps and categories; an "+
			"app its keys, users and categories; an organization's category the same category in each app) "+
			"answers 400 limit_above_parent naming that parent. Lowering a parent below a child is allowed, "+
			"and the child's effective_limit_micros shows it. A category starting team: answers 400 use_team.",
		http.StatusConflict)
	create.DefaultStatus = http.StatusCreated
	create.Responses = created
	huma.Register(api, create, func(ctx context.Context, request *createLimitRequest) (*createdLimitResponse, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.createLimit(ctx, organization, request.Body, audit)
	})
	manageCreate := manage(create)
	manageCreate.Responses = created
	huma.Register(api, manageCreate, func(ctx context.Context, request *manageCreateLimitRequest) (*createdLimitResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		organization := strings.TrimSpace(request.Body.OrganizationId)
		if organization == "" {
			return nil, errManageOrganization
		}
		return s.createLimit(ctx, organization, request.Body.LimitInput, s.manageAudit(ctx, organization))
	})

	get := op("getLimit", http.MethodGet, "/v1/spend/limits/{limit_id}", "Read a spend limit", "One limit of the caller's organization.")
	huma.Register(api, get, func(ctx context.Context, request *limitIDRequest) (*limitResponse, error) {
		organization, _, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.getLimit(ctx, organization, request.LimitId)
	})
	huma.Register(api, manage(get), func(ctx context.Context, request *limitIDRequest) (*limitResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.getLimit(ctx, "", request.LimitId)
	})

	update := op("updateLimit", http.MethodPatch, "/v1/spend/limits/{limit_id}", "Change a spend limit's amount or note",
		"Only the amount and the note change. A higher amount is held to the parents' as a new limit is "+
			"(400 limit_above_parent). The router that wrote it holds to it at once, every other within 5 s.")
	huma.Register(api, update, func(ctx context.Context, request *updateLimitRequest) (*limitResponse, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.updateLimit(ctx, organization, request.LimitId, request.Body, audit)
	})
	huma.Register(api, manage(update), func(ctx context.Context, request *updateLimitRequest) (*limitResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.updateLimit(ctx, "", request.LimitId, request.Body, LimitAuditOf(managedBy(ctx), RequestIDFrom(ctx), ""))
	})

	remove := op("deleteLimit", http.MethodDelete, "/v1/spend/limits/{limit_id}", "Delete a spend limit",
		"The subject is no longer held to it; a default at a level above takes over where there is one.")
	remove.DefaultStatus = http.StatusNoContent
	remove.Responses = map[string]*huma.Response{"204": {Description: "The limit was deleted"}}
	huma.Register(api, remove, func(ctx context.Context, request *limitIDRequest) (*struct{}, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return nil, s.deleteLimit(ctx, organization, request.LimitId, audit)
	})
	manageRemove := manage(remove)
	manageRemove.Responses = remove.Responses
	huma.Register(api, manageRemove, func(ctx context.Context, request *limitIDRequest) (*struct{}, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return nil, s.deleteLimit(ctx, "", request.LimitId, LimitAuditOf(managedBy(ctx), RequestIDFrom(ctx), ""))
	})

	read := op("getLimitUsage", http.MethodGet, "/v1/spend/limits/usage", "Read what a subject spent against its limits",
		"The subject's current window in every interval: what it spent, the limit in force with its "+
			"effective amount, its baseline since a reset and its room, and once for the organization the "+
			"operator's spend hard limit, which is what admission refuses first. Spend is read from the "+
			"usage records, so it is what admission counts within a few seconds.")
	huma.Register(api, read, func(ctx context.Context, request *LimitUsageParams) (*subjectUsageResponse, error) {
		organization, _, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.limitUsage(ctx, organization, s.billedOrganization(ctx), request)
	})
	huma.Register(api, manage(read), func(ctx context.Context, request *manageLimitUsageRequest) (*subjectUsageResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		return s.limitUsage(ctx, request.OrganizationId, true, &request.LimitUsageParams)
	})

	reset := op("resetLimitUsage", http.MethodPost, "/v1/spend/limits/usage/reset", "Reset a subject's spend for its limits",
		"From now until each window ends, the subject's limits count only what it spends after the reset. "+
			"The usage records, the credit and the operator's spend hard limit are untouched, so usage and "+
			"invoices still show everything. Work recorded just before the reset that reaches the usage "+
			"records after it counts after it, which can refuse a few seconds of in-flight spend. A subject "+
			"with no limit answers 404 no_limit.")
	huma.Register(api, reset, func(ctx context.Context, request *resetLimitUsageRequest) (*subjectUsageResponse, error) {
		organization, audit, err := s.limitsCaller(ctx)
		if err != nil {
			return nil, err
		}
		return s.resetLimitUsage(ctx, organization, s.billedOrganization(ctx), request.Body, audit)
	})
	huma.Register(api, manage(reset), func(ctx context.Context, request *manageResetLimitUsageRequest) (*subjectUsageResponse, error) {
		if s.store == nil {
			return nil, errNoLimits
		}
		organization := strings.TrimSpace(request.Body.OrganizationId)
		if organization == "" {
			return nil, errManageOrganization
		}
		return s.resetLimitUsage(ctx, organization, true, request.Body.LimitResetInput, s.manageAudit(ctx, organization))
	})
}

// limitsCaller is the organization a customer's limits call is about and who it is recorded
// as made by. An app seen in no organization, as with a self-hosted router's own keys, is its
// own.
func (s *Server) limitsCaller(ctx context.Context) (string, store.LimitAudit, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", store.LimitAudit{}, errMissingCustomer
	}
	if s.store == nil {
		return "", store.LimitAudit{}, errNoLimits
	}
	organization := OrganizationFrom(ctx)
	if organization == "" {
		organization = customerID
	}
	return organization, LimitAuditOf(ActorFrom(ctx), RequestIDFrom(ctx), customerID), nil
}

// billedOrganization reports whether the caller's organization is one the usage service bills,
// rather than an app that is its own.
func (s *Server) billedOrganization(ctx context.Context) bool { return OrganizationFrom(ctx) != "" }

// manageAudit is who a management write on an organization's limits is recorded as made by.
// The audit log keeps it under the organization, since staff act as no app.
func (s *Server) manageAudit(ctx context.Context, organization string) store.LimitAudit {
	return LimitAuditOf(managedBy(ctx), RequestIDFrom(ctx), organization)
}

// LimitAuditOf is a write of limits recorded against customer by actor. An empty customer is
// filled in with the organization of the row written, for a management call by id.
func LimitAuditOf(actor Actor, requestID, customer string) store.LimitAudit {
	return store.LimitAudit{CustomerID: customer, Source: actor.Source, ActorID: actor.ID, ActorName: actor.Name, RequestID: requestID}
}

// actorName is who a write is by, as a row's updated_by and a reset's actor keep it.
func actorName(audit store.LimitAudit) string {
	if audit.ActorID != "" {
		return audit.Source + ":" + audit.ActorID
	}
	return audit.Source
}

// target is the subject, the level and the ids a write names, as the table keeps them. A team
// is kept as the category team:<name>.
func target(sent LimitTarget, level LimitLevel, organization string) (usage.SubjectRef, store.LimitLevel, error) {
	ref := usage.SubjectRef{OrganizationID: organization, AppID: strings.TrimSpace(sent.AppId), APIKey: sent.ApiKey, UserID: sent.UserId}
	if _, team := usage.TeamOf(sent.Category); team {
		return ref, "", APIError{Type: ErrorTypeInvalidRequest, Code: codeUseTeam,
			Message: fmt.Sprintf("category %q names a team: send subject team or app_team with team", sent.Category)}
	}
	switch sent.Subject {
	case LimitSubjectTeam, LimitSubjectAppTeam:
		if sent.Category != "" {
			return ref, "", invalidRequest("a team's limit names its team in team, not category")
		}
		if err := usage.CheckCategories([]string{usage.TeamPrefix + sent.Team}); err != nil || sent.Team == "" {
			return ref, "", invalidRequest(fmt.Sprintf("team %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . and -", sent.Team, usage.MaxTeamLength))
		}
		ref.Category = usage.TeamPrefix + sent.Team
		ref.Subject = usage.SubjectCategory
		if sent.Subject == LimitSubjectAppTeam {
			ref.Subject = usage.SubjectAppCategory
		}
	case LimitSubjectCategory, LimitSubjectAppCategory:
		if sent.Team != "" {
			return ref, "", invalidRequest("team is for subject team or app_team")
		}
		if err := usage.CheckCategories([]string{sent.Category}); err != nil || sent.Category == "" {
			return ref, "", invalidRequest(fmt.Sprintf("category %q is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -", sent.Category))
		}
		ref.Category, ref.Subject = sent.Category, usage.Subject(sent.Subject)
	case LimitSubjectOrganization, LimitSubjectApp, LimitSubjectAPIKey, LimitSubjectUser:
		if sent.Team != "" || sent.Category != "" {
			return ref, "", invalidRequest("team and category are only for a team's or a category's limit")
		}
		ref.Subject = usage.Subject(sent.Subject)
	default:
		return ref, "", invalidRequest(fmt.Sprintf("subject %q is not one a limit is on", sent.Subject))
	}
	chosen := store.LimitLevel(level)
	if chosen == "" {
		chosen = impliedLevel(ref)
	}
	if !slices.Contains(store.LimitLevels(ref.Subject), chosen) {
		return ref, "", invalidRequest(fmt.Sprintf("a limit on %s is not at level %s", sent.Subject, chosen))
	}
	// The ids are the level's: a default names no key or user, and the organization's level no app.
	wants := map[string]bool{
		"app_id":  chosen != store.LevelOrganization,
		"api_key": chosen == store.LevelAPIKey,
		"user_id": chosen == store.LevelUser,
	}
	for name, value := range map[string]string{"app_id": ref.AppID, "api_key": ref.APIKey, "user_id": ref.UserID} {
		switch {
		case wants[name] && value == "":
			return ref, "", invalidRequest(fmt.Sprintf("a limit on %s at level %s names %s", sent.Subject, chosen, name))
		case !wants[name] && value != "":
			return ref, "", invalidRequest(fmt.Sprintf("a limit on %s at level %s names no %s", sent.Subject, chosen, name))
		}
	}
	if ref.AppID != "" && usage.IDProblem(ref.AppID) != "" {
		return ref, "", invalidRequest("app_id holds a character an id may not, or is over 255 bytes")
	}
	if ref.APIKey != "" && !limitKeyID.MatchString(ref.APIKey) {
		return ref, "", invalidRequest("api_key is 1 to 128 characters of A-Z, a-z, 0-9, _ and -, and an origin such as hook:chat is no key")
	}
	if ref.UserID != "" && (!limitUserID.MatchString(ref.UserID) || usage.IDProblem(ref.UserID) != "") {
		return ref, "", invalidRequest("user_id is 1 to 255 characters of A-Z, a-z, 0-9, @, _, ., - and space, as Stream Chat's user ids are")
	}
	return ref, chosen, nil
}

// impliedLevel is the level of the most specific id a write names.
func impliedLevel(ref usage.SubjectRef) store.LimitLevel {
	switch {
	case ref.Subject == usage.SubjectAPIKey && ref.APIKey != "":
		return store.LevelAPIKey
	case ref.Subject == usage.SubjectUser && ref.UserID != "":
		return store.LevelUser
	case ref.Subject == usage.SubjectApp || ref.Subject == usage.SubjectAppCategory || ref.AppID != "":
		return store.LevelApp
	}
	return store.LevelOrganization
}

// parents are the subjects that contain the subjects a row limits, nearest first: what its
// amount is held to.
func parents(subject usage.Subject, level store.LimitLevel, organization, app, category string) []usage.SubjectRef {
	org := usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectOrganization}
	ofApp := usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectApp, AppID: app}
	switch subject {
	case usage.SubjectApp, usage.SubjectCategory:
		return []usage.SubjectRef{org}
	case usage.SubjectAPIKey, usage.SubjectUser:
		if level == store.LevelOrganization {
			return []usage.SubjectRef{org}
		}
		return []usage.SubjectRef{ofApp, org}
	case usage.SubjectAppCategory:
		return []usage.SubjectRef{ofApp, {OrganizationID: organization, Subject: usage.SubjectCategory, Category: category}, org}
	}
	return nil
}

// aboveParent is the refusal of an amount above a parent's limit in the same interval, or nil.
func aboveParent(limits []store.Limit, row store.Limit) error {
	for _, parent := range parents(row.Subject, row.Level, row.OrganizationID, row.AppID, row.Category) {
		found, ok := policy.Resolve(limits, parent, row.Interval)
		if !ok || row.LimitMicros <= found.LimitMicros {
			continue
		}
		return APIError{Type: ErrorTypeInvalidRequest, Code: codeLimitAboveParent, Message: fmt.Sprintf(
			"limit_micros %d is above the %s limit of %d on %s (limit %d): a subject never gets more than one that contains it",
			row.LimitMicros, row.Interval, found.LimitMicros, describedParent(parent), found.ID)}
	}
	return nil
}

func describedParent(ref usage.SubjectRef) string {
	switch ref.Subject {
	case usage.SubjectApp:
		return "app " + ref.AppID
	case usage.SubjectCategory:
		if team, ok := usage.TeamOf(ref.Category); ok {
			return "team " + team
		}
		return "category " + ref.Category
	}
	return "the organization"
}

// effective is the lowest of a row's own amount and its parents' in the same interval.
func effective(limits []store.Limit, row store.Limit) int64 {
	lowest := row.LimitMicros
	for _, parent := range parents(row.Subject, row.Level, row.OrganizationID, row.AppID, row.Category) {
		if found, ok := policy.Resolve(limits, parent, row.Interval); ok {
			lowest = min(lowest, found.LimitMicros)
		}
	}
	return lowest
}

// limitOf is a stored row as the API reads it: a team's category back as its team.
func limitOf(row store.Limit, limits []store.Limit) Limit {
	subject, team, category := apiSubject(row.Subject, row.Category)
	return Limit{
		Id: row.ID, Subject: subject, Level: LimitLevel(row.Level), AppId: row.AppID, ApiKey: row.APIKey, UserId: row.UserID,
		Team: team, Category: category, Interval: LimitInterval(row.Interval), LimitMicros: row.LimitMicros,
		EffectiveLimitMicros: effective(limits, row), Note: row.Note, UpdatedBy: row.UpdatedBy,
		CreatedAt: row.CreatedAt, UpdatedAt: row.UpdatedAt,
	}
}

// apiSubject is a stored subject as the API names it, with a team out of its category.
func apiSubject(subject usage.Subject, category string) (LimitSubject, string, string) {
	if team, ok := usage.TeamOf(category); ok {
		if subject == usage.SubjectAppCategory {
			return LimitSubjectAppTeam, team, ""
		}
		return LimitSubjectTeam, team, ""
	}
	return LimitSubject(subject), "", category
}

func (s *Server) listLimits(ctx context.Context, organization string, request *ListLimitsParams) (*limitPageResponse, error) {
	after, err := decodeCursor[limitPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	filter := store.LimitFilter{
		OrganizationID: organization, Level: store.LimitLevel(request.Level), AppID: request.AppId, APIKey: request.ApiKey,
		UserID: request.UserId, Category: request.Category, Interval: usage.Interval(request.Interval),
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
	var position int64
	if after != nil {
		position = after.ID
	}
	rows, err := s.store.ListLimits(ctx, filter, position, request.Limit)
	if err != nil {
		return nil, err
	}
	all, err := s.store.LimitsFor(ctx, []string{organization})
	if err != nil {
		return nil, err
	}
	rows, more := page(rows, store.LimitPage(request.Limit))
	answer := LimitPage{Items: make([]Limit, len(rows)), HasMore: more}
	for i, row := range rows {
		answer.Items[i] = limitOf(row, all)
	}
	if more {
		answer.NextCursor = encodeCursor(limitPosition{ID: rows[len(rows)-1].ID})
	}
	return &limitPageResponse{Body: answer}, nil
}

func (s *Server) createLimit(ctx context.Context, organization string, sent LimitInput, audit store.LimitAudit) (*createdLimitResponse, error) {
	ref, level, err := target(sent.LimitTarget, sent.Level, organization)
	if err != nil {
		return nil, err
	}
	interval := usage.Interval(sent.Interval)
	if !interval.Valid() {
		return nil, invalidRequest(fmt.Sprintf("interval %q is not daily, weekly, biweekly or monthly", sent.Interval))
	}
	row := store.Limit{
		Subject: ref.Subject, Level: level, OrganizationID: organization, AppID: ref.AppID, APIKey: ref.APIKey,
		UserID: ref.UserID, Category: ref.Category, Interval: interval, LimitMicros: sent.LimitMicros, Note: sent.Note,
		UpdatedBy: actorName(audit),
	}
	all, err := s.store.LimitsFor(ctx, []string{organization})
	if err != nil {
		return nil, err
	}
	if err := aboveParent(all, row); err != nil {
		return nil, err
	}
	err = s.store.CreateLimit(ctx, &row, audit)
	var exists *store.ErrLimitExists
	if errors.As(err, &exists) {
		return nil, APIError{Type: ErrorTypeConflict, Code: codeLimitExists, Message: fmt.Sprintf(
			"the subject has a %s limit at that level already: limit %d; change it instead", interval, exists.ID)}
	}
	if err != nil {
		return nil, err
	}
	s.policies.LimitsChanged(ctx, organization)
	return &createdLimitResponse{Status: http.StatusCreated, Body: limitOf(row, append(all, row))}, nil
}

func (s *Server) getLimit(ctx context.Context, organization string, id int64) (*limitResponse, error) {
	row, err := s.store.Limit(ctx, organization, id)
	if errors.Is(err, store.ErrNoLimit) {
		return nil, errNoSuchLimit
	}
	if err != nil {
		return nil, err
	}
	all, err := s.store.LimitsFor(ctx, []string{row.OrganizationID})
	if err != nil {
		return nil, err
	}
	return &limitResponse{Body: limitOf(row, all)}, nil
}

func (s *Server) updateLimit(ctx context.Context, organization string, id int64, change LimitChange, audit store.LimitAudit) (*limitResponse, error) {
	existing, err := s.store.Limit(ctx, organization, id)
	if errors.Is(err, store.ErrNoLimit) {
		return nil, errNoSuchLimit
	}
	if err != nil {
		return nil, err
	}
	if audit.CustomerID == "" {
		audit.CustomerID = existing.OrganizationID
	}
	all, err := s.store.LimitsFor(ctx, []string{existing.OrganizationID})
	if err != nil {
		return nil, err
	}
	if change.LimitMicros != nil && *change.LimitMicros > existing.LimitMicros {
		raised := existing
		raised.LimitMicros = *change.LimitMicros
		if err := aboveParent(all, raised); err != nil {
			return nil, err
		}
	}
	updated, err := s.store.UpdateLimit(ctx, existing.OrganizationID, id,
		store.LimitChange{LimitMicros: change.LimitMicros, Note: change.Note, UpdatedBy: actorName(audit)}, audit)
	if errors.Is(err, store.ErrNoLimit) {
		return nil, errNoSuchLimit
	}
	if err != nil {
		return nil, err
	}
	s.policies.LimitsChanged(ctx, existing.OrganizationID)
	for i := range all {
		if all[i].ID == id {
			all[i] = updated
		}
	}
	return &limitResponse{Body: limitOf(updated, all)}, nil
}

func (s *Server) deleteLimit(ctx context.Context, organization string, id int64, audit store.LimitAudit) error {
	existing, err := s.store.Limit(ctx, organization, id)
	if errors.Is(err, store.ErrNoLimit) {
		return errNoSuchLimit
	}
	if err != nil {
		return err
	}
	if audit.CustomerID == "" {
		audit.CustomerID = existing.OrganizationID
	}
	_, err = s.store.DeleteLimit(ctx, existing.OrganizationID, id, audit)
	if errors.Is(err, store.ErrNoLimit) {
		return errNoSuchLimit
	}
	if err != nil {
		return err
	}
	s.policies.LimitsChanged(ctx, existing.OrganizationID)
	return nil
}

// usageTarget is the subject a read or a reset names, whose ids are exactly the subject's own:
// a usage read is of one subject, never a default.
func usageTarget(sent LimitTarget, organization string) (usage.SubjectRef, error) {
	level := store.LevelOrganization
	switch sent.Subject {
	case LimitSubjectApp, LimitSubjectAppCategory, LimitSubjectAppTeam:
		level = store.LevelApp
	case LimitSubjectAPIKey:
		level = store.LevelAPIKey
	case LimitSubjectUser:
		level = store.LevelUser
	}
	ref, _, err := target(sent, LimitLevel(level), organization)
	return ref, err
}

func (s *Server) limitUsage(ctx context.Context, organization string, billed bool, request *LimitUsageParams) (*subjectUsageResponse, error) {
	ref, err := usageTarget(LimitTarget{
		Subject: request.Subject, AppId: request.AppId, ApiKey: request.ApiKey, UserId: request.UserId,
		Team: request.Team, Category: request.Category,
	}, organization)
	if err != nil {
		return nil, err
	}
	answer, err := s.subjectUsage(ctx, ref, billed)
	if err != nil {
		return nil, err
	}
	return &subjectUsageResponse{Body: answer}, nil
}

// subjectUsage is what a subject spent in its current windows against its limits, with the
// operator's spend hard limit on the organization.
func (s *Server) subjectUsage(ctx context.Context, ref usage.SubjectRef, billed bool) (SubjectUsage, error) {
	if s.policies == nil {
		return SubjectUsage{}, errNoLimits
	}
	limits, err := s.store.LimitsFor(ctx, []string{ref.OrganizationID})
	if err != nil {
		return SubjectUsage{}, err
	}
	now := time.Now()
	earliest := now
	for _, interval := range usage.Intervals {
		if start, _ := usage.Window(interval, now); start.Before(earliest) {
			earliest = start
		}
	}
	resets, err := s.store.Resets(ctx, []string{ref.OrganizationID}, earliest)
	if err != nil {
		return SubjectUsage{}, err
	}
	spent, err := s.policies.SubjectSpend(ctx, ref, billed)
	if err != nil {
		return SubjectUsage{}, stack.Wrap(err)
	}
	grants, err := s.subjectGrants(ctx, ref, now)
	if err != nil {
		return SubjectUsage{}, err
	}
	drawable := slices.ContainsFunc(grants, func(grant LimitGrant) bool { return grant.State == GrantLive })
	subject, team, category := apiSubject(ref.Subject, ref.Category)
	answer := SubjectUsage{Subject: subject, AppId: ref.AppID, ApiKey: ref.APIKey, UserId: ref.UserID, Team: team, Category: category, Grants: grants}
	for _, interval := range usage.Intervals {
		start, end := usage.Window(interval, now)
		window := LimitWindow{Interval: LimitInterval(interval), WindowStart: start, WindowEnd: end}
		if read, ok := spent[interval]; ok && read.WindowStart.Equal(start) {
			window.SpentMicros = read.SpentMicros
		}
		if limit, ok := policy.Resolve(limits, ref, interval); ok {
			window.BaselineMicros = policy.Baseline(resets, ref, interval, start)
			window.Limit = &LimitInForce{Id: limit.ID, Level: LimitLevel(limit.Level), LimitMicros: limit.LimitMicros, EffectiveLimitMicros: effective(limits, limit)}
			room := policy.Room(limit, window.BaselineMicros, window.SpentMicros)
			window.RoomMicros = &room
			window.OnGrant = room <= 0 && drawable
		}
		answer.Windows = append(answer.Windows, window)
	}
	if billed {
		standing, ok, err := s.policies.Standing(ctx, ref.OrganizationID)
		if err != nil {
			return SubjectUsage{}, stack.Wrap(err)
		}
		if ok {
			answer.SpendHardLimit = hardLimitUsage(standing, now)
		}
	}
	return answer, nil
}

// hardLimitUsage is the operator's spend hard limit as the standing has it, or nil when it sets
// none.
func hardLimitUsage(standing usage.Standing, now time.Time) *SpendHardLimitUsage {
	window := func(interval usage.Interval, spent usage.WindowSpend) *SpendHardLimitWindow {
		limit, ok := standing.Limits.SpendHardLimitMicros(interval)
		if !ok {
			return nil
		}
		start, end := usage.Window(interval, now)
		micros := int64(0)
		if spent.WindowStart.Equal(start) {
			micros = spent.SpentMicros
		}
		return &SpendHardLimitWindow{LimitMicros: limit, SpentMicros: micros, RoomMicros: limit - micros, WindowEnd: end}
	}
	answer := &SpendHardLimitUsage{
		Daily:   window(usage.Daily, standing.SpendWindows.Daily),
		Monthly: window(usage.Monthly, standing.SpendWindows.Monthly),
	}
	if answer.Daily == nil && answer.Monthly == nil {
		return nil
	}
	return answer
}

func (s *Server) resetLimitUsage(ctx context.Context, organization string, billed bool, sent LimitResetInput, audit store.LimitAudit) (*subjectUsageResponse, error) {
	ref, err := usageTarget(sent.LimitTarget, organization)
	if err != nil {
		return nil, err
	}
	intervals := usage.Intervals
	if len(sent.Intervals) > 0 {
		intervals = nil
		for _, interval := range sent.Intervals {
			if !usage.Interval(interval).Valid() {
				return nil, invalidRequest(fmt.Sprintf("interval %q is not daily, weekly, biweekly or monthly", interval))
			}
			if !slices.Contains(intervals, usage.Interval(interval)) {
				intervals = append(intervals, usage.Interval(interval))
			}
		}
	}
	limits, err := s.store.LimitsFor(ctx, []string{organization})
	if err != nil {
		return nil, err
	}
	limited := slices.ContainsFunc(usage.Intervals, func(interval usage.Interval) bool {
		_, ok := policy.Resolve(limits, ref, interval)
		return ok
	})
	if !limited {
		return nil, APIError{Type: ErrorTypeNotFound, Code: codeNoLimit, Message: "the subject has no limit, so there is nothing to reset"}
	}
	spent, err := s.policies.SubjectSpend(ctx, ref, billed)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	now := time.Now()
	resets := make([]store.UsageReset, 0, len(intervals))
	for _, interval := range intervals {
		start, _ := usage.Window(interval, now)
		baseline := int64(0)
		if read, ok := spent[interval]; ok && read.WindowStart.Equal(start) {
			baseline = read.SpentMicros
		}
		resets = append(resets, store.UsageReset{
			OrganizationID: organization, Subject: ref.Subject, AppID: ref.AppID, APIKey: ref.APIKey, UserID: ref.UserID,
			Category: ref.Category, Interval: interval, WindowStart: start, BaselineMicros: baseline,
			Reason: sent.Reason, Actor: actorName(audit),
		})
	}
	if err := s.store.AddResets(ctx, resets, audit); err != nil {
		return nil, err
	}
	s.policies.LimitsChanged(ctx, organization)
	answer, err := s.subjectUsage(ctx, ref, billed)
	if err != nil {
		return nil, err
	}
	return &subjectUsageResponse{Body: answer}, nil
}
