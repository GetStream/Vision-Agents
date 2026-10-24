package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// errUserPolicyQuotaOnly answers a user's policy that sets anything but a quota.
var errUserPolicyQuotaOnly = invalidRequest("a user's policy holds a quota only: data_policy, prompt_injection, " +
	"allowed_models, tags and require_own_stream_app are set on the app or the organization")

// UserPolicy is one end user's own quota, what they are held to, and what they spent today.
type UserPolicy struct {
	UserId    string         `json:"user_id"`
	Quota     *Quota         `json:"quota,omitempty" doc:"The user's own quota. Left out, the user has none of their own and is held to the defaults."`
	Effective EffectiveQuota `json:"effective"`
	Messages  int64          `json:"messages" doc:"Responses counted today toward a quota the customer set, the app's backend acting for the user included. A day held only to the router's default counts the user's devices alone, which this leaves out."`
	Tokens    int64          `json:"tokens" doc:"Tokens counted today toward a quota the customer set, counted as messages is."`
	ResetsAt  time.Time      `json:"resets_at" doc:"When the day ends and the count starts again: the next UTC midnight."`
}

func (*UserPolicy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One end user of the calling app: their own quota, if any, the quota they are " +
		"held to, and what they spent today."
	return schema
}

// EffectiveQuota is the quota a user is held to, each number and who set it.
type EffectiveQuota struct {
	MessagesPerUserPerDay int64      `json:"messages_per_user_per_day" doc:"Responses the user may ask for today. 0 is no cap."`
	MessagesSetBy         QuotaLevel `json:"messages_set_by"`
	TokensPerUserPerDay   int64      `json:"tokens_per_user_per_day" doc:"Tokens the user's responses may cost today. 0 is no cap."`
	TokensSetBy           QuotaLevel `json:"tokens_set_by"`
}

func (*EffectiveQuota) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The quota a user is held to: each number is the first set of the user's own, " +
		"the app's, the organization's and the router's default. Both are 0 on a router that counts nothing."
	return schema
}

// QuotaLevel is who set one number of a quota.
type QuotaLevel string

const (
	QuotaLevelUser         QuotaLevel = "user"
	QuotaLevelApp          QuotaLevel = "app"
	QuotaLevelOrganization QuotaLevel = "organization"
	QuotaLevelDefault      QuotaLevel = "default"
)

func (QuotaLevel) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "QuotaLevel", namedEnum(registry, "QuotaLevel", "Who set a number of "+
		"a quota: the user's own policy, the app's, the organization's, or the router's default. The "+
		"router's default counts the user's devices only; every other counts the app's backend acting "+
		"for the user too.",
		string(QuotaLevelUser), string(QuotaLevelApp), string(QuotaLevelOrganization), string(QuotaLevelDefault)))
}

// UserPolicyInput is what a user's policy is replaced with.
type UserPolicyInput struct {
	Quota *Quota `json:"quota,omitempty" doc:"The user's own quota. Left out, null, or setting neither number, removes the user's own and the defaults apply."`
}

func (*UserPolicyInput) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One end user's own policy, which holds a quota only."
	schema.AdditionalProperties = false
	return schema
}

// UserPolicyEntry is one user with a policy of their own, as a list names them.
type UserPolicyEntry struct {
	UserId    string    `json:"user_id"`
	Quota     *Quota    `json:"quota,omitempty"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UserPolicyPage is a page of the app's users with a policy of their own.
type UserPolicyPage struct {
	Items      []UserPolicyEntry `json:"items"`
	HasMore    bool              `json:"has_more"`
	NextCursor *string           `json:"next_cursor,omitempty"`
}

// userPolicyPosition is where a page of users ends.
type userPolicyPosition struct {
	UserID string `json:"u"`
}

// ListUserPoliciesParams page through the app's users with a policy.
type ListUserPoliciesParams struct {
	Limit  int    `query:"limit" minimum:"1" maximum:"100" doc:"Up to 100. Omitted is 100."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page."`
}

type userIDRequest struct {
	UserId string `path:"user_id" doc:"The end user, by the id the app's tokens name them with."`
}

type updateUserPolicyRequest struct {
	UserId string `path:"user_id" doc:"The end user, by the id the app's tokens name them with."`
	Body   UserPolicyInput
}

type userPolicyResponse struct{ Body UserPolicy }
type userPolicyPageResponse struct{ Body UserPolicyPage }

// PolicyScopeName is which policy a staff operation reads or writes.
type PolicyScopeName string

func (PolicyScopeName) Schema(registry huma.Registry) *huma.Schema {
	return namedConstants(registry, "PolicyScope", namedEnum(registry, "PolicyScope",
		"Whose policy: an organization's, an app's, or one end user's.",
		string(store.ScopeOrganization), string(store.ScopeApp), string(store.ScopeUser)))
}

type customerPolicyRequest struct {
	OrganizationId string          `path:"organization_id"`
	Scope          PolicyScopeName `path:"scope"`
	Id             string          `path:"id" doc:"The organization id, the app id, or <app id>/<user id> for a user, URL-encoded."`
}

type updateCustomerPolicyRequest struct {
	OrganizationId string          `path:"organization_id"`
	Scope          PolicyScopeName `path:"scope"`
	Id             string          `path:"id" doc:"The organization id, the app id, or <app id>/<user id> for a user, URL-encoded."`
	Body           Policy
}

func (s *Server) registerUserPolicies(api huma.API) {
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden}
	serverSide := "\n\nServer-side only: it needs a server-side token, so it cannot be reached from an end user's device."
	huma.Register(api, huma.Operation{
		OperationID: "listUserPolicies",
		Method:      http.MethodGet,
		Path:        "/v1/policies/users",
		Summary:     "List the calling app's users with a policy of their own",
		Description: "By user id, in byte order. A user with no policy of their own is held to the app's " +
			"quota, the organization's, or the router's default, and is not listed." + serverSide,
		Responses: map[string]*huma.Response{"200": {Description: "A page of users"}},
		Errors:    errs,
	}, s.listUserPolicies)
	huma.Register(api, huma.Operation{
		OperationID: "getUserPolicy",
		Method:      http.MethodGet,
		Path:        "/v1/policies/users/{user_id}",
		Summary:     "Read one end user's quota",
		Description: "The user's own quota, if they have one, the quota they are held to and who set each " +
			"number of it, and what they spent today toward a quota the customer set. Any user id may be " +
			"read, with a policy of their own or not." + serverSide,
		Responses: map[string]*huma.Response{"200": {Description: "The user's quota"}},
		Errors:    errs,
	}, s.getUserPolicy)
	huma.Register(api, huma.Operation{
		OperationID: "updateUserPolicy",
		Method:      http.MethodPut,
		Path:        "/v1/policies/users/{user_id}",
		Summary:     "Replace one end user's own quota",
		Description: "The user's own quota wins over the app's, the organization's and the router's " +
			"default, each number on its own, and counts the app's backend acting for the user as well " +
			"as the user's devices. A quota left out, null, or setting neither number, removes the user's own. " +
			"A user's policy holds a quota only, so any other field is a 400. The router that wrote it " +
			"holds to it at once, every other within 5 s." + serverSide,
		Responses: map[string]*huma.Response{"200": {Description: "The user's quota as it now stands"}},
		Errors:    errs,
	}, s.updateUserPolicy)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteUserPolicy",
		Method:        http.MethodDelete,
		Path:          "/v1/policies/users/{user_id}",
		Summary:       "Remove one end user's own quota",
		Description:   "The user falls back to the app's quota, the organization's, or the router's default. A user with none of their own answers 204 too." + serverSide,
		DefaultStatus: http.StatusNoContent,
		Responses:     map[string]*huma.Response{"204": {Description: "The user has no policy of their own"}},
		Errors:        errs,
	}, s.deleteUserPolicy)

	staff := "\n\nStream staff only, for support: it needs the manage key, and a write is audited with source manage. " +
		"An app or a user of an app that is not the organization's answers 404, as does an organization id " +
		"that is not the one in the path."
	manageErrs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound}
	huma.Register(api, huma.Operation{
		OperationID: "getCustomerPolicy",
		Method:      http.MethodGet,
		Path:        "/v1/manage/organizations/{organization_id}/policies/{scope}/{id}",
		Summary:     "Read a customer's policy",
		Description: "An organization's, an app's or one end user's policy, as the customer's own read answers it." + staff,
		Tags:        []string{staffTag},
		Security:    manageSecurity,
		Responses:   map[string]*huma.Response{"200": {Description: "The policy"}},
		Errors:      manageErrs,
	}, s.getCustomerPolicy)
	huma.Register(api, huma.Operation{
		OperationID: "updateCustomerPolicy",
		Method:      http.MethodPut,
		Path:        "/v1/manage/organizations/{organization_id}/policies/{scope}/{id}",
		Summary:     "Replace a customer's policy",
		Description: "Held to the customer's own rules: a user's policy holds a quota only, a budget is a 400 " +
			"budget_moved (spend limits are /v1/manage/spend/limits), and an organization's " +
			"require_own_stream_app is not changed here. The customer can change what staff wrote. An app's or a " +
			"user's change is audited under the app, so the app's own queryAudit lists it; an organization's is " +
			"audited under the organization id, which no app's queryAudit lists, so only the management audit " +
			"shows it." + staff,
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The policy was stored"}},
		Errors:    manageErrs,
	}, s.updateCustomerPolicy)
}

// userPolicyCaller is the app a user policy call is about.
func (s *Server) userPolicyCaller(ctx context.Context) (string, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", errMissingCustomer
	}
	if s.policies == nil {
		return "", errNoPolicies
	}
	return customerID, nil
}

// checkUserID holds a user id to Stream Chat's rule, as a spend limit's is.
func checkUserID(userID string) error {
	if !limitUserID.MatchString(userID) || usage.IDProblem(userID) != "" {
		return invalidRequest("user_id is 1 to 255 characters of A-Z, a-z, 0-9, @, _, ., - and space, as Stream Chat's user ids are")
	}
	return nil
}

func (s *Server) listUserPolicies(ctx context.Context, request *ListUserPoliciesParams) (*userPolicyPageResponse, error) {
	appID, err := s.userPolicyCaller(ctx)
	if err != nil {
		return nil, err
	}
	after, err := decodeCursor[userPolicyPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	var position string
	if after != nil {
		position = after.UserID
	}
	rows, err := s.store.UserPolicies(ctx, appID, position, request.Limit)
	if err != nil {
		return nil, err
	}
	limit := request.Limit
	if limit <= 0 || limit > store.UserPolicyLimit {
		limit = store.UserPolicyLimit
	}
	rows, more := page(rows, limit)
	answer := UserPolicyPage{Items: make([]UserPolicyEntry, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		_, userID, _ := store.SplitUserScopeID(row.ScopeID)
		answer.Items = append(answer.Items, UserPolicyEntry{UserId: userID, Quota: quotaFor(row.Document.Quota), UpdatedAt: row.UpdatedAt})
	}
	if more && len(answer.Items) > 0 {
		answer.NextCursor = encodeCursor(userPolicyPosition{UserID: answer.Items[len(answer.Items)-1].UserId})
	}
	return &userPolicyPageResponse{Body: answer}, nil
}

func (s *Server) getUserPolicy(ctx context.Context, request *userIDRequest) (*userPolicyResponse, error) {
	appID, err := s.userPolicyCaller(ctx)
	if err != nil {
		return nil, err
	}
	if err := checkUserID(request.UserId); err != nil {
		return nil, err
	}
	answer, err := s.userPolicyOf(ctx, appID, request.UserId)
	if err != nil {
		return nil, err
	}
	return &userPolicyResponse{Body: answer}, nil
}

func (s *Server) updateUserPolicy(ctx context.Context, request *updateUserPolicyRequest) (*userPolicyResponse, error) {
	appID, err := s.userPolicyCaller(ctx)
	if err != nil {
		return nil, err
	}
	if err := checkUserID(request.UserId); err != nil {
		return nil, err
	}
	if err := s.saveUserPolicy(ctx, appID, request.UserId, request.Body.Quota, appID); err != nil {
		return nil, err
	}
	answer, err := s.userPolicyOf(ctx, appID, request.UserId)
	if err != nil {
		return nil, err
	}
	return &userPolicyResponse{Body: answer}, nil
}

func (s *Server) deleteUserPolicy(ctx context.Context, request *userIDRequest) (*struct{}, error) {
	appID, err := s.userPolicyCaller(ctx)
	if err != nil {
		return nil, err
	}
	if err := checkUserID(request.UserId); err != nil {
		return nil, err
	}
	return nil, s.saveUserPolicy(ctx, appID, request.UserId, nil, appID)
}

// saveUserPolicy replaces a user's own quota, or removes it when sent sets no number, and
// audits the change under auditedAs.
func (s *Server) saveUserPolicy(ctx context.Context, appID, userID string, sent *Quota, auditedAs string) error {
	id := store.UserScopeID(appID, userID)
	before, err := s.configs.Policy(ctx, store.ScopeUser, id)
	if err != nil {
		return err
	}
	after := store.PolicyDocument{Quota: quotaOf(sent)}
	action := store.AuditUpdated
	if after.Quota == nil {
		if _, err := s.policies.Delete(ctx, store.ScopeUser, id); err != nil {
			return err
		}
		action = store.AuditDeleted
	} else if err := s.policies.Save(ctx, store.ScopeUser, id, after); err != nil {
		return err
	}
	s.auditAs(ctx, auditedAs, auditRecord{
		ResourceType: store.AuditPolicy, ResourceID: "user:" + id, ResourceName: string(store.ScopeUser),
		Action: action, Changes: auditDiff(before, after),
	})
	return nil
}

// userPolicyOf is a user's own quota, what they are held to and what they spent today.
func (s *Server) userPolicyOf(ctx context.Context, appID, userID string) (UserPolicy, error) {
	own, err := s.configs.Policy(ctx, store.ScopeUser, store.UserScopeID(appID, userID))
	if err != nil {
		return UserPolicy{}, err
	}
	held := s.quota.Effective(ctx, appID, userID)
	messages, tokens, reset, err := s.quota.Today(ctx, appID, userID)
	if err != nil {
		s.logger.Error("could not read what a user spent today", "customer", appID, "error", err)
	}
	return UserPolicy{
		UserId: userID, Quota: quotaFor(own.Quota),
		Effective: EffectiveQuota{
			MessagesPerUserPerDay: held.Messages.PerDay, MessagesSetBy: quotaLevel(held.Messages.By),
			TokensPerUserPerDay: held.Tokens.PerDay, TokensSetBy: quotaLevel(held.Tokens.By),
		},
		Messages: messages, Tokens: tokens, ResetsAt: reset,
	}, nil
}

func quotaLevel(by quota.Setter) QuotaLevel {
	switch by.Level {
	case quota.LevelUser:
		return QuotaLevelUser
	case quota.LevelApp:
		return QuotaLevelApp
	case quota.LevelOrganization:
		return QuotaLevelOrganization
	}
	return QuotaLevelDefault
}

// customerScope is the scope and id a staff call names, once the id is known to be the
// organization's: the organization itself, one of its apps, or a user of one.
func (s *Server) customerScope(ctx context.Context, organizationID string, scope PolicyScopeName, id string) (store.PolicyScope, string, error) {
	unknown := notFound(fmt.Sprintf("organization %s has no %s %q", organizationID, scope, id))
	switch store.PolicyScope(scope) {
	case store.ScopeOrganization:
		if id != organizationID {
			return "", "", unknown
		}
		return store.ScopeOrganization, id, nil
	case store.ScopeApp:
		return store.ScopeApp, id, s.checkOrganizationApp(ctx, organizationID, id, unknown)
	case store.ScopeUser:
		appID, userID, ok := store.SplitUserScopeID(id)
		if !ok {
			return "", "", invalidRequest("a user's policy is named <app id>/<user id>")
		}
		if err := checkUserID(userID); err != nil {
			return "", "", err
		}
		return store.ScopeUser, id, s.checkOrganizationApp(ctx, organizationID, appID, unknown)
	}
	return "", "", invalidRequest(fmt.Sprintf("scope %q is not organization, app or user", scope))
}

// checkOrganizationApp answers unknown for an app that is not the organization's.
func (s *Server) checkOrganizationApp(ctx context.Context, organizationID, appID string, unknown APIError) error {
	in, err := s.store.AppInOrganization(ctx, appID, organizationID)
	if err != nil {
		return err
	}
	if !in {
		return unknown
	}
	return nil
}

func (s *Server) getCustomerPolicy(ctx context.Context, request *customerPolicyRequest) (*policyResponse, error) {
	if s.policies == nil {
		return nil, errNoPolicies
	}
	scope, id, err := s.customerScope(ctx, request.OrganizationId, request.Scope, request.Id)
	if err != nil {
		return nil, err
	}
	policy, err := s.policyOf(ctx, scope, id)
	if err != nil {
		return nil, err
	}
	return &policyResponse{Body: policy}, nil
}

func (s *Server) updateCustomerPolicy(ctx context.Context, request *updateCustomerPolicyRequest) (*policyResponse, error) {
	if s.policies == nil {
		return nil, errNoPolicies
	}
	scope, id, err := s.customerScope(ctx, request.OrganizationId, request.Scope, request.Id)
	if err != nil {
		return nil, err
	}
	sent := request.Body
	if sent.Budget != nil {
		return nil, errBudgetMoved
	}
	// The audit log keeps an app's and its users' under the app, and an organization's under it.
	auditedAs := id
	if scope == store.ScopeUser {
		appID, userID, _ := store.SplitUserScopeID(id)
		if sent.DataPolicy != nil || sent.PromptInjection != nil || sent.AllowedModels != nil ||
			sent.Tags != nil || sent.RequireOwnStreamApp != nil {
			return nil, errUserPolicyQuotaOnly
		}
		if err := s.saveUserPolicy(ctx, appID, userID, sent.Quota, appID); err != nil {
			return nil, err
		}
		policy, err := s.policyOf(ctx, scope, id)
		if err != nil {
			return nil, err
		}
		return &policyResponse{Body: policy}, nil
	}
	policy, err := s.savePolicy(ctx, scope, id, sent, auditedAs)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}
