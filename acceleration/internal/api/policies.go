package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var (
	errNoPolicies     = notConfigured("policies are not available: no database configured")
	errNoOrganization = APIError{
		Type: ErrorTypeInvalidRequest, Code: codeMissingOrganization,
		Message: "this request names no organization: it needs X-Stream-Organization-Id, " +
			"or an API key belonging to one",
	}

	// errBudgetMoved answers a policy that still sends a budget.
	errBudgetMoved = APIError{
		Type: ErrorTypeInvalidRequest, Code: codeBudgetMoved,
		Message: "spend limits moved to /v1/spend/limits: a policy has no budget",
	}
)

// Policy is what an organization or an app decided about data handling, prompt injection,
// which models may be used and how usage is labelled.
type Policy struct {
	Budget               map[string]any    `json:"budget,omitempty" deprecated:"true" doc:"Removed: spend limits are /v1/spend/limits, at every level. A policy that sends budget is refused with 400 budget_moved, and none is read back."`
	Quota                *Quota            `json:"quota,omitempty"`
	DataPolicy           *DataPolicy       `json:"data_policy,omitempty"`
	PromptInjection      *bool             `json:"prompt_injection,omitempty" doc:"Screen what every LLM response is asked for prompt injection. The newest input - the user's turn and any tool results - goes to a decision model beside the model call, so it adds nothing to time to first token. The end of the response is held until the verdict, and a response whose input reads as an injection fails with prompt_injection before its tool calls can be acted on."`
	AllowedModels        *[]string         `json:"allowed_models,omitempty" example:"[\"deepseek/DeepSeek-V4-Flash-0731\"]" doc:"The only models requests may be routed to, as provider/model names, in every modality. Left out allows every model, and an empty list allows none. A request that could only go to models not on the list is refused, and a failover never reaches one."`
	Tags                 map[string]string `json:"tags,omitempty" example:"{\"application\":\"support\"}" doc:"Labels recorded on every row of usage, over whatever the request labelled it with, so spend is attributed whatever a caller sends. Together with the request's own they must fit in 16 tags."`
	RequireOwnStreamApp  *bool             `json:"require_own_stream_app,omitempty" doc:"Keep the app out of the router's own Stream app: in app mode it is never written there for want of a registered Stream app of its own, and what it wrote there before can only be read. True at either scope requires it, so an app cannot turn its organization's off. An organization's is set by the router's operator and read here; sending it back unchanged is fine, and changing it is refused."`
	ResumeHistory        *ResumeHistory    `json:"resume_history,omitempty" doc:"How the scope's sessions are handed back a conversation they carry on. The most specific level wins, field by field: a server-side session request, then its agent config, then the app's policy, then the organization's, then the router's default (text only, 100 turns, 60000 characters)."`
	BalanceWarningMicros *int64            `json:"balance_warning_micros,omitempty" minimum:"0" example:"20000000" doc:"The organization's own warning threshold, in millionths of a dollar: at or below it, every answer to a backend says X-Stream-Balance-Warning: low, and the organization's stream event destinations get one account.balance_low. It wins over the tier's. Left out, the tier's applies. Only an organization's policy has one: an app's that sets it is refused."`
	ToolOutputRetention  *string           `json:"tool_output_retention,omitempty" pattern:"^(|none|[1-9][0-9]*[hd])$" patternDescription:"none, a number of hours such as 12h, or a number of days such as 30d" example:"30d" doc:"How long a session keeps what its tools were given and gave back on the turns it records: a tool's arguments, its result, a search's query and the wording of a tool's error. none records none of it; a duration such as 12h or 30d (at least 1h, at most 3650d) keeps it that long after the call and then takes it out, within minutes. The item stays, with its tool name, its time, its call id and whether it failed, so a session's items read in the same order and tool usage still counts the call. Left out, or empty, it is kept as long as the session. The shortest of the organization's, the app's and the agent config's tools.output_retention applies, so an agent can shorten it and never lengthen it. It applies to tool calls recorded after it is set. Deleting a session deletes all of it at once."`
}

func (*Policy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What an organization or an app decided about data handling, " +
		"prompt injection, which models may be used and how usage is labelled. Every field " +
		"is optional, and a field left out is no opinion rather than off. Spend limits are " +
		"not here: they are /v1/spend/limits."
	return schema
}

// Quota is what one end user may spend in a UTC day: on an organization's policy the default
// for every user of its apps, on an app's the default for its users, and on a user's their own.
type Quota struct {
	// The maximums only catch a typo: a person who asked for a response every ten seconds
	// round the clock would still be under 10,000 a day, so 10,000,000 is a thousand of
	// them, and 10^12 tokens is a thousand times 10,000,000 responses of 100,000 tokens.
	MessagesPerUserPerDay *int64 `json:"messages_per_user_per_day,omitempty" minimum:"0" maximum:"10000000" example:"20000" doc:"How many responses one end user may ask for in a UTC day. 0 is no cap. Left out, the next level's applies: the app's, then the organization's, then the router's default."`
	TokensPerUserPerDay   *int64 `json:"tokens_per_user_per_day,omitempty" minimum:"0" maximum:"1000000000000" example:"50000000" doc:"How many tokens, input and output, one end user's responses may cost in a UTC day. 0 is no cap. Left out, the next level's applies, as for messages."`
}

func (*Quota) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What one end user may spend in a UTC day. Each number is the first set of " +
		"the user's own (/v1/policies/users/{user_id}), the app's, the organization's and the " +
		"router's default. A quota set here counts everything recorded with the user: what their " +
		"devices ask for and what the app's backend asks for in their name. (An app's quota counted " +
		"devices only before; the router's default still does.) A device is counted by its address " +
		"as well. A turn is counted on the day of the user who sent it. A request over the limit is a " +
		"429 rate_limited with Retry-After, whose message names the quota (user <id>, every user of app " +
		"<id>, every user of organization <id>), and a counted request carries X-Quota-Limit, " +
		"X-Quota-Remaining and X-Quota-Reset."
	return schema
}

type policyRequest struct {
	Body Policy
}

type policyResponse struct {
	Body Policy
}

func (s *Server) registerPolicies(api huma.API) {
	errs := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden}
	huma.Register(api, huma.Operation{
		OperationID: "getAppPolicy",
		Method:      http.MethodGet,
		Path:        "/v1/policies/app",
		Summary:     "The calling app's policy",
		Description: "What the app itself decided. Its organization's policy applies as well, " +
			"as a floor the app can tighten and cannot loosen: the stricter data policy wins, " +
			"prompt injection is screened if either turns it on, only a model both allow may be " +
			"routed to, and the organization's tags win over the app's.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The app's policy"},
		},
		Errors: errs,
	}, s.getAppPolicy)
	huma.Register(api, huma.Operation{
		OperationID: "updateAppPolicy",
		Method:      http.MethodPut,
		Path:        "/v1/policies/app",
		Summary:     "Replace the calling app's policy",
		Description: "A field left out is no opinion, so the organization's setting shows " +
			"through.\n\nServer-side only: it needs a server-side token, so it cannot be " +
			"reached from an end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The policy was stored"}},
		Errors:    errs,
	}, s.updateAppPolicy)
	huma.Register(api, huma.Operation{
		OperationID: "getOrganizationPolicy",
		Method:      http.MethodGet,
		Path:        "/v1/policies/organization",
		Summary:     "The calling app's organization's policy",
		Description: "Applies to every app the router has seen the organization name. A " +
			"request that names no organization is a 400.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The organization's policy"},
		},
		Errors: errs,
	}, s.getOrganizationPolicy)
	huma.Register(api, huma.Operation{
		OperationID: "updateOrganizationPolicy",
		Method:      http.MethodPut,
		Path:        "/v1/policies/organization",
		Summary:     "Replace the calling app's organization's policy",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached " +
			"from an end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The policy was stored"}},
		Errors:    errs,
	}, s.updateOrganizationPolicy)
}

// getAppPolicy returns what the calling app decided.
func (s *Server) getAppPolicy(ctx context.Context, _ *struct{}) (*policyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.policies == nil {
		return nil, errNoPolicies
	}
	policy, err := s.policyOf(ctx, store.ScopeApp, customerID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// updateAppPolicy replaces what the calling app decided.
func (s *Server) updateAppPolicy(ctx context.Context, request *policyRequest) (*policyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.policies == nil {
		return nil, errNoPolicies
	}
	if request.Body.Budget != nil {
		return nil, errBudgetMoved
	}
	policy, err := s.savePolicy(ctx, store.ScopeApp, customerID, request.Body, customerID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// getOrganizationPolicy returns what the calling app's organization decided.
func (s *Server) getOrganizationPolicy(ctx context.Context, _ *struct{}) (*policyResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if s.policies == nil {
		return nil, errNoPolicies
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return nil, errNoOrganization
	}
	policy, err := s.policyOf(ctx, store.ScopeOrganization, organizationID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// updateOrganizationPolicy replaces what the calling app's organization decided.
func (s *Server) updateOrganizationPolicy(ctx context.Context, request *policyRequest) (*policyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.policies == nil {
		return nil, errNoPolicies
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return nil, errNoOrganization
	}
	if request.Body.Budget != nil {
		return nil, errBudgetMoved
	}
	policy, err := s.savePolicy(ctx, store.ScopeOrganization, organizationID, request.Body, customerID)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// savePolicy validates and stores a policy, audits it under auditedAs, and returns it as it
// now reads.
func (s *Server) savePolicy(ctx context.Context, scope store.PolicyScope, id string, sent Policy, auditedAs string) (Policy, error) {
	document := store.PolicyDocument{
		Quota:                quotaOf(sent.Quota),
		DataPolicy:           dataPolicyOf(sent.DataPolicy),
		PromptInjection:      sent.PromptInjection,
		AllowedModels:        sent.AllowedModels,
		Tags:                 sent.Tags,
		RequireOwnStreamApp:  sent.RequireOwnStreamApp,
		BalanceWarningMicros: sent.BalanceWarningMicros,
		ToolOutputRetention:  options.Retention(value(sent.ToolOutputRetention)),
	}
	if sent.ResumeHistory != nil {
		resume := resumeHistoryOr(store.ResumeHistory{}, sent.ResumeHistory)
		document.ResumeHistory = &resume
	}
	// Credit is the organization's, so the balance it is warned at is too.
	if scope == store.ScopeApp && sent.BalanceWarningMicros != nil {
		return Policy{}, errors.New("an app has no balance warning of its own: set balance_warning_micros on the organization's policy")
	}
	// Any app's backend can write its organization's policy, so the one setting that keeps
	// every app of an organization out of the shared Stream app is the operator's alone. It
	// is never written over HTTP, and the store keeps it as it was. Sending back what was
	// read is fine, so a policy read and written again round-trips; changing it is refused.
	if scope == store.ScopeOrganization {
		if sent.RequireOwnStreamApp != nil {
			existing, err := s.store.Policy(ctx, scope, id)
			if err != nil {
				return Policy{}, err
			}
			if *sent.RequireOwnStreamApp != value(existing.RequireOwnStreamApp) {
				return Policy{}, errors.New("an organization's require_own_stream_app is set by the router's operator")
			}
		}
		document.RequireOwnStreamApp = nil
	}
	if message, ok := toolOutputRetentionComplaint("tool_output_retention", document.ToolOutputRetention); !ok {
		return Policy{}, stack.Wrap(errors.New(message))
	}
	if err := document.DataPolicy.Validate(); err != nil {
		return Policy{}, err
	}
	if err := routing.Tags(sent.Tags).Validate(); err != nil {
		return Policy{}, err
	}
	for _, model := range value(sent.AllowedModels) {
		if !s.routes(model) {
			return Policy{}, stack.Wrap(fmt.Errorf("allowed_models names %q, which is not a provider/model this deployment routes", model))
		}
	}
	// Read before the write, so the entry can say what the setting was.
	before, err := s.configs.Policy(ctx, scope, id)
	if err != nil {
		return Policy{}, err
	}
	if err := s.policies.Save(ctx, scope, id, document); err != nil {
		return Policy{}, err
	}
	s.auditAs(ctx, auditedAs, auditRecord{
		ResourceType: store.AuditPolicy, ResourceID: id, ResourceName: string(scope),
		Action: store.AuditUpdated, Changes: auditDiff(before, document),
	})
	return s.policyOf(ctx, scope, id)
}

// policyOf reads a scope's policy.
func (s *Server) policyOf(ctx context.Context, scope store.PolicyScope, id string) (Policy, error) {
	document, err := s.configs.Policy(ctx, scope, id)
	if err != nil {
		return Policy{}, err
	}
	return Policy{
		Quota:                quotaFor(document.Quota),
		DataPolicy:           dataPolicyFor(document.DataPolicy),
		PromptInjection:      document.PromptInjection,
		AllowedModels:        document.AllowedModels,
		Tags:                 document.Tags,
		RequireOwnStreamApp:  document.RequireOwnStreamApp,
		ResumeHistory:        renderedResumeHistory(value(document.ResumeHistory)),
		BalanceWarningMicros: document.BalanceWarningMicros,
		ToolOutputRetention:  optional(string(document.ToolOutputRetention)),
	}, nil
}

// The shortest and longest tool output retention a setting may name. Less than an hour is
// within reach of how often redaction runs, so it would promise more than is done; ten
// years only catches a typo, since leaving it out keeps it for as long as the session.
const (
	minToolOutputRetention = time.Hour
	maxToolOutputRetention = 3650 * 24 * time.Hour
)

// toolOutputRetentionComplaint refuses a tool output retention that is not none, empty, or
// a duration between the bounds.
func toolOutputRetentionComplaint(field string, retention options.Retention) (string, bool) {
	if retention == "" || retention == options.RetentionNone {
		return "", true
	}
	window, ok := retention.Window()
	if !ok || window < minToolOutputRetention || window > maxToolOutputRetention {
		return fmt.Sprintf("%s is none, or a duration from 1h to 3650d such as 30d, not %q", field, retention), false
	}
	return "", true
}

// quotaOf is a sent quota as it is stored. One that sets neither number is no opinion.
func quotaOf(sent *Quota) *store.Quota {
	if sent == nil || sent.MessagesPerUserPerDay == nil && sent.TokensPerUserPerDay == nil {
		return nil
	}
	return &store.Quota{
		MessagesPerUserPerDay: sent.MessagesPerUserPerDay,
		TokensPerUserPerDay:   sent.TokensPerUserPerDay,
	}
}

// quotaFor is a stored quota as it reads out.
func quotaFor(stored *store.Quota) *Quota {
	if stored == nil {
		return nil
	}
	return &Quota{
		MessagesPerUserPerDay: stored.MessagesPerUserPerDay,
		TokensPerUserPerDay:   stored.TokensPerUserPerDay,
	}
}

// routes reports whether a "provider/model" is one some modality here routes to. A
// customer's own model is taken on its word, since it may be added after the policy is.
func (s *Server) routes(model string) bool {
	if _, own := routing.CustomName(model); own {
		return true
	}
	for _, router := range s.routers {
		if _, ok := router.Config().Provider(model); ok {
			return true
		}
	}
	return false
}
