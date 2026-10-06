package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	noPolicies     = "policies are not available: no database configured"
	noOrganization = "this request names no organization: it needs " +
		"X-Stream-Organization-Id, or an API key belonging to one"
)

// Policy is what an organization or an app decided about spend, data handling, prompt
// injection, which models may be used and how usage is labelled.
type Policy struct {
	Budget          *Budget           `json:"budget,omitempty"`
	DataPolicy      *DataPolicy       `json:"data_policy,omitempty"`
	PromptInjection *bool             `json:"prompt_injection,omitempty" doc:"Screen what every LLM response is asked for prompt injection. The newest input - the user's turn and any tool results - goes to the classifier (lcm) beside the model call, so it adds nothing to time to first token. The end of the response is held until the verdict, and a response whose input reads as an injection fails with prompt_injection before its tool calls can be acted on."`
	AllowedModels   *[]string         `json:"allowed_models,omitempty" example:"[\"deepseek/DeepSeek-V4-Flash-0731\"]" doc:"The only models requests may be routed to, as provider/model names, in every modality. Left out allows every model, and an empty list allows none. A request that could only go to models not on the list is refused, and a failover never reaches one."`
	Tags            map[string]string `json:"tags,omitempty" example:"{\"application\":\"support\"}" doc:"Labels recorded on every row of usage, over whatever the request labelled it with, so spend is attributed whatever a caller sends. Together with the request's own they must fit in 16 tags."`
}

func (*Policy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What an organization or an app decided about spend, data handling, " +
		"prompt injection, which models may be used and how usage is labelled. Every field " +
		"is optional, and a field left out is no opinion rather than off."
	return schema
}

// Budget caps spend across every modality, reset on a UTC boundary each interval.
type Budget struct {
	LimitMicros int64          `json:"limit_micros" minimum:"1" example:"100000000" doc:"The cap, in millionths of a dollar."`
	Interval    BudgetInterval `json:"interval"`
	SpentMicros *int64         `json:"spent_micros,omitempty" readOnly:"true" doc:"What has been spent in the current interval."`
	ResetsAt    *time.Time     `json:"resets_at,omitempty" readOnly:"true" doc:"When the current interval ends."`
}

func (*Budget) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A cap on spend across every modality, reset on a UTC boundary each " +
		"interval. Once it is spent every new session and every LLM response is refused until " +
		"the next interval. Checks are cached for a few seconds, so a busy app can overshoot " +
		"by what it spends in that time."
	return schema
}

// BudgetInterval is how often a budget resets.
type BudgetInterval string

const (
	BudgetIntervalHourly  BudgetInterval = "hourly"
	BudgetIntervalDaily   BudgetInterval = "daily"
	BudgetIntervalWeekly  BudgetInterval = "weekly"
	BudgetIntervalMonthly BudgetInterval = "monthly"
)

func (BudgetInterval) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "BudgetInterval", "How often a budget resets. A week starts on Monday.",
		string(BudgetIntervalHourly), string(BudgetIntervalDaily),
		string(BudgetIntervalWeekly), string(BudgetIntervalMonthly))
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
			"as a floor the app can tighten and cannot loosen: both budgets are enforced, the " +
			"stricter data policy wins, prompt injection is screened if either turns it on, " +
			"only a model both allow may be routed to, and the organization's tags win over " +
			"the app's.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The app's policy, with what its budget has spent so far"},
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
			"200": {Description: "The organization's policy, with what its budget has spent so far"},
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
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.policies == nil {
		return nil, huma.Error400BadRequest(noPolicies)
	}
	policy, err := s.policyOf(ctx, store.ScopeApp, customerID)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// updateAppPolicy replaces what the calling app decided.
func (s *Server) updateAppPolicy(ctx context.Context, request *policyRequest) (*policyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.policies == nil {
		return nil, huma.Error400BadRequest(noPolicies)
	}
	policy, err := s.savePolicy(ctx, store.ScopeApp, customerID, request.Body)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// getOrganizationPolicy returns what the calling app's organization decided.
func (s *Server) getOrganizationPolicy(ctx context.Context, _ *struct{}) (*policyResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.policies == nil {
		return nil, huma.Error400BadRequest(noPolicies)
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return nil, huma.Error400BadRequest(noOrganization)
	}
	policy, err := s.policyOf(ctx, store.ScopeOrganization, organizationID)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// updateOrganizationPolicy replaces what the calling app's organization decided.
func (s *Server) updateOrganizationPolicy(ctx context.Context, request *policyRequest) (*policyResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.policies == nil {
		return nil, huma.Error400BadRequest(noPolicies)
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return nil, huma.Error400BadRequest(noOrganization)
	}
	policy, err := s.savePolicy(ctx, store.ScopeOrganization, organizationID, request.Body)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &policyResponse{Body: policy}, nil
}

// savePolicy validates and stores a policy, and returns it as it now reads.
func (s *Server) savePolicy(ctx context.Context, scope store.PolicyScope, id string, sent Policy) (Policy, error) {
	document := store.PolicyDocument{
		DataPolicy:      dataPolicyOf(sent.DataPolicy),
		PromptInjection: sent.PromptInjection,
		AllowedModels:   sent.AllowedModels,
		Tags:            sent.Tags,
	}
	if !document.DataPolicy.Valid() {
		return Policy{}, stack.Wrap(fmt.Errorf("retention is none or a duration such as 30d, not %q",
			document.DataPolicy.Retention))
	}
	if sent.Budget != nil {
		budget := store.Budget{LimitMicros: sent.Budget.LimitMicros, Interval: store.BudgetInterval(sent.Budget.Interval)}
		if budget.LimitMicros < 1 {
			return Policy{}, stack.Wrap(fmt.Errorf("a budget's limit_micros must be at least 1, not %d", budget.LimitMicros))
		}
		if !budget.Interval.Valid() {
			return Policy{}, stack.Wrap(fmt.Errorf("a budget resets hourly, daily, weekly or monthly, not %q", budget.Interval))
		}
		document.Budget = &budget
	}
	if err := routing.Tags(sent.Tags).Validate(); err != nil {
		return Policy{}, err
	}
	for _, model := range value(sent.AllowedModels) {
		if !s.routes(model) {
			return Policy{}, stack.Wrap(fmt.Errorf("allowed_models names %q, which is not a provider/model this deployment routes", model))
		}
	}
	if err := s.policies.Save(ctx, scope, id, document); err != nil {
		return Policy{}, err
	}
	return s.policyOf(ctx, scope, id)
}

// policyOf reads a scope's policy, with what its budget has spent in the current interval.
func (s *Server) policyOf(ctx context.Context, scope store.PolicyScope, id string) (Policy, error) {
	document, err := s.configs.Policy(ctx, scope, id)
	if err != nil {
		return Policy{}, err
	}
	policy := Policy{
		DataPolicy:      dataPolicyFor(document.DataPolicy),
		PromptInjection: document.PromptInjection,
		AllowedModels:   document.AllowedModels,
		Tags:            document.Tags,
	}
	if document.Budget != nil {
		spent, resets, err := s.policies.Spent(ctx, scope, id, *document.Budget)
		if err != nil {
			return Policy{}, err
		}
		policy.Budget = &Budget{
			LimitMicros: document.Budget.LimitMicros,
			Interval:    BudgetInterval(document.Budget.Interval),
			SpentMicros: &spent,
			ResetsAt:    &resets,
		}
	}
	return policy, nil
}

// routes reports whether a "provider/model" is one some modality here routes to.
func (s *Server) routes(model string) bool {
	for _, router := range s.routers {
		if _, ok := router.Config().Provider(model); ok {
			return true
		}
	}
	return false
}
