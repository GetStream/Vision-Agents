package api

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	noPolicies     = "policies are not available: no database configured"
	noOrganization = "this request names no organization: it needs " +
		"X-Stream-Organization-Id, or an API key belonging to one"
)

// GetAppPolicy returns what the calling app decided.
func (s *Server) GetAppPolicy(ctx context.Context, _ GetAppPolicyRequestObject) (GetAppPolicyResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetAppPolicy401JSONResponse{missingCustomer()}, nil
	}
	if s.policies == nil {
		return GetAppPolicy400JSONResponse{badRequest(noPolicies)}, nil
	}
	policy, err := s.policyOf(ctx, store.ScopeApp, customerID)
	if err != nil {
		return GetAppPolicy400JSONResponse{badRequest(err.Error())}, nil
	}
	return GetAppPolicy200JSONResponse(policy), nil
}

// UpdateAppPolicy replaces what the calling app decided.
func (s *Server) UpdateAppPolicy(ctx context.Context, request UpdateAppPolicyRequestObject) (UpdateAppPolicyResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateAppPolicy401JSONResponse{missingCustomer()}, nil
	}
	if s.policies == nil {
		return UpdateAppPolicy400JSONResponse{badRequest(noPolicies)}, nil
	}
	if request.Body == nil {
		return UpdateAppPolicy400JSONResponse{badRequest("a request body is required")}, nil
	}
	policy, err := s.savePolicy(ctx, store.ScopeApp, customerID, *request.Body)
	if err != nil {
		return UpdateAppPolicy400JSONResponse{badRequest(err.Error())}, nil
	}
	return UpdateAppPolicy200JSONResponse(policy), nil
}

// GetOrganizationPolicy returns what the calling app's organization decided.
func (s *Server) GetOrganizationPolicy(ctx context.Context, _ GetOrganizationPolicyRequestObject) (GetOrganizationPolicyResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return GetOrganizationPolicy401JSONResponse{missingCustomer()}, nil
	}
	if s.policies == nil {
		return GetOrganizationPolicy400JSONResponse{badRequest(noPolicies)}, nil
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return GetOrganizationPolicy400JSONResponse{badRequest(noOrganization)}, nil
	}
	policy, err := s.policyOf(ctx, store.ScopeOrganization, organizationID)
	if err != nil {
		return GetOrganizationPolicy400JSONResponse{badRequest(err.Error())}, nil
	}
	return GetOrganizationPolicy200JSONResponse(policy), nil
}

// UpdateOrganizationPolicy replaces what the calling app's organization decided.
func (s *Server) UpdateOrganizationPolicy(ctx context.Context, request UpdateOrganizationPolicyRequestObject) (UpdateOrganizationPolicyResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return UpdateOrganizationPolicy401JSONResponse{missingCustomer()}, nil
	}
	if s.policies == nil {
		return UpdateOrganizationPolicy400JSONResponse{badRequest(noPolicies)}, nil
	}
	organizationID := OrganizationFrom(ctx)
	if organizationID == "" {
		return UpdateOrganizationPolicy400JSONResponse{badRequest(noOrganization)}, nil
	}
	if request.Body == nil {
		return UpdateOrganizationPolicy400JSONResponse{badRequest("a request body is required")}, nil
	}
	policy, err := s.savePolicy(ctx, store.ScopeOrganization, organizationID, *request.Body)
	if err != nil {
		return UpdateOrganizationPolicy400JSONResponse{badRequest(err.Error())}, nil
	}
	return UpdateOrganizationPolicy200JSONResponse(policy), nil
}

// savePolicy validates and stores a policy, and returns it as it now reads.
func (s *Server) savePolicy(ctx context.Context, scope store.PolicyScope, id string, sent Policy) (Policy, error) {
	document := store.PolicyDocument{
		DataPolicy:      dataPolicyOf(sent.DataPolicy),
		PromptInjection: sent.PromptInjection,
	}
	if !document.DataPolicy.Valid() {
		return Policy{}, fmt.Errorf("retention is none or a duration such as 30d, not %q",
			document.DataPolicy.Retention)
	}
	if sent.Budget != nil {
		budget := store.Budget{LimitMicros: sent.Budget.LimitMicros, Interval: store.BudgetInterval(sent.Budget.Interval)}
		if budget.LimitMicros < 1 {
			return Policy{}, fmt.Errorf("a budget's limit_micros must be at least 1, not %d", budget.LimitMicros)
		}
		if !budget.Interval.Valid() {
			return Policy{}, fmt.Errorf("a budget resets hourly, daily, weekly or monthly, not %q", budget.Interval)
		}
		document.Budget = &budget
	}
	if err := s.policies.Save(ctx, scope, id, document); err != nil {
		return Policy{}, err
	}
	return s.policyOf(ctx, scope, id)
}

// policyOf reads a scope's policy, with what its budget has spent in the current interval.
func (s *Server) policyOf(ctx context.Context, scope store.PolicyScope, id string) (Policy, error) {
	document, err := s.store.Policy(ctx, scope, id)
	if err != nil {
		return Policy{}, err
	}
	policy := Policy{
		DataPolicy:      dataPolicyFor(document.DataPolicy),
		PromptInjection: document.PromptInjection,
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
