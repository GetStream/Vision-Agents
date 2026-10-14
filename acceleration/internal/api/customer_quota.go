package api

import (
	"context"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type customerQuotaRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
}

type setCustomerQuotaRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
	Body       Quota
}

type customerQuotaResponse struct {
	Body Quota
}

// getCustomerQuota returns what one end user of an app may spend in a day.
func (s *Server) getCustomerQuota(ctx context.Context, request *customerQuotaRequest) (*customerQuotaResponse, error) {
	if s.policies == nil {
		return nil, errNoPolicies
	}
	document, err := s.store.Policy(ctx, store.ScopeApp, request.CustomerID)
	if err != nil {
		return nil, err
	}
	return &customerQuotaResponse{Body: value(quotaFor(document.Quota))}, nil
}

// setCustomerQuota replaces what one end user of an app may spend in a day, and leaves the
// rest of the app's policy as it is.
func (s *Server) setCustomerQuota(ctx context.Context, request *setCustomerQuotaRequest) (*customerQuotaResponse, error) {
	if s.policies == nil {
		return nil, errNoPolicies
	}
	document, err := s.store.Policy(ctx, store.ScopeApp, request.CustomerID)
	if err != nil {
		return nil, err
	}
	document.Quota = quotaOf(&request.Body)
	if err := s.policies.Save(ctx, store.ScopeApp, request.CustomerID, document); err != nil {
		return nil, err
	}
	s.logger.Info("set an app's daily limit per end user", "customer", request.CustomerID,
		"messages", request.Body.MessagesPerUserPerDay, "tokens", request.Body.TokensPerUserPerDay)
	return &customerQuotaResponse{Body: value(quotaFor(document.Quota))}, nil
}

// registerCustomerQuota declares Stream staff's setting of an app's daily limit per end
// user, which takes the ops key.
func (s *Server) registerCustomerQuota(api huma.API) {
	staffErrors := []int{http.StatusBadRequest, http.StatusUnauthorized}
	path := "/v1/ops/customers/{customer_id}/quota"
	huma.Register(api, huma.Operation{
		OperationID: "getCustomerQuota",
		Method:      http.MethodGet,
		Path:        path,
		Summary:     "Get an app's daily limit per end user",
		Description: "What one end user of the app may spend in a UTC day. A number left out " +
			"is the router's default.\n\nStream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The app's daily limit"}},
		Errors:    staffErrors,
	}, s.getCustomerQuota)
	huma.Register(api, huma.Operation{
		OperationID: "setCustomerQuota",
		Method:      http.MethodPut,
		Path:        path,
		Summary:     "Set an app's daily limit per end user",
		Description: "Replaces what one end user of the app may spend in a UTC day, and leaves " +
			"the rest of the app's policy as it is. A number left out is the router's default, " +
			"and 0 is no cap. The app reads it in its policy and cannot change it. The node " +
			"that stores it applies it at once, and the others within ten seconds.\n\n" +
			"Stream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The app's daily limit, as stored"}},
		Errors:    staffErrors,
	}, s.setCustomerQuota)
}
