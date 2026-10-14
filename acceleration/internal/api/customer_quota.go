package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type customerQuotaRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
}

// SetCustomerQuotaRequest is Stream staff's new daily limit for an app, and who set it.
type SetCustomerQuotaRequest struct {
	MessagesPerUserPerDay *int64 `json:"messages_per_user_per_day,omitempty" minimum:"0" maximum:"10000000" example:"20000" doc:"How many responses one end user may ask for in a UTC day. 0 is no cap. Left out, the router's default applies."`
	TokensPerUserPerDay   *int64 `json:"tokens_per_user_per_day,omitempty" minimum:"0" maximum:"1000000000000" example:"50000000" doc:"How many tokens, input and output, one end user's responses may cost in a UTC day. 0 is no cap. Left out, the router's default applies."`
	ChangedBy             string `json:"changed_by" minLength:"1" maxLength:"255" doc:"Who is changing it. Kept with the change."`
	Reason                string `json:"reason,omitempty" maxLength:"4096" doc:"Why, for Stream staff. Kept with the change. The app is never shown it."`
}

type setCustomerQuotaRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
	Body       *SetCustomerQuotaRequest
}

// QuotaChange is one change Stream staff made to an app's daily limit per end user.
type QuotaChange struct {
	ID        string    `json:"id"`
	ChangedBy string    `json:"changed_by" doc:"Who made the change."`
	Reason    string    `json:"reason" doc:"Why, as staff gave it. Empty when they gave none."`
	ChangedAt time.Time `json:"changed_at" doc:"When it was made."`
	Previous  Quota     `json:"previous" doc:"The limit before the change. A number left out was the router's default."`
	New       Quota     `json:"new" doc:"The limit after the change. A number left out is the router's default."`
}

// QuotaChangePage is a page of changes.
type QuotaChangePage struct {
	Items      []QuotaChange `json:"items" nullable:"false"`
	HasMore    bool          `json:"has_more"`
	NextCursor *string       `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last one."`
}

type listCustomerQuotaChangesRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
	// 200 and 25 are store.QuotaChangeLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listCustomerQuotaChangesResponse struct {
	Body QuotaChangePage
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
// rest of the app's policy as it is. It records who changed it and what it was before.
func (s *Server) setCustomerQuota(ctx context.Context, request *setCustomerQuotaRequest) (*customerQuotaResponse, error) {
	if s.policies == nil {
		return nil, errNoPolicies
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	changedBy := strings.TrimSpace(request.Body.ChangedBy)
	if changedBy == "" {
		return nil, invalidRequest("say who is changing the limit")
	}
	quota := quotaOf(&Quota{MessagesPerUserPerDay: request.Body.MessagesPerUserPerDay, TokensPerUserPerDay: request.Body.TokensPerUserPerDay})
	if _, err := s.policies.ChangeAppQuota(ctx, request.CustomerID, changedBy,
		strings.TrimSpace(request.Body.Reason), quota); err != nil {
		return nil, err
	}
	s.logger.Info("set an app's daily limit per end user", "customer", request.CustomerID,
		"changed_by", changedBy, "messages", request.Body.MessagesPerUserPerDay,
		"tokens", request.Body.TokensPerUserPerDay)
	return &customerQuotaResponse{Body: value(quotaFor(quota))}, nil
}

// listCustomerQuotaChanges returns who changed an app's daily limit per end user, newest
// first.
func (s *Server) listCustomerQuotaChanges(ctx context.Context, request *listCustomerQuotaChangesRequest) (*listCustomerQuotaChangesResponse, error) {
	if s.store == nil {
		return nil, errNoPolicies
	}
	cursor, err := decodeCursor[store.QuotaChangePosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.AppQuotaChanges(ctx, request.CustomerID, request.Limit, cursor)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.QuotaChangeLimit(request.Limit))
	listed := QuotaChangePage{Items: make([]QuotaChange, 0, len(kept)), HasMore: more}
	for _, row := range kept {
		listed.Items = append(listed.Items, QuotaChange{
			ID: row.ID, ChangedBy: row.ChangedBy, Reason: row.Reason, ChangedAt: row.ChangedAt,
			Previous: Quota{MessagesPerUserPerDay: row.PreviousMessagesPerUserPerDay, TokensPerUserPerDay: row.PreviousTokensPerUserPerDay},
			New:      Quota{MessagesPerUserPerDay: row.NewMessagesPerUserPerDay, TokensPerUserPerDay: row.NewTokensPerUserPerDay},
		})
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.QuotaChangePosition{ChangedAt: last.ChangedAt, ID: last.ID})
	}
	return &listCustomerQuotaChangesResponse{Body: listed}, nil
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
			"that stores it applies it at once, and the others within ten seconds. Every " +
			"change is recorded with who made it, when, and the numbers before and after.\n\n" +
			"Stream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The app's daily limit, as stored"}},
		Errors:    staffErrors,
	}, s.setCustomerQuota)
	huma.Register(api, huma.Operation{
		OperationID: "listCustomerQuotaChanges",
		Method:      http.MethodGet,
		Path:        path + "/changes",
		Summary:     "List the changes made to an app's daily limit per end user",
		Description: "Every change Stream staff made, newest first: who made it, why, when, and " +
			"the limit before and after.\n\nStream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "A page of changes"}},
		Errors:    staffErrors,
	}, s.listCustomerQuotaChanges)
}
