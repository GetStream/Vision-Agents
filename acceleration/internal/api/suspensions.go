package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var (
	errNoSuspensions = notConfigured("suspensions are not available: this deployment has no database")
	errNotSuspended  = APIError{Type: ErrorTypeNotFound, Code: codeTelephonyNotSuspended,
		Message: "this app's telephony is not suspended"}
)

// TelephonySuspension is an app Stream staff stopped from using the phone.
type TelephonySuspension struct {
	CustomerId  string    `json:"customer_id" doc:"The app that is suspended."`
	Reason      string    `json:"reason" doc:"Why, for Stream staff. The app is never shown it."`
	SuspendedBy string    `json:"suspended_by" doc:"Who suspended it."`
	SuspendedAt time.Time `json:"suspended_at" doc:"When it was suspended, or last changed."`
}

// SuspendTelephonyRequest suspends an app's telephony.
type SuspendTelephonyRequest struct {
	Reason      string `json:"reason" minLength:"1" maxLength:"4096" doc:"Why the app is suspended. The app is never shown it."`
	SuspendedBy string `json:"suspended_by" minLength:"1" maxLength:"255" doc:"Who is suspending it."`
}

type telephonySuspensionRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
}

type suspendTelephonyRequest struct {
	CustomerID string `path:"customer_id" doc:"The app."`
	Body       *SuspendTelephonyRequest
}

type telephonySuspensionResponse struct {
	Body TelephonySuspension
}

// getTelephonySuspension says why an app's telephony is suspended.
func (s *Server) getTelephonySuspension(ctx context.Context, request *telephonySuspensionRequest) (*telephonySuspensionResponse, error) {
	if s.store == nil {
		return nil, errNoSuspensions
	}
	suspension, err := s.store.TelephonySuspension(ctx, request.CustomerID)
	if errors.Is(err, store.ErrNotSuspended) {
		return nil, errNotSuspended
	}
	if err != nil {
		return nil, err
	}
	return &telephonySuspensionResponse{Body: telephonySuspension(suspension)}, nil
}

// suspendTelephony stops an app using the phone.
func (s *Server) suspendTelephony(ctx context.Context, request *suspendTelephonyRequest) (*telephonySuspensionResponse, error) {
	if s.store == nil {
		return nil, errNoSuspensions
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	suspension := store.TelephonySuspension{
		CustomerID:  request.CustomerID,
		Reason:      strings.TrimSpace(request.Body.Reason),
		SuspendedBy: strings.TrimSpace(request.Body.SuspendedBy),
	}
	if suspension.Reason == "" || suspension.SuspendedBy == "" {
		return nil, invalidRequest("say why the app is suspended and who is suspending it")
	}
	if err := s.store.SuspendTelephony(ctx, &suspension); err != nil {
		return nil, err
	}
	s.logger.Info("suspended an app's telephony",
		"customer", suspension.CustomerID, "suspended_by", suspension.SuspendedBy)
	return &telephonySuspensionResponse{Body: telephonySuspension(suspension)}, nil
}

// liftTelephonySuspension lets an app use the phone again.
func (s *Server) liftTelephonySuspension(ctx context.Context, request *telephonySuspensionRequest) (*struct{}, error) {
	if s.store == nil {
		return nil, errNoSuspensions
	}
	err := s.store.LiftTelephonySuspension(ctx, request.CustomerID)
	if errors.Is(err, store.ErrNotSuspended) {
		return nil, errNotSuspended
	}
	if err != nil {
		return nil, err
	}
	s.logger.Info("lifted an app's telephony suspension", "customer", request.CustomerID)
	return nil, nil
}

func telephonySuspension(suspension store.TelephonySuspension) TelephonySuspension {
	return TelephonySuspension{
		CustomerId:  suspension.CustomerID,
		Reason:      suspension.Reason,
		SuspendedBy: suspension.SuspendedBy,
		SuspendedAt: suspension.SuspendedAt,
	}
}

// registerSuspensions declares Stream staff's suspension of an app's telephony, which takes
// the ops key.
func (s *Server) registerSuspensions(api huma.API) {
	staffErrors := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound}
	path := "/v1/ops/customers/{customer_id}/telephony-suspension"
	huma.Register(api, huma.Operation{
		OperationID: "getTelephonySuspension",
		Method:      http.MethodGet,
		Path:        path,
		Summary:     "Get why an app's telephony is suspended",
		Description: "The suspension an app is under, or 404 when it is not suspended.\n\nStream staff only: it needs the ops key.",
		Security:    opsSecurity,
		Responses:   map[string]*huma.Response{"200": {Description: "The suspension"}},
		Errors:      staffErrors,
	}, s.getTelephonySuspension)
	huma.Register(api, huma.Operation{
		OperationID: "suspendTelephony",
		Method:      http.MethodPut,
		Path:        path,
		Summary:     "Suspend an app's telephony",
		Description: "Stops an app using the phone, for abuse or fraud. While it is suspended the app " +
			"buys no numbers, places no calls or transfers, answers no incoming call and starts no " +
			"campaign, and its running campaigns pause at their next call. Calls already in " +
			"progress are left alone. Suspending an app already suspended replaces the reason.\n\n" +
			"Stream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The suspension"}},
		Errors:    staffErrors,
	}, s.suspendTelephony)
	huma.Register(api, huma.Operation{
		OperationID:   "liftTelephonySuspension",
		Method:        http.MethodDelete,
		Path:          path,
		Summary:       "Lift an app's telephony suspension",
		DefaultStatus: http.StatusNoContent,
		Description: "Lets the app use the phone again. Its paused campaigns stay paused until the " +
			"app starts them.\n\nStream staff only: it needs the ops key.",
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"204": {Description: "The suspension is lifted"}},
		Errors:    staffErrors,
	}, s.liftTelephonySuspension)
}
