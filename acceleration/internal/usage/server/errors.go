package server

import (
	"encoding/json"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

// Codes of the management API's failures, for a program to branch on.
const (
	codeInvalidRequest  = "invalid_request"
	codeUnauthenticated = "unauthenticated"
	codeNotConfigured   = "not_configured"
	codeDefaultTier     = "default_tier"
	codeKeyReused       = "idempotency_key_reused"
)

// apiError is a problem answer with a code beside Huma's fields. The router's API package sets
// Huma's error constructor process-wide, so the service builds its own coded errors rather
// than changing it.
type apiError struct {
	huma.ErrorModel
	// Code says what went wrong, for a program to branch on.
	Code string `json:"code"`
}

// problem is the coded error of status.
func problem(status int, code, detail string) *apiError {
	return &apiError{ErrorModel: huma.ErrorModel{Title: http.StatusText(status), Status: status, Detail: detail}, Code: code}
}

// invalidRequest is a 400 saying what is wrong with the request.
func invalidRequest(detail string) *apiError {
	return problem(http.StatusBadRequest, codeInvalidRequest, detail)
}

// writeProblem answers e from a middleware, before the operation runs.
func writeProblem(ctx huma.Context, e *apiError) {
	ctx.SetHeader("Content-Type", "application/problem+json")
	ctx.SetStatus(e.Status)
	_ = json.NewEncoder(ctx.BodyWriter()).Encode(e)
}
