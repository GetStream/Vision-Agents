package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"reflect"

	"github.com/danielgtaylor/huma/v2"
)

// errorDocs is where every code is explained, under an anchor of its own name.
const errorDocs = "https://getstream.io/agents/docs/api/errors/#"

// somethingWentWrong is all a client is told of a failure that is not its own: what went
// wrong is in the access log, under the request id.
const somethingWentWrong = "something went wrong"

// ErrorType is the kind of a failure, and decides the status it is answered with.
type ErrorType string

const (
	ErrorTypeInvalidRequest       ErrorType = "invalid_request"
	ErrorTypeAuthentication       ErrorType = "authentication"
	ErrorTypePermission           ErrorType = "permission"
	ErrorTypeNotFound             ErrorType = "not_found"
	ErrorTypeMethodNotAllowed     ErrorType = "method_not_allowed"
	ErrorTypeNotAcceptable        ErrorType = "not_acceptable"
	ErrorTypeConflict             ErrorType = "conflict"
	ErrorTypeGone                 ErrorType = "gone"
	ErrorTypePayloadTooLarge      ErrorType = "payload_too_large"
	ErrorTypeUnsupportedMediaType ErrorType = "unsupported_media_type"
	ErrorTypeRateLimited          ErrorType = "rate_limited"
	ErrorTypeInternal             ErrorType = "internal"
	ErrorTypeUnavailable          ErrorType = "unavailable"
)

// errorTypes are the types in the order the spec lists them, with the status each is
// answered with and the code a failure of that type has when nothing names it better.
var errorTypes = []struct {
	errorType ErrorType
	status    int
	code      string
}{
	{ErrorTypeInvalidRequest, http.StatusBadRequest, "invalid_request"},
	{ErrorTypeAuthentication, http.StatusUnauthorized, "unauthenticated"},
	{ErrorTypePermission, http.StatusForbidden, "forbidden"},
	{ErrorTypeNotFound, http.StatusNotFound, "not_found"},
	{ErrorTypeMethodNotAllowed, http.StatusMethodNotAllowed, "method_not_allowed"},
	{ErrorTypeNotAcceptable, http.StatusNotAcceptable, "not_acceptable"},
	{ErrorTypeConflict, http.StatusConflict, "conflict"},
	{ErrorTypeGone, http.StatusGone, "gone"},
	{ErrorTypePayloadTooLarge, http.StatusRequestEntityTooLarge, "payload_too_large"},
	{ErrorTypeUnsupportedMediaType, http.StatusUnsupportedMediaType, "unsupported_media_type"},
	{ErrorTypeRateLimited, http.StatusTooManyRequests, "rate_limited"},
	{ErrorTypeInternal, http.StatusInternalServerError, "internal_error"},
	{ErrorTypeUnavailable, http.StatusServiceUnavailable, "unavailable"},
}

func (ErrorType) Schema(registry huma.Registry) *huma.Schema {
	values := make([]string, 0, len(errorTypes))
	for _, known := range errorTypes {
		values = append(values, string(known.errorType))
	}
	return namedEnum(registry, "ErrorType", "The kind of failure, which decides the status "+
		"it is answered with: invalid_request 400, authentication 401, permission 403, "+
		"not_found 404, method_not_allowed 405, not_acceptable 406, conflict 409, gone 410, "+
		"payload_too_large 413, unsupported_media_type 415, rate_limited 429, internal 500, "+
		"unavailable 503.", values...)
}

// Codes a failure is told apart by when its type alone does not say enough. Every other
// failure has the code of its type.
const (
	codeValidationFailed       = "validation_failed"
	codeMissingCustomer        = "missing_customer"
	codeMissingOrganization    = "missing_organization"
	codeServerSideOnly         = "server_side_only"
	codeNotConfigured          = "not_configured"
	codeModalityNotRouted      = "modality_not_routed"
	codeAgentConfigNotFound    = "agent_config_not_found"
	codeCallNotFound           = "call_not_found"
	codeCampaignNotFound       = "campaign_not_found"
	codeChannelAccountNotFound = "channel_account_not_found"
	codeCommandNotFound        = "command_not_found"
	codeConnectionNotFound     = "connection_not_found"
	codeKnowledgeDocNotFound   = "knowledge_document_not_found"
	codeKnowledgeURLNotFound   = "knowledge_url_not_found"
	codePluginNotFound         = "plugin_not_found"
	codeRouterConfigNotFound   = "router_config_not_found"
	codeSessionNotFound        = "session_not_found"
	codeSimulationNotFound     = "simulation_not_found"
	codeSimulationRunNotFound  = "simulation_run_not_found"
	codeSkillNotFound          = "skill_not_found"
	codeVoiceNotFound          = "voice_not_found"
)

// APIError is a failure an operation answers with on purpose: its type decides the
// status, and the client reads all of it. Any other error an operation returns is answered
// as an internal one, saying nothing of what went wrong.
type APIError struct {
	Type    ErrorType
	Code    string
	Message string
}

func (e APIError) Error() string { return e.Message }

// Status is the HTTP status the failure is answered with.
func (e APIError) Status() int {
	for _, known := range errorTypes {
		if known.errorType == e.Type {
			return known.status
		}
	}
	return http.StatusInternalServerError
}

// GetStatus is Status, under the name Huma asks for it by.
func (e APIError) GetStatus() int { return e.Status() }

// DocURL is where the failure's code is explained.
func (e APIError) DocURL() string { return errorDocs + e.Code }

// MarshalJSON answers the failure in the error envelope every failure has.
func (e APIError) MarshalJSON() ([]byte, error) {
	encoded := &bytes.Buffer{}
	encoder := json.NewEncoder(encoded)
	// A message quotes what was wrong with the request, which is read rather than put in
	// a page: "expected length >= 1" should not arrive as "\u003e=".
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(ErrorResponse{Error: ErrorDetail{
		Message: e.Message,
		Type:    e.Type,
		Code:    e.Code,
		DocURL:  e.DocURL(),
	}})
	return bytes.TrimSuffix(encoded.Bytes(), []byte("\n")), err
}

// Schema documents the failure as the envelope it is answered in.
func (APIError) Schema(registry huma.Registry) *huma.Schema {
	return registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")
}

// ErrorResponse is the body of every failure the router answers with.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

func (*ErrorResponse) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The body of every failure. A failure that is not the caller's own " +
		"is a 500 of type internal saying only \"" + somethingWentWrong + "\": quote the " +
		"response's X-Request-Id to find out more."
	return schema
}

// ErrorDetail is what went wrong.
type ErrorDetail struct {
	Message string    `json:"message" doc:"What went wrong, for a person to read. Its wording may change; branch on code."`
	Type    ErrorType `json:"type"`
	Code    string    `json:"code" doc:"What went wrong, for a program to branch on. Every type has a code of its own name (invalid_request, unauthenticated, forbidden, not_found, method_not_allowed, not_acceptable, conflict, gone, payload_too_large, unsupported_media_type, rate_limited, internal_error, unavailable) that a failure has when nothing names it better. The others are validation_failed, missing_customer, missing_organization, server_side_only, not_configured (this deployment does not offer the feature), modality_not_routed, and <resource>_not_found for agent_config, call, campaign, channel_account, command, connection, knowledge_document, knowledge_url, plugin, router_config, session, simulation, simulation_run, skill and voice. More may be added, so a client should expect one it does not know."`
	DocURL  string    `json:"doc_url" format:"uri" doc:"Where the code is explained."`
}

// coded is a message answered from more than one place, with the code that tells it apart.
type coded struct {
	code    string
	message string
}

// newAPIError is a failure of errorType, with the code message names or else its type's.
func newAPIError[M string | coded](errorType ErrorType, message M) APIError {
	failure := APIError{Type: errorType}
	switch message := any(message).(type) {
	case coded:
		failure.Code, failure.Message = message.code, message.message
	case string:
		failure.Message = message
		for _, known := range errorTypes {
			if known.errorType == errorType {
				failure.Code = known.code
			}
		}
	}
	return failure
}

func invalidRequest[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeInvalidRequest, message)
}

func unauthenticated[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeAuthentication, message)
}

func forbidden[M string | coded](message M) APIError {
	return newAPIError(ErrorTypePermission, message)
}

func notFound[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeNotFound, message)
}

func conflict[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeConflict, message)
}

func gone[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeGone, message)
}

func payloadTooLarge[M string | coded](message M) APIError {
	return newAPIError(ErrorTypePayloadTooLarge, message)
}

func rateLimited[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeRateLimited, message)
}

func unavailable[M string | coded](message M) APIError {
	return newAPIError(ErrorTypeUnavailable, message)
}

// internalError is the answer to a failure that is not the caller's.
func internalError() APIError {
	return newAPIError(ErrorTypeInternal, somethingWentWrong)
}

// statusError is the failure Huma reports with a status of its own: a body it could not
// read, one that does not validate, a media type it does not serve.
func statusError(status int, message string) APIError {
	if status >= http.StatusInternalServerError {
		return internalError()
	}
	for _, known := range errorTypes {
		if known.status == status {
			return newAPIError(known.errorType, message)
		}
	}
	return invalidRequest(message)
}

// writeError answers a request served by hand with failure, in the envelope every
// operation answers with.
func writeError(w http.ResponseWriter, failure APIError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(failure.Status())
	_ = json.NewEncoder(w).Encode(failure)
}

// writeOperationError answers an operation with failure from a middleware that stops the
// request before the operation runs.
func writeOperationError(ctx huma.Context, failure APIError) {
	ctx.SetHeader("Content-Type", "application/json")
	ctx.SetStatus(failure.Status())
	_ = json.NewEncoder(ctx.BodyWriter()).Encode(failure)
}
