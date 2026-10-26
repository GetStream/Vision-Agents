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
	ErrorTypePaymentRequired      ErrorType = "payment_required"
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
	ErrorTypeTimeout              ErrorType = "timeout"
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
	{ErrorTypePaymentRequired, http.StatusPaymentRequired, "payment_required"},
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
	{ErrorTypeTimeout, http.StatusGatewayTimeout, "timeout"},
}

func (ErrorType) Schema(registry huma.Registry) *huma.Schema {
	values := make([]string, 0, len(errorTypes))
	for _, known := range errorTypes {
		values = append(values, string(known.errorType))
	}
	return namedEnum(registry, "ErrorType", "The kind of failure, which decides the status "+
		"it is answered with: invalid_request 400, authentication 401, payment_required 402, "+
		"permission 403, not_found 404, method_not_allowed 405, not_acceptable 406, conflict 409, gone 410, "+
		"payload_too_large 413, unsupported_media_type 415, rate_limited 429, internal 500, "+
		"unavailable 503, timeout 504.", values...)
}

// Codes a failure is told apart by when its type alone does not say enough. Every other
// failure has the code of its type.
const (
	codeValidationFailed        = "validation_failed"
	codeMissingCustomer         = "missing_customer"
	codeMissingOrganization     = "missing_organization"
	codeServerSideOnly          = "server_side_only"
	codeNotConfigured           = "not_configured"
	codeModalityNotRouted       = "modality_not_routed"
	codeAgentConfigNotFound     = "agent_config_not_found"
	codeCallNotFound            = "call_not_found"
	codeCampaignNotFound        = "campaign_not_found"
	codeChannelAccountNotFound  = "channel_account_not_found"
	codeCommandNotFound         = "command_not_found"
	codeConnectionNotFound      = "connection_not_found"
	codeKnowledgeDocNotFound    = "knowledge_document_not_found"
	codeKnowledgeURLNotFound    = "knowledge_url_not_found"
	codePluginNotFound          = "plugin_not_found"
	codeRouterConfigNotFound    = "router_config_not_found"
	codeSessionNotFound         = "session_not_found"
	codeSimulationNotFound      = "simulation_not_found"
	codeSimulationRunNotFound   = "simulation_run_not_found"
	codeSkillNotFound           = "skill_not_found"
	codeVoiceNotFound           = "voice_not_found"
	codeModelNotFound           = "model_not_found"
	codeCardedSessionToNative   = "carded_session_to_native"
	codeUnsyncedChanges         = "unsynced_changes"
	codePhoneNumberNeedsAddress = "phone_number_needs_address"
	codePhoneVendorFailed       = "phone_vendor_failed"
	codeTelephonySuspended      = "telephony_suspended"
	codeTelephonyNotSuspended   = "telephony_not_suspended"
	codeNameTaken               = "name_taken"
	codeScheduledTaskNotFound   = "scheduled_task_not_found"
	codeInvalidTimezone         = "invalid_timezone"
	codeTooManyTasks            = "too_many_tasks"
	codeRoleTaken               = "role_taken"
	codeChannelConnectionTaken  = "channel_connection_taken"
	codeNoOAuthClient           = "no_oauth_client"
	codeInvalidConfig           = "invalid_config"
	codeConcurrencyLimited      = "concurrency_limited"
	codeModelFailed             = "model_failed"
	codeImageUnavailable        = "image_unavailable"
	codeInvalidUsageCategories  = "invalid_usage_categories"
	codeMemoryUnavailable       = "memory_unavailable"
	// The price list's management operations (price_lists.go).
	codeInvalidPrice         = "invalid_price"
	codeConfirmationRequired = "confirmation_required"
	codeIdempotencyKeyReused = "idempotency_key_reused"
	codePriceChangeInForce   = "price_change_in_force"
	codePriceChangeNotFound  = "price_change_not_found"
	codePriceChangeExists    = "price_change_exists"
	codePriceListNotSeeded   = "price_list_not_seeded"
	codePriceListBusy        = "price_list_busy"
	// Admission's refusals about money (money.go).
	codeAccountSuspended    = "account_suspended"
	codeInsufficientCredits = "insufficient_credits"
	codeSystemLimitReached  = "system_limit_reached"
	codeSpendLimitReached   = "spend_limit_reached"
	codeUserLimitReached    = "user_limit_reached"
	// codeSpendCheckUnavailable is billable work refused while this router cannot share what it
	// holds with the others (money.go).
	codeSpendCheckUnavailable = "spend_check_unavailable"
	// A policy that still sends a budget, which spend limits replaced (policies.go).
	codeBudgetMoved = "budget_moved"
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
	Code    string    `json:"code" doc:"What went wrong, for a program to branch on. Every type has a code of its own name (invalid_request, unauthenticated, forbidden, payment_required, not_found, method_not_allowed, not_acceptable, conflict, gone, payload_too_large, unsupported_media_type, rate_limited, internal_error, unavailable, timeout) that a failure has when nothing names it better. The others are validation_failed, missing_customer, missing_organization, server_side_only, not_configured (this deployment does not offer the feature), modality_not_routed, unsynced_changes (a sync asked to check would write over somebody's edits), name_taken (a 409: another agent config, router config or voice, or another skill of the same agent config, already has the name, so another name will do), channel_connection_taken (a 409: another live agent config binds the channel connection as fixed, and one agent answers a channel connection's messages, so the message names that config), no_oauth_client (a 400: a consent found no OAuth client for the connector, and the message names each way the connector lets the app give it one), invalid_config (a 400: an agent config a run names does not start a session), insufficient_credits (a 402: the organization's credit is spent, so work that costs money is refused until credit is added; a model that costs nothing is still served), system_limit_reached (a 402: the organization's spend hard limit for the day or the month is reached, whatever its credit, and the message names the interval, the amount and when it resets), spend_limit_reached (a 402: one of the organization's own spend limits is spent, on the organization, an app, a key, a usage category or a team, and the message says which and when it resets, so retrying before then is refused again), user_limit_reached (a 429: a spend limit on the one end user the request is for is spent, which stops that user and no one else, and Retry-After says when its window ends), spend_check_unavailable (a 503: the router could not share what the organization is spending with the other routers for a while, so work that costs money is refused until it can; a model that costs nothing is still served, and Retry-After says to try again in a few seconds), budget_moved (a 400: a policy sent a budget, which spend limits replaced), limit_above_parent (a 400: a spend limit is above a limit of its parent in the same interval, and the message names the parent's limit), limit_exists (a 409: the subject already has a limit at that level and interval, and the message names its id), no_limit (a 404: the subject has no spend limit to reset or grant past), use_team (a 400: a category starting team: was sent, where the team field names a team), invalid_expiry (a 400: a grant's expires_at is not in the future, or is more than 366 days away), too_many_grants (a 409: the organization has 20 live grants, as many as it may), account_suspended (a 403: the organization is suspended, and every request is refused, one for a model that costs nothing included, until it is restored; the message says to contact support), concurrency_limited (a 429: the app has as many runs in flight as it may, and Retry-After says when to try again), model_failed (a 503: the model failed or gave no answer), image_unavailable (a 400: an image a run was handed could not be fetched or is not a JPEG, PNG, WebP or GIF, which retrying the same image does not fix), invalid_usage_categories (a 400: the usage categories the request names, in the token's usage_categories and team claims or in X-Stream-Usage-Categories, are more than 10, one is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : -, more than one names a team, or the team is over 59 characters), memory_unavailable (a 503: the memory store could not delete what was asked, so nothing that waits on it was deleted either, and the same request later can succeed), and <resource>_not_found for agent_config, call, campaign, channel_account, command, connection, grant, knowledge_document, knowledge_url, limit, plugin, router_config, scheduled_task, session, simulation, simulation_run, skill and voice. A scheduled task also has invalid_schedule (a schedule that does not read: at and in both or neither, a date, time or duration that does not parse, or until without repeat), schedule_in_past, schedule_too_far (a one-shot, or an in, more than a year ahead), invalid_timezone, timezone_unknown (no zone was given and neither the user nor the config has one), invalid_repeat, interval_too_short (a run moved to less than 15 minutes from now and earlier than the run it replaces) and too_many_tasks (a 409: the user is at their active-task limit). More may be added, so a client should expect one it does not know."`
	DocURL  string    `json:"doc_url" format:"uri" doc:"Where the code is explained."`
}

// newAPIError is a failure of errorType with the code of its type. A failure a client
// tells apart by more than its type is an APIError value of its own, with a code of its own.
func newAPIError(errorType ErrorType, message string) APIError {
	failure := APIError{Type: errorType, Message: message}
	for _, known := range errorTypes {
		if known.errorType == errorType {
			failure.Code = known.code
		}
	}
	return failure
}

func invalidRequest(message string) APIError {
	return newAPIError(ErrorTypeInvalidRequest, message)
}

func unauthenticated(message string) APIError {
	return newAPIError(ErrorTypeAuthentication, message)
}

func forbidden(message string) APIError {
	return newAPIError(ErrorTypePermission, message)
}

// paymentRequired is work refused for money: the code says whose money and why.
func paymentRequired(code, message string) APIError {
	return APIError{Type: ErrorTypePaymentRequired, Code: code, Message: message}
}

func notFound(message string) APIError {
	return newAPIError(ErrorTypeNotFound, message)
}

func conflict(message string) APIError {
	return newAPIError(ErrorTypeConflict, message)
}

func gone(message string) APIError {
	return newAPIError(ErrorTypeGone, message)
}

func payloadTooLarge(message string) APIError {
	return newAPIError(ErrorTypePayloadTooLarge, message)
}

func rateLimited(message string) APIError {
	return newAPIError(ErrorTypeRateLimited, message)
}

func unavailable(message string) APIError {
	return newAPIError(ErrorTypeUnavailable, message)
}

// notConfigured is a feature this deployment does not offer: the caller asked for it, so
// it is the request that cannot be served, whatever it says.
func notConfigured(message string) APIError {
	return APIError{Type: ErrorTypeInvalidRequest, Code: codeNotConfigured, Message: message}
}

// invalidTimezone is a zone a request named that nobody knows.
func invalidTimezone(message string) APIError {
	return APIError{Type: ErrorTypeInvalidRequest, Code: codeInvalidTimezone, Message: message}
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
