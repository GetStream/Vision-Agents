package stream

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// RequestIDHeader names the request the router answered, which is what to quote to support
// about one that failed.
const RequestIDHeader = "X-Request-Id"

// RouterError is what the router said went wrong: a request it answered with a failure
// rather than with what was asked for, or a socket it would not open.
//
// This package, client and agents all return one, so every refusal is read the same way:
//
//	var refused *stream.RouterError
//	if errors.As(err, &refused) && refused.Code == "not_configured" {
//	    ...
//	}
type RouterError struct {
	// Status is the HTTP status the router answered with.
	Status int
	// Type is the kind of failure, which is what decides the status: invalid_request,
	// authentication, permission, not_found, conflict, rate_limited, internal, unavailable
	// and a few more. Empty when what answered was not the router, such as a proxy.
	Type string
	// Code is what went wrong, for a program to branch on: not_configured,
	// validation_failed, server_side_only, session_not_found and others. More are added, so
	// expect one this SDK has never heard of.
	Code string
	// Message is what went wrong, for a person to read. When the answer was not the
	// router's envelope it is the body as it arrived, or the status when there was none.
	Message string
	// DocURL is where Code is explained.
	DocURL string
	// RequestID is the X-Request-Id the router answered with. A 500 says only "something
	// went wrong", so this is what to quote to support.
	RequestID string
	// Operation is what was being asked for, when the error names it: "looking up the
	// agent docs".
	Operation string

	// opening is what Error says ahead of the operation: the package that asked, which is
	// how this SDK's errors have always started.
	opening string
	// cause is what the socket dialler gave up with, for a refused upgrade.
	cause error
}

// NewRouterError reads what the router said out of an answer that was not the one asked
// for, and is how client and agents report a refusal as well as this package.
//
// from is the package asking and operation what it was asking for, which the error opens
// with. otherwise is the message for an answer with no body.
func NewRouterError(response *http.Response, body []byte, from, operation, otherwise string) *RouterError {
	refused := &RouterError{
		Status:    response.StatusCode,
		Message:   otherwise,
		RequestID: response.Header.Get(RequestIDHeader),
		Operation: operation,
		opening:   from,
	}

	var envelope acceleration.ErrorResponse
	if json.Unmarshal(body, &envelope) == nil && envelope.Error != (acceleration.ErrorDetail{}) {
		refused.Type = string(envelope.Error.Type)
		refused.Code = envelope.Error.Code
		refused.DocURL = envelope.Error.DocUrl
		if envelope.Error.Message != "" {
			refused.Message = envelope.Error.Message
		}
		return refused
	}
	if text := strings.TrimSpace(string(body)); text != "" {
		refused.Message = text
	}
	return refused
}

func (e *RouterError) Error() string {
	said := make([]string, 0, 3)
	for _, part := range []string{e.opening, e.Operation, e.Message} {
		if part != "" {
			said = append(said, part)
		}
	}
	return strings.Join(said, ": ")
}

// Unwrap is what the socket dialler said, for a refused upgrade, and nil otherwise.
func (e *RouterError) Unwrap() error { return e.cause }

// refusedSocket is a WebSocket upgrade the router answered with a failure instead. The
// dialler has already read the body, so this cannot block.
func refusedSocket(response *http.Response, cause error) *RouterError {
	body, _ := io.ReadAll(response.Body)
	refused := NewRouterError(response, body, "stream: the router refused the socket with "+response.Status,
		"", cause.Error())
	refused.cause = cause
	return refused
}

// readable is the HTTP client the generated one is given, so that a failure whose body is
// not the router's envelope still arrives as a response.
//
// The generated parser decodes every failure that says it is JSON into the envelope, and
// returns the decode error instead of the response when it is not one: a router from before
// the envelope, or a proxy answering for it.
type readable struct{ doer acceleration.HttpRequestDoer }

func (r readable) Do(request *http.Request) (*http.Response, error) {
	response, err := r.doer.Do(request)
	if err != nil || response.StatusCode < http.StatusBadRequest ||
		!strings.Contains(response.Header.Get("Content-Type"), "json") {
		return response, err
	}

	body, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if err != nil {
		return nil, err
	}
	response.Body = io.NopCloser(bytes.NewReader(body))
	if json.Unmarshal(body, new(acceleration.ErrorResponse)) != nil {
		response.Header = response.Header.Clone()
		response.Header.Del("Content-Type")
	}
	return response, nil
}
