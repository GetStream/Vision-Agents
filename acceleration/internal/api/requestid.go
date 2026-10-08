package api

import (
	"context"
	"errors"
	"net/http"
	"runtime/debug"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// RequestIDHeader names one request on both sides of it: answered on every response, and
// on every access log line, so a failure a client reports can be found in the log.
const RequestIDHeader = "X-Request-Id"

// maxRequestIDLength bounds an id a caller chose, which is repeated on every line the
// request logs.
const maxRequestIDLength = 128

type requestIDContextKey struct{}

// RequestIDFrom returns the identifier withRequestID gave the request.
func RequestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(requestIDContextKey{}).(string)
	return id
}

// withRequestID gives every request an identifier and answers with it.
//
// One a proxy in front already assigned is kept, so the proxy's log and this one name the
// request the same way. Anything else -- missing, too long, or not printable ASCII -- is
// replaced rather than refused, because it ends up in the log as written.
func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(RequestIDHeader)
		if !validRequestID(id) {
			id = uuid.NewString()
		}
		w.Header().Set(RequestIDHeader, id)
		ctx := context.WithValue(r.Context(), requestIDContextKey{}, id)
		// The connector audit rows the request causes, such as a refresh, name it too.
		ctx = core.WithCorrelation(ctx, core.Correlation{RequestID: id})
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func validRequestID(id string) bool {
	if id == "" || len(id) > maxRequestIDLength {
		return false
	}
	for i := range len(id) {
		if id[i] < '!' || id[i] > '~' {
			return false
		}
	}
	return true
}

// requestFailure is the error behind a request's answer, left for withRequestLog to report:
// the answer itself is deliberately vaguer than the error.
type requestFailure struct {
	err   error
	trace string
}

type requestFailureContextKey struct{}

func recordFailure(ctx context.Context, err error, trace string) {
	if failure, ok := ctx.Value(requestFailureContextKey{}).(*requestFailure); ok {
		failure.err = err
		failure.trace = trace
	}
}

// answerFailure is what a Huma operation's failure is answered with, set as
// huma.NewErrorWithContext. Huma calls it for an error an operation returned that names no
// status of its own, and for a request that did not validate.
//
// The errors are left for withRequestLog. A server error is answered as an internal one,
// saying "something went wrong" and leaving the rest to the request id: what failed -- a database or a vendor saying no -- is nothing
// the caller can act on, and is what the log is for. Its stack is the one stack.Wrap
// recorded where the error entered this code base, or, for an error that never was, the
// stack it reached Huma on, which names the operation but not where inside it.
func answerFailure(ctx huma.Context, status int, message string, errs ...error) huma.StatusError {
	// Work waiting on the deployment's own app is a 503 the caller can retry, not a failure.
	// Huma takes headers only from an error an operation returned itself, so Retry-After is
	// set here.
	if errors.Is(errors.Join(errs...), streamapp.ErrDeploymentAppUnknown) {
		waiting := streamWaitingError{errStreamUnknown}
		for name, values := range waiting.GetHeaders() {
			for _, value := range values {
				ctx.SetHeader(name, value)
			}
		}
		return waiting
	}
	if status < http.StatusInternalServerError {
		if len(errs) > 0 {
			// Without a secret's value, as the answer leaves it out.
			recordFailure(ctx.Context(), errors.Join(withoutRefusedValues(errs)...), "")
		}
		return huma.NewError(status, message, errs...)
	}

	err := errors.New(message)
	switch len(errs) {
	case 0:
	case 1:
		err = errs[0]
	default:
		err = errors.Join(errs...)
	}
	trace := stack.Trace(err)
	if trace == "" {
		trace = string(debug.Stack())
	}
	recordFailure(ctx.Context(), err, trace)
	return internalError()
}

// writeFailure answers a request served by hand that failed in a way that is not the
// caller's, the way answerFailure answers an operation that did.
func writeFailure(w http.ResponseWriter, r *http.Request, err error) {
	trace := stack.Trace(err)
	if trace == "" {
		trace = string(debug.Stack())
	}
	recordFailure(r.Context(), err, trace)
	writeError(w, internalError())
}
