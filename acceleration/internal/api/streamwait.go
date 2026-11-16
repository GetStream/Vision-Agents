package api

import (
	"net/http"
	"strconv"
	"time"
)

// streamRetryAfter is how long a caller is told to wait while the router does not yet know
// which Stream app is its own, which it learns from Stream within a few attempts.
const streamRetryAfter = 30 * time.Second

// errStreamUnknown is what work that may be the deployment app's answers until that app's
// id is known: try again shortly, rather than a guess at which app to act in.
var errStreamUnknown = unavailable("the router does not know which Stream app is its own yet: try again shortly")

// writeStreamWaiting answers a request waiting on the deployment's own app: try again.
func writeStreamWaiting(w http.ResponseWriter) {
	w.Header().Set("Retry-After", strconv.Itoa(int(streamRetryAfter.Seconds())))
	writeError(w, errStreamUnknown)
}

// streamWaitingError is errStreamUnknown with the Retry-After Huma answers it with.
type streamWaitingError struct{ APIError }

func (streamWaitingError) GetHeaders() http.Header {
	return http.Header{"Retry-After": {strconv.Itoa(int(streamRetryAfter.Seconds()))}}
}

// streamWaiting is the same answer from an operation declared in Go.
func streamWaiting() error {
	return streamWaitingError{errStreamUnknown}
}
