package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// streamRetryAfter is how long a caller is told to wait while the router does not yet know
// which Stream app is its own, which it learns from Stream within a few attempts.
const streamRetryAfter = 30 * time.Second

// streamUnknown is what work that may be the deployment app's answers until that app's id
// is known: try again shortly, rather than a guess at which app to act in.
const streamUnknown = "the router does not know which Stream app is its own yet: try again shortly"

// answerFailure answers an error a generated operation returned. Work waiting on the
// deployment's own app is a 503 the caller can retry; anything else is a 500, as before.
func answerFailure(w http.ResponseWriter, _ *http.Request, err error) {
	if errors.Is(err, streamapp.ErrDeploymentAppUnknown) {
		w.Header().Set("Retry-After", strconv.Itoa(int(streamRetryAfter.Seconds())))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(Error{Error: streamUnknown})
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// streamWaiting is the same answer from an operation declared in Go.
func streamWaiting() error {
	return &apiError{
		status:  http.StatusServiceUnavailable,
		headers: http.Header{"Retry-After": {strconv.Itoa(int(streamRetryAfter.Seconds()))}},
		Message: streamUnknown,
	}
}

// mintingKeyHeader names which of the calling app's registered keys a gateway that writes
// it itself wants the app's tokens minted with.
const mintingKeyHeader = "X-Stream-Api-Key"

type mintingKeyContextKey struct{}

// minting is the bound identity a token is minted with: the key the gateway named, when the
// deployment trusts it to and the key is one the app holds; otherwise as it was resolved.
func (s *Server) minting(ctx context.Context, bound streamapp.Bound) (streamapp.Bound, error) {
	named, _ := ctx.Value(mintingKeyContextKey{}).(string)
	if named == "" || s.stream == nil {
		return bound, nil
	}
	return s.stream.WithKey(ctx, bound, named)
}
