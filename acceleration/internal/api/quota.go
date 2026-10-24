package api

import (
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
)

const (
	quotaLimitHeader     = "X-Quota-Limit"
	quotaRemainingHeader = "X-Quota-Remaining"
	quotaResetHeader     = "X-Quota-Reset"
)

// quotaHeaders tells a counted caller where they stand, the way Stream Chat does: the
// day's allowance, what is left of it and the unix time it resets at. A caller nothing was
// counted for is told nothing.
func quotaHeaders(header http.Header, status quota.Status) {
	if status.Limit == 0 {
		return
	}
	header.Set(quotaLimitHeader, strconv.FormatInt(status.Limit, 10))
	header.Set(quotaRemainingHeader, strconv.FormatInt(status.Remaining, 10))
	header.Set(quotaResetHeader, strconv.FormatInt(status.Reset.Unix(), 10))
}

// modelWork is the routes that start model responses a quota counts: a session, which may
// greet; a turn; a run; and the sockets that take turns, refused only as they are opened.
var modelWork = func() *http.ServeMux {
	routes := http.NewServeMux()
	nothing := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	for _, pattern := range []string{
		"POST /v1/agents/sessions",
		"POST /v1/agents/sessions/{id}/respond",
		"POST /v1/agents/sessions/{id}/responses",
		"POST /v1/agents/runs",
		"GET /v1/agents/sessions/{id}/events",
		"GET /v1/{modality}/stream",
	} {
		routes.Handle(pattern, nothing)
	}
	return routes
}()

// startsModelWork reports whether a request starts model work: a write to one of modelWork's
// routes, or a socket opened on one. A read never does.
func startsModelWork(r *http.Request) bool {
	if _, pattern := modelWork.Handler(r); pattern == "" {
		return false
	}
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		return strings.EqualFold(r.Header.Get("Upgrade"), "websocket")
	}
	return true
}

// withQuota refuses a caller who has spent their day before any work is started.
//
// It is not the only place the limit is enforced, and it is not the one that matters most:
// a socket and a call both keep asking for responses long after the request that opened
// them was admitted, so the limit is really enforced where a response is created. What this
// adds is the answer. A handler that discovers the limit is spent can only return an error,
// and none of the generated operations declare a 429, so without this the caller would be
// told 500 by a server that knew perfectly well what was wrong.
//
// It sits inside withRequestLog so a refusal is logged, and outside withServerSide for the
// same reason that one sits where it does: refusing a caller for who they are means having
// worked out who they are first.
func (s *Server) withQuota(next http.Handler) http.Handler {
	if s.quota == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Stopping or deleting an existing session cannot create new model work. Keep the
		// normal authentication/ownership handler, even after generation is spent.
		sessionID, sessionPath := strings.CutPrefix(r.URL.Path, "/v1/agents/sessions/")
		stopped, stopping := strings.CutSuffix(sessionID, "/stop")
		endingSession := sessionPath &&
			(r.Method == http.MethodDelete && sessionID != "" && !strings.Contains(sessionID, "/") ||
				r.Method == http.MethodPost && stopping && stopped != "" && !strings.Contains(stopped, "/"))
		// A request nobody authenticated has no account to count against. A backend, and a
		// caller with neither a name nor an address, are told apart by Allow itself: a
		// backend naming a user is held to the quotas the customer set for that user, and
		// never to the deployment's default, which guards against devices alone.
		customerID, known := CustomerFrom(r.Context())
		if !known || endingSession {
			next.ServeHTTP(w, r)
			return
		}
		caller := CallerFrom(r.Context())
		// A backend is refused only work that asks a model for something: it reads and
		// manages its user's sessions, and everything else, whatever that user has spent.
		if caller.ServerSide && !startsModelWork(r) {
			next.ServeHTTP(w, r)
			return
		}

		status, err := s.quota.Allow(r.Context(), customerID, caller)
		quotaHeaders(w.Header(), status)
		if err != nil {
			if !errors.Is(err, quota.ErrExhausted) {
				next.ServeHTTP(w, r)
				return
			}
			s.logger.Info("refused a caller who has spent their day",
				"method", r.Method, "path", r.URL.Path, "customer", customerID)
			w.Header().Set("Retry-After", strconv.Itoa(max(int(time.Until(status.Reset).Seconds()), 1)))
			writeError(w, rateLimited(err.Error()))
			return
		}
		next.ServeHTTP(w, r)
	})
}
