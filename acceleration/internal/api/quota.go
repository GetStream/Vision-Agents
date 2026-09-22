package api

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
)

// retryAfter is what an exhausted caller is told to wait. A daily limit resets at midnight
// UTC, and the honest number is the time until then.
func retryAfter(now time.Time) int {
	midnight := now.UTC().Truncate(24 * time.Hour).Add(24 * time.Hour)
	seconds := int(midnight.Sub(now.UTC()).Seconds())
	if seconds < 1 {
		return 1
	}
	return seconds
}

// exemptFromQuota reports whether the day's limit says nothing about this request.
//
// Three reasons, and only the middle one is a decision. A request nobody authenticated has
// no account to count against; a caller with neither a name nor an address has no bucket
// to count in.
//
// The middle one is that a process the customer runs is trusted with its own spend. It is
// asked about directly rather than inferred from having no caller, because a backend may
// name one — that is how it says which of its users a session belongs to — and counting
// that name would charge a person's day for work their customer chose to do for them.
func exemptFromQuota(ctx context.Context) bool {
	_, known := CustomerFrom(ctx)
	return !known || ServerSideFrom(ctx) || CallerFrom(ctx).Anonymous()
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
		// Closing an existing session cannot create new model work. Keep the
		// normal authentication/ownership handler, even after generation is spent.
		sessionID, sessionPath := strings.CutPrefix(r.URL.Path, "/v1/agents/sessions/")
		closingSession := r.Method == http.MethodDelete && sessionPath && sessionID != "" && !strings.Contains(sessionID, "/")
		if exemptFromQuota(r.Context()) || closingSession {
			next.ServeHTTP(w, r)
			return
		}
		customerID, _ := CustomerFrom(r.Context())
		caller := CallerFrom(r.Context())

		if err := s.quota.Allow(r.Context(), customerID, caller); err != nil {
			if !errors.Is(err, quota.ErrExhausted) {
				next.ServeHTTP(w, r)
				return
			}
			s.logger.Info("refused a caller who has spent their day",
				"method", r.Method, "path", r.URL.Path, "customer", customerID)
			w.Header().Set("Retry-After", strconv.Itoa(retryAfter(time.Now())))
			writeError(w, http.StatusTooManyRequests, err.Error())
			return
		}
		next.ServeHTTP(w, r)
	})
}
