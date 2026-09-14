package api

import (
	"errors"
	"net/http"
	"strconv"
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
		customerID, known := CustomerFrom(r.Context())
		caller := CallerFrom(r.Context())
		if !known || caller.Anonymous() {
			next.ServeHTTP(w, r)
			return
		}

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
