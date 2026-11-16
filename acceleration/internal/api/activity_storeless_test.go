package api

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/require"
)

// Defect: a deployment with no database, so no activity record, failing every request an app
// makes on a recorder that is not there.
func TestARouterWithNoActivityRecordStillServesAnAppsRequests(t *testing.T) {
	routes := chi.NewRouter()
	routes.Get("/v1/agents/configs", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	server := &Server{logger: slog.New(slog.DiscardHandler)}
	request := httptest.NewRequest(http.MethodGet, "/v1/agents/configs", nil)
	request = request.WithContext(context.WithValue(request.Context(), customerContextKey{}, "app"))
	answer := httptest.NewRecorder()

	server.withRequestLog(routes, routes).ServeHTTP(answer, request)

	require.Equal(t, http.StatusOK, answer.Code)
}
