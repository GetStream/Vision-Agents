//go:build integration

package api

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/redis/rueidis"
)

func TestExhaustedQuotaAllowsSessionDeletionThroughMiddleware(t *testing.T) {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	redis, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, Username: os.Getenv("ROUTER_REDIS_USERNAME"), Password: os.Getenv("ROUTER_REDIS_PASSWORD"), DisableCache: true})
	if err != nil {
		t.Fatal(err)
	}
	defer redis.Close()
	limiter, err := quota.New(redis, quota.Limits{MessagesPerDay: 1}, slog.Default())
	if err != nil {
		t.Fatal(err)
	}
	customer := fmt.Sprintf("cleanup-test-%d", time.Now().UnixNano())
	caller := routing.Caller{UserID: "owner"}
	ctx := context.WithValue(context.Background(), customerContextKey{}, customer)
	ctx = context.WithValue(ctx, callerContextKey{}, caller)
	limiter.Debit(ctx, customer, caller, 1)
	if limiter.Allow(ctx, customer, caller) == nil {
		t.Fatal("quota was not exhausted")
	}
	server := &Server{quota: limiter, logger: slog.Default()}
	// The downstream route remains responsible for authenticating ownership.
	handler := server.withQuota(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/agents/sessions/owned" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	for _, test := range []struct {
		method, path string
		status       int
	}{
		{"DELETE", "/v1/agents/sessions/owned", 204},
		{"DELETE", "/v1/agents/sessions/not-owned", 404},
		{"POST", "/v1/agents/sessions", 429},
		{"POST", "/v1/agents/sessions/owned/respond", 429},
		{"DELETE", "/v1/agents/sessions/owned/respond", 429},
		{"DELETE", "/v1/agents/configs/owned", 429},
	} {
		request := httptest.NewRequest(test.method, test.path, nil).WithContext(ctx)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != test.status {
			t.Fatalf("%s %s: got %d, want %d", test.method, test.path, response.Code, test.status)
		}
	}
	if limiter.Allow(ctx, customer, caller) == nil {
		t.Fatal("cleanup changed the spent generation quota")
	}
}
