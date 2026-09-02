package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

const (
	testKey    = "vak_live_0123456789abcdef00000000"
	testSecret = "vas_live_s3cret"
)

// upstream records what the router would have received, and answers 200.
type upstream struct {
	got *http.Request
	url *url.URL
}

func newUpstream(t *testing.T) *upstream {
	t.Helper()
	recorder := &upstream{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recorder.got = r.Clone(context.Background())
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	t.Cleanup(server.Close)

	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)
	recorder.url = parsed
	return recorder
}

// gateway wires the proxy against a recording upstream, with one known key.
func gateway(t *testing.T, perSecond float64, burst int) (http.Handler, *upstream) {
	t.Helper()
	router := newUpstream(t)

	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		if presented != testKey {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{OrganizationID: "org-1", AppID: "app-1", Secret: testSecret}, nil
	})
	require.NoError(t, err)

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return newProxy(router.url, authenticator, newLimiter(perSecond, burst), logger), router
}

func token(t *testing.T, secret string) string {
	t.Helper()
	signed, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
	}).SignedString([]byte(secret))
	require.NoError(t, err)
	return signed
}

// authed is a request carrying a valid credential.
func authed(t *testing.T, target string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, target, nil)
	r.Header.Set(auth.APIKeyHeader, testKey)
	r.Header.Set("Authorization", "Bearer "+token(t, testSecret))
	return r
}

func TestTheRouterIsToldWhoTheCallerIs(t *testing.T) {
	handler, router := gateway(t, 100, 100)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authed(t, "/v1/calls"))

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, "app-1", router.got.Header.Get(auth.AppHeader))
	require.Equal(t, "org-1", router.got.Header.Get(auth.OrganizationHeader))
}

func TestACallerCannotNameItself(t *testing.T) {
	// The router believes these headers without verifying them. If a caller's
	// own copy survived the proxy, the whole arrangement would be decorative.
	handler, router := gateway(t, 100, 100)

	request := authed(t, "/v1/calls")
	request.Header.Set(auth.AppHeader, "someone-else")
	request.Header.Set(auth.OrganizationHeader, "another-org")
	request.Header.Set(auth.CustomerHeader, "a-third-tenant")

	handler.ServeHTTP(httptest.NewRecorder(), request)

	require.Equal(t, "app-1", router.got.Header.Get(auth.AppHeader))
	require.Equal(t, "org-1", router.got.Header.Get(auth.OrganizationHeader))
	require.Empty(t, router.got.Header.Get(auth.CustomerHeader))
}

func TestASocketCannotNameItselfInTheQueryEither(t *testing.T) {
	// The router reads customer_id off the query string when there is no
	// header, so leaving it there would be the same hole by another route.
	handler, router := gateway(t, 100, 100)

	handler.ServeHTTP(httptest.NewRecorder(),
		authed(t, "/v1/agents/sessions/abc/events?customer_id=someone-else&interim=true"))

	require.NotContains(t, router.got.URL.RawQuery, "customer_id")
	require.Contains(t, router.got.URL.RawQuery, "interim=true")
	require.Equal(t, "app-1", router.got.Header.Get(auth.AppHeader))
}

func TestTheCredentialIsNotForwarded(t *testing.T) {
	// The router has no use for it, and it should not reach that far.
	handler, router := gateway(t, 100, 100)

	handler.ServeHTTP(httptest.NewRecorder(), authed(t, "/v1/calls"))

	require.Empty(t, router.got.Header.Get("Authorization"))
	require.Empty(t, router.got.Header.Get(auth.APIKeyHeader))
}

func TestAnUnauthenticatedRequestNeverReachesTheRouter(t *testing.T) {
	handler, router := gateway(t, 100, 100)

	attempts := map[string]*http.Request{
		"no credential": httptest.NewRequest(http.MethodGet, "/v1/calls", nil),
		"unknown key":   httptest.NewRequest(http.MethodGet, "/v1/calls", nil),
		"malformed key": httptest.NewRequest(http.MethodGet, "/v1/calls", nil),
		"wrong secret":  httptest.NewRequest(http.MethodGet, "/v1/calls", nil),
		"proxy headers": httptest.NewRequest(http.MethodGet, "/v1/calls", nil),
	}
	attempts["unknown key"].Header.Set(auth.APIKeyHeader, "vak_live_ffffffffffffffff00000000")
	attempts["unknown key"].Header.Set("Authorization", "Bearer "+token(t, testSecret))
	attempts["malformed key"].Header.Set(auth.APIKeyHeader, "nonsense")
	attempts["malformed key"].Header.Set("Authorization", "Bearer "+token(t, testSecret))
	attempts["wrong secret"].Header.Set(auth.APIKeyHeader, testKey)
	attempts["wrong secret"].Header.Set("Authorization", "Bearer "+token(t, "vas_live_wrong"))
	attempts["proxy headers"].Header.Set(auth.AppHeader, "app-1")
	attempts["proxy headers"].Header.Set(auth.OrganizationHeader, "org-1")

	var bodies []string
	for name, request := range attempts {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)

		require.Equal(t, http.StatusUnauthorized, recorder.Code, name)
		require.Nil(t, router.got, "%s reached the router", name)
		bodies = append(bodies, recorder.Body.String())
	}
	// One answer for every reason, so the difference cannot be used to find out
	// which keys exist.
	for _, body := range bodies {
		require.Equal(t, bodies[0], body)
	}
}

func TestHealthAnswersWithoutACredential(t *testing.T) {
	// A liveness probe holds no API key.
	handler, router := gateway(t, 100, 100)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/health", nil))

	require.Equal(t, http.StatusOK, recorder.Code)
	require.NotNil(t, router.got)
}

func TestARunawayAppThrottlesItself(t *testing.T) {
	handler, _ := gateway(t, 1, 2)

	var statuses []int
	for range 5 {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, authed(t, "/v1/calls"))
		statuses = append(statuses, recorder.Code)
	}

	require.Equal(t, http.StatusOK, statuses[0])
	require.Equal(t, http.StatusOK, statuses[1])
	require.Equal(t, http.StatusTooManyRequests, statuses[4])
}

func TestThrottlingIsPerApp(t *testing.T) {
	// One integration running hot must not slow another app down.
	limiter := newLimiter(1, 1)

	require.True(t, limiter.allow("app-1"))
	require.False(t, limiter.allow("app-1"))
	require.True(t, limiter.allow("app-2"))
}

func TestATooManyRequestsAnswerSaysWhenToRetry(t *testing.T) {
	handler, _ := gateway(t, 1, 1)

	handler.ServeHTTP(httptest.NewRecorder(), authed(t, "/v1/calls"))

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authed(t, "/v1/calls"))

	require.Equal(t, http.StatusTooManyRequests, recorder.Code)
	require.NotEmpty(t, recorder.Header().Get("Retry-After"))
}

func TestAnUnreachableRouterIsABadGateway(t *testing.T) {
	// Not a 500: the gateway is fine, the thing behind it is not, and an
	// operator reading the status code should be sent to the right process.
	unreachable, err := url.Parse("http://127.0.0.1:1")
	require.NoError(t, err)

	authenticator, err := auth.New(auth.APIKey, func(context.Context, string) (auth.App, error) {
		return auth.App{OrganizationID: "org-1", AppID: "app-1", Secret: testSecret}, nil
	})
	require.NoError(t, err)

	handler := newProxy(unreachable, authenticator, newLimiter(100, 100),
		slog.New(slog.NewTextHandler(io.Discard, nil)))

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, authed(t, "/v1/calls"))

	require.Equal(t, http.StatusBadGateway, recorder.Code)
	require.True(t, strings.Contains(recorder.Body.String(), "router"))
}
