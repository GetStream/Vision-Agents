package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

const customerSite = "https://shop.customer.example"

func preflight(mode auth.Mode, origin string) *httptest.ResponseRecorder {
	handler := withCORS(mode, http.NotFoundHandler())
	request := httptest.NewRequest(http.MethodOptions, "/v1/sessions", nil)
	request.Header.Set("Origin", origin)
	request.Header.Set("Access-Control-Request-Method", http.MethodPost)
	request.Header.Set("Access-Control-Request-Headers", "authorization,stream-auth-type,api_key,content-type")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

// dialSession opens a socket from a page on origin to a server upgrading as mode does.
func dialSession(t *testing.T, mode auth.Mode, origin string) (*http.Response, error) {
	t.Helper()
	upgrader := newUpgrader(mode)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		_ = connection.Close()
	}))
	t.Cleanup(server.Close)
	header := http.Header{}
	if origin != "" {
		header.Set("Origin", origin)
	}
	connection, response, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), header)
	if err == nil {
		_ = connection.Close()
	}
	if response != nil {
		_ = response.Body.Close()
	}
	return response, err
}

func TestACustomersWebsiteMayCallTheAPIWithItsOwnCredential(t *testing.T) {
	for _, mode := range []auth.Mode{auth.APIKey, auth.Proxy} {
		response := preflight(mode, customerSite)

		require.Equal(t, http.StatusNoContent, response.Code, mode)
		require.Equal(t, customerSite, response.Header().Get("Access-Control-Allow-Origin"), mode)
		for _, header := range []string{"Authorization", auth.AuthTypeHeader, auth.APIKeyHeader} {
			require.Contains(t, response.Header().Get("Access-Control-Allow-Headers"), header, mode)
		}
	}
}

func TestACustomersWebsiteMayReadTheRateLimitOfAnAnswer(t *testing.T) {
	for _, mode := range []auth.Mode{auth.APIKey, auth.Proxy} {
		handler := withCORS(mode, http.NotFoundHandler())
		request := httptest.NewRequest(http.MethodGet, "/v1/sessions", nil)
		request.Header.Set("Origin", customerSite)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)

		for _, header := range []string{"X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"} {
			require.Contains(t, response.Header().Get("Access-Control-Expose-Headers"), header, mode)
		}
	}
}

func TestACustomersWebsiteMayOpenASessionSocket(t *testing.T) {
	for _, mode := range []auth.Mode{auth.APIKey, auth.Proxy} {
		_, err := dialSession(t, mode, customerSite)
		require.NoError(t, err, mode)
	}
}

func TestALocalRouterWithoutAuthAnswersOnlyPagesOnThisMachine(t *testing.T) {
	for _, origin := range []string{"http://localhost:3000", "http://127.0.0.1:5173", "http://[::1]:8080"} {
		require.Equal(t, origin, preflight(auth.NoAuth, origin).Header().Get("Access-Control-Allow-Origin"))
		_, err := dialSession(t, auth.NoAuth, origin)
		require.NoError(t, err, origin)
	}

	require.Empty(t, preflight(auth.NoAuth, customerSite).Header().Get("Access-Control-Allow-Origin"),
		"a site open in the developer's browser could otherwise name any tenant")
	response, err := dialSession(t, auth.NoAuth, customerSite)
	require.ErrorIs(t, err, websocket.ErrBadHandshake)
	require.Equal(t, http.StatusForbidden, response.StatusCode)
}

func TestASocketFromNoBrowserIsAcceptedInEveryMode(t *testing.T) {
	for _, mode := range []auth.Mode{auth.APIKey, auth.Proxy, auth.NoAuth} {
		_, err := dialSession(t, mode, "")
		require.NoError(t, err, mode)
	}
}
