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

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// TestASocketSurvivesTheProxy is the case most likely to break without anyone
// noticing until a call drops: three of the router's endpoints are long-lived
// WebSockets, and a proxy that quietly answered the upgrade itself, or buffered
// it, would look fine to every request-shaped test above.
func TestASocketSurvivesTheProxy(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

	var sawApp, sawOrg string
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawApp = r.Header.Get(auth.AppHeader)
		sawOrg = r.Header.Get(auth.OrganizationHeader)

		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()

		kind, message, err := connection.ReadMessage()
		if err != nil {
			return
		}
		_ = connection.WriteMessage(kind, append([]byte("echo:"), message...))
	}))
	defer router.Close()

	routerURL, err := url.Parse(router.URL)
	require.NoError(t, err)

	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		if presented != testKey {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{OrganizationID: "org-1", AppID: "app-1", Secret: testSecret}, nil
	})
	require.NoError(t, err)

	proxy := httptest.NewServer(newProxy(routerURL, authenticator, newLimiter(100, 100),
		slog.New(slog.NewTextHandler(io.Discard, nil))))
	defer proxy.Close()

	header := http.Header{}
	header.Set(auth.APIKeyHeader, testKey)
	header.Set("Authorization", "Bearer "+token(t, testSecret))

	connection, response, err := websocket.DefaultDialer.Dial(
		strings.Replace(proxy.URL, "http", "ws", 1)+"/v1/dispatch", header)
	require.NoError(t, err)
	defer connection.Close()
	require.Equal(t, http.StatusSwitchingProtocols, response.StatusCode)

	require.NoError(t, connection.WriteMessage(websocket.TextMessage, []byte("hello")))
	_, message, err := connection.ReadMessage()
	require.NoError(t, err)
	require.Equal(t, "echo:hello", string(message))

	// The identity has to survive the upgrade too: the router reads it once,
	// on the handshake, and the connection is then trusted for its whole life.
	require.Equal(t, "app-1", sawApp)
	require.Equal(t, "org-1", sawOrg)
}

// TestAnUnauthenticatedSocketIsRefusedBeforeTheUpgrade proves the credential is
// checked on the handshake rather than after, when refusing costs a live
// connection.
func TestAnUnauthenticatedSocketIsRefusedBeforeTheUpgrade(t *testing.T) {
	handler, router := gateway(t, 100, 100)
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	_, response, err := websocket.DefaultDialer.Dial(
		strings.Replace(proxy.URL, "http", "ws", 1)+"/v1/dispatch", nil)
	require.Error(t, err)
	require.Equal(t, http.StatusUnauthorized, response.StatusCode)
	require.Nil(t, router.got)
}
