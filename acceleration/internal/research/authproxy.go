package research

import (
	"crypto/subtle"
	"crypto/tls"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

// A process can always read its own /proc/<pid>/environ regardless of file permissions,
// user or namespace, so a credential placed in Cursor's environment is readable by the
// model running inside it. Cursor is therefore started with CredentialPlaceholder and
// pointed at this loopback proxy instead. The real credential stays in the root-owned
// worker and is attached to the single request that needs it: the API key exchange.
// Everything after that carries the short-lived session token Cursor received in return,
// which is passed through untouched.
const CredentialPlaceholder = "stream-research-no-credential-in-this-process"

// The one upstream route that accepts the API key itself.
const authExchangePath = "/auth/exchange_user_api_key"

// Overridden only by tests, which stand in a local upstream.
var cursorAPI = "https://api2.cursor.sh"

type AuthProxy struct {
	listener net.Listener
	server   *http.Server
}

// StartAuthProxy listens on the loopback interface only. Cursor has to be able to reach
// it, so it cannot be authenticated: anything Cursor could present as proof would have
// to live in its environment or argv, which is exactly what this avoids. The proxy is
// not an open relay - it forwards only to Cursor's API, and it never writes the
// credential into a response, so reaching it does not disclose the credential.
func StartAuthProxy(credential string) (*AuthProxy, error) {
	if credential == "" {
		return nil, errors.New("research: Cursor credential required")
	}
	upstream, err := url.Parse(cursorAPI)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	presented := []byte("Bearer " + CredentialPlaceholder)
	proxy := &httputil.ReverseProxy{
		FlushInterval: -1, // Model responses stream; never buffer them.
		Transport: &http.Transport{
			TLSClientConfig:     &tls.Config{MinVersion: tls.VersionTLS12},
			ForceAttemptHTTP2:   true,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 20 * time.Second,
		},
		Director: func(r *http.Request) {
			r.URL.Scheme = upstream.Scheme
			r.URL.Host = upstream.Host
			r.Host = upstream.Host
			offered := []byte(r.Header.Get("Authorization"))
			if r.URL.Path == authExchangePath && subtle.ConstantTimeCompare(offered, presented) == 1 {
				r.Header.Set("Authorization", "Bearer "+credential)
			}
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Warn("cursor api request failed", "path", r.URL.Path, "error", err)
			w.WriteHeader(http.StatusBadGateway)
		},
	}
	p := &AuthProxy{listener: listener, server: &http.Server{Handler: proxy, ReadHeaderTimeout: 10 * time.Second}}
	go func() {
		if err := p.server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("cursor credential proxy stopped", "error", err)
		}
	}()
	return p, nil
}

// Endpoint is the CURSOR_API_ENDPOINT value for the research process.
func (p *AuthProxy) Endpoint() string {
	return "http://" + p.listener.Addr().String()
}
func (p *AuthProxy) Close() error { return p.server.Close() }
