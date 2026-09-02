// Command gateway is the authenticating proxy in front of the router.
//
// The router runs with ROUTER_AUTH_MODE=noauth: it believes the organization and
// app named in the request headers and verifies nothing, which is only safe
// because a NetworkPolicy means this process is the only thing that can reach
// it. Everything the router does not do is here — authenticating the caller,
// rate limiting them, and naming them to the router in headers it can trust.
//
// The headers are overwritten rather than merged. A caller that sent its own
// copy of X-Stream-App-Id would otherwise be choosing which tenant it is.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

const (
	addressEnvVar  = "GATEWAY_ADDR"
	upstreamEnvVar = "GATEWAY_UPSTREAM"
	logLevelEnvVar = "GATEWAY_LOG_LEVEL"
	// rateEnvVar and burstEnvVar are per app, not per organization: one runaway
	// integration should throttle itself rather than the account's production
	// traffic.
	rateEnvVar  = "GATEWAY_RATE_PER_SECOND"
	burstEnvVar = "GATEWAY_BURST"

	defaultAddress  = ":8081"
	defaultUpstream = "http://127.0.0.1:8080"
	defaultRate     = 50
	defaultBurst    = 100

	// A session socket pings every 30s and gives up on a silent peer after 90s,
	// so anything shorter here would break a call that was merely quiet.
	idleTimeout       = 10 * time.Minute
	readHeaderTimeout = 10 * time.Second
	shutdownGrace     = 30 * time.Second
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel()}))
	slog.SetDefault(logger)

	if err := run(logger); err != nil {
		logger.Error("gateway stopped", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	upstream, err := url.Parse(envOr(upstreamEnvVar, defaultUpstream))
	if err != nil {
		return err
	}

	// The gateway is the thing that verifies keys, so it runs the same api_key
	// authenticator the router would in a deployment without a proxy.
	lookup, closeStore, err := newLookup(os.Getenv(postgresEnvVar), os.Getenv(kekEnvVar), logger)
	if err != nil {
		return err
	}
	defer func() {
		if err := closeStore(); err != nil {
			logger.Error("could not close the key store", "error", err)
		}
	}()

	authenticator, err := auth.New(auth.APIKey, lookup)
	if err != nil {
		return err
	}

	handler := newProxy(upstream, authenticator, newLimiter(rateLimit(), burst()), logger)

	server := &http.Server{
		Addr:              envOr(addressEnvVar, defaultAddress),
		Handler:           handler,
		ReadHeaderTimeout: readHeaderTimeout,
		// No write or read deadline: a session socket is meant to stay open for
		// the length of a conversation, and a deadline here would cut it.
		IdleTimeout: idleTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		logger.Info("gateway listening", "addr", server.Addr, "upstream", upstream.String())
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
		}
	}()

	select {
	case err := <-errs:
		return err
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

// newProxy builds the handler: authenticate, rate limit, rewrite, forward.
func newProxy(upstream *url.URL, authenticator auth.Authenticator, limiter *limiter, logger *slog.Logger) http.Handler {
	proxy := &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(upstream)
			r.Out.Host = r.In.Host
			// The principal was resolved before this ran and is carried on the
			// outbound request only. SetURL does not copy headers between them,
			// so what the client sent cannot leak through here.
			principal := principalFrom(r.In.Context())
			r.Out.Header.Set(auth.OrganizationHeader, principal.OrganizationID)
			r.Out.Header.Set(auth.AppHeader, principal.AppID)
			// Set from the principal rather than forwarded, because the router believes
			// this header: a caller sending its own copy would otherwise promote itself
			// to the backend it is not and rewrite the agent it is only meant to talk to.
			r.Out.Header.Set(auth.AuthTypeHeader, authTypeOf(principal))
			r.Out.Header.Del(auth.CustomerHeader)
			r.Out.Header.Del("Authorization")
			r.Out.Header.Del(auth.APIKeyHeader)
			r.SetXForwarded()
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			logger.Error("could not reach the router", "error", err)
			writeError(w, http.StatusBadGateway, "the router is unavailable")
		},
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The health check has to answer without a credential, or a liveness
		// probe would need one.
		if r.URL.Path == "/health" {
			proxy.ServeHTTP(w, r)
			return
		}

		principal, err := authenticator.Authenticate(r.Context(), r)
		if err != nil {
			// One answer for every reason, so the difference between an unknown
			// key and a bad token cannot be used to find out which keys exist.
			writeError(w, http.StatusUnauthorized, "unauthenticated")
			return
		}

		if !limiter.allow(principal.AppID) {
			w.Header().Set("Retry-After", "1")
			writeError(w, http.StatusTooManyRequests, "too many requests")
			return
		}

		// The credential names the caller; the query string must not. A socket
		// that kept customer_id would let a caller ask for another tenant's
		// events on a connection this process had already authenticated.
		if query := r.URL.Query(); query.Has(auth.CustomerParam) {
			query.Del(auth.CustomerParam)
			r.URL.RawQuery = query.Encode()
		}

		proxy.ServeHTTP(w, r.WithContext(withPrincipal(r.Context(), principal)))
	})
}

type principalContextKey struct{}

func withPrincipal(ctx context.Context, principal auth.Principal) context.Context {
	return context.WithValue(ctx, principalContextKey{}, principal)
}

func principalFrom(ctx context.Context) auth.Principal {
	principal, _ := ctx.Value(principalContextKey{}).(auth.Principal)
	return principal
}

// authTypeOf names which kind of credential the caller proved it had.
func authTypeOf(principal auth.Principal) string {
	if principal.ServerSide {
		return auth.AuthTypeServer
	}
	return auth.AuthTypeJWT
}

func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(`{"error":"` + message + `"}`))
}

func envOr(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func rateLimit() float64 {
	if value, err := strconv.ParseFloat(os.Getenv(rateEnvVar), 64); err == nil && value > 0 {
		return value
	}
	return defaultRate
}

func burst() int {
	if value, err := strconv.Atoi(os.Getenv(burstEnvVar)); err == nil && value > 0 {
		return value
	}
	return defaultBurst
}

func logLevel() slog.Level {
	switch strings.ToLower(os.Getenv(logLevelEnvVar)) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
