package api

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/textproto"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// connectionProxyPath is where a connection's direct calls are sent: the provider's own API
// path follows it.
const connectionProxyPath = "/v1/agents/connections/{id}/proxy/"

// proxyMethods are the methods the proxy forwards: the five specifiedOperations declares
// (openapi.go), which are what a provider's REST API is called with.
var proxyMethods = []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete}

// proxyBase is the manifest endpoint role a direct call goes to (core.Manifest.Endpoints,
// «api_base»). Every path of the call is under it.
const proxyBase = "api_base"

// maxProxyBody is the largest request body the proxy forwards. It is read whole so the request
// can be sent once more after a 401 (core.Transports needs GetBody for that). The cap is Huma's
// default MaxBodyBytes (ensureMaxBodyBytes in huma.go, v2.39.1), the one an operation of this
// API reads a body with unless it raises its own. A choice, not a provider limit.
const maxProxyBody = 1 << 20

// routerHeaders are what the router reads to authenticate and name the caller (internal/auth).
// They are the caller's to the router, never the provider's, so none is forwarded. The
// scheme's Wrap then sets the provider's own credential.
// The gateway's server token for the app and its key, and the app it checked a webhook's
// signature for (auth.ServerTokenHeader, auth.ServerTokenKeyHeader, auth.HookAppHeader), are the
// router's as well: the token acts in the customer's Stream app (docs/stream-tenancy.md).
var routerHeaders = []string{"Authorization", auth.APIKeyHeader, auth.AuthTypeHeader, auth.OrganizationHeader,
	auth.AppHeader, auth.CustomerHeader, auth.UserHeader, auth.EmailHeader, auth.ServerTokenHeader, auth.ServerTokenKeyHeader,
	auth.HookAppHeader}

// plainHeaders are the caller's headers a direct call may forward and still have its error
// answer's start logged: each names a format, a coding, a language or the client's software,
// never a credential. Accept, Accept-Encoding and Accept-Language (RFC 9110 sections 12.5.1,
// 12.5.3 and 12.5.4) are what the caller takes; User-Agent (section 10.1.5) is the client's name,
// which Go's client sends (Go-http-client/1.1) as every SDK does, and Go's client sends
// Accept-Encoding: gzip too. A body's own headers are not listed: a call that sends one leaves
// no excerpt anyway. Any other header, a Cookie or a provider's own key such as
// X-Provider-Secret, may carry a secret in any form, so the line keeps no excerpt.
var plainHeaders = []string{"Accept", "Accept-Encoding", "Accept-Language", "User-Agent"}

// routerParams are the query parameters the router reads a credential and a caller from
// (auth.credentials, auth.go:457-470, and the customer and user it names).
var routerParams = []string{auth.APIKeyParam, auth.TokenParam, auth.CustomerParam, auth.UserParam}

// hopHeaders are the hop-by-hop headers, which a proxy does not forward (RFC 9110 section
// 7.6.1; the list is net/http/httputil's hopHeaders in go1.27, reverseproxy.go), and the
// forwarding headers net/http/httputil's Rewrite strips (reverseproxy.go:491-494), so the
// provider is not told the caller's address or the router's own front.
var hopHeaders = []string{"Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate",
	"Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade"}

var forwardingHeaders = []string{"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto"}

// proxyCallWindow is the window ProxyCallsPerMinute is counted over.
const proxyCallWindow = time.Minute

// proxyConnection forwards a direct call to the provider of a connection: the request as it
// came, under the manifest's api_base, with the connection's credential in place of the
// router's, and the provider's answer as it came. The connection's client does the rest
// (core.Transports): it resolves the credential, sends again once after a 401 the scheme can
// renew past, follows no redirect off the origin, and dials only public addresses.
//
// Who may call it is who may read the connection (mayReach): the app's backend for an
// app-owned connection, and for a user's, a backend acting for that user.
func (s *Server) proxyConnection(w http.ResponseWriter, r *http.Request) {
	// The spec marks it server-side only, but its path runs on past one segment, which the
	// server-side matcher's pattern does not, so it is refused here as well.
	if s.refuseClientSide(w, r) {
		return
	}
	ctx := r.Context()
	connection, err := s.reachableConnection(ctx, chi.URLParam(r, "id"))
	if err != nil {
		writeHandError(w, r, err)
		return
	}
	if connection.Status != store.ConnectionConnected {
		// 409: the request conflicts with the state of the resource (RFC 9110 section 15.5.10).
		writeError(w, conflict(fmt.Sprintf("the connection is %s: connect it before calling the provider", connection.Status)))
		return
	}
	scheme, found := s.connectors.Schemes[connection.AuthScheme]
	if !found {
		writeError(w, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme)))
		return
	}
	manifest, err := s.connectionManifest(ctx, connection, connection.DefinitionRevision)
	if err != nil {
		writeHandError(w, r, err)
		return
	}
	base, found := manifest.Endpoints[proxyBase]
	if !found {
		writeError(w, invalidRequest(fmt.Sprintf("connector %q has no %s endpoint, so it takes no direct calls", connection.ConnectorID, proxyBase)))
		return
	}
	target, err := proxyTarget(base, proxiedPath(r), r.URL.RawQuery)
	if err != nil {
		writeError(w, invalidRequest(err.Error()))
		return
	}
	limit := manifest.RateLimitKey(connection.CustomerID, coreConnection(connection))
	if wait := s.connectorLimiter.Wait(ctx, limit); wait > 0 {
		w.Header().Set("Retry-After", waitSeconds(wait))
		writeError(w, rateLimited(fmt.Sprintf("the provider asked to wait: retry after %s", waitSeconds(wait)+"s")))
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxProxyBody))
	if maxed := (*http.MaxBytesError)(nil); errors.As(err, &maxed) {
		writeError(w, payloadTooLarge(fmt.Sprintf("the body is over %d bytes", maxProxyBody)))
		return
	}
	if err != nil {
		writeError(w, invalidRequest("the body could not be read"))
		return
	}
	// All customers share the router's egress: one customer's calls to a connector are capped
	// so they cannot spend a provider's limit for everyone (AI-958).
	calls := core.CallsKey(connection.CustomerID, connection.ConnectorID)
	if wait := s.connectorLimiter.Take(ctx, calls, s.proxyCallsPerMinute, proxyCallWindow); wait > 0 {
		w.Header().Set("Retry-After", waitSeconds(wait))
		writeError(w, rateLimited(fmt.Sprintf("this app's direct calls to %s are over %d a minute: retry after %ss",
			connection.ConnectorID, s.proxyCallsPerMinute, waitSeconds(wait))))
		return
	}

	observed, exchange := core.WithExchange(ctx)
	// A bytes.Reader body gets GetBody, so the client can send it again after a 401.
	request, err := http.NewRequestWithContext(observed, r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		writeHandError(w, r, err)
		return
	}
	request.Header = forwardedHeader(r.Header)
	quotable := len(body) == 0 && request.URL.RawQuery == "" && onlyPlainHeaders(request.Header)
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	started := time.Now()
	response, err := s.connectorTransports.Client(ref, scheme).Do(request)
	latency := time.Since(started).Milliseconds()
	if err != nil {
		s.auditProxyCall(ctx, connection, target.Host, nil, latency)
		s.logger.Info("a direct call did not reach the provider", "connection", connection.ID, "error", err)
		writeError(w, unavailable("the call did not reach the provider, or its answer did not come back"))
		return
	}
	defer response.Body.Close()
	status := response.StatusCode
	s.auditProxyCall(ctx, connection, target.Host, &status, latency)
	// A 429 holds the connection's key until its Retry-After, as for a session's tool call
	// (session dispatcher.send), so the next call is refused here without being sent.
	if status == http.StatusTooManyRequests {
		s.connectorLimiter.Block(ctx, limit, exchange.RetryAfter())
	}
	for name, values := range response.Header {
		// The router's own request id stays: it is what the audit row names.
		if name == RequestIDHeader {
			continue
		}
		w.Header()[name] = values
	}
	// A link under api_base goes to this connection's proxy instead (proxiedLinks).
	w.Header().Del("Link")
	if links := proxiedLinks(response.Header.Values("Link"), base, s.connectionProxyURL(connection.ID)); len(links) > 0 {
		w.Header()["Link"] = links
	}
	for _, name := range hopHeaders {
		w.Header().Del(name)
	}
	// The answer is the provider's: withTiming adds neither its Server-Timing nor a duration field.
	leaveUntimed(ctx)
	w.WriteHeader(status)
	// The status is sent: a body cut off midway can only end the answer early.
	_, _ = io.Copy(w, response.Body)
	// A status other than 2xx is logged once the copy is done, so the caller's answer is neither
	// read early nor held for the line (AI-1245): the host the call went to and, for a call that
	// forwarded nothing that could be secret, the start of the body as the copy read it
	// (core.Exchange.ErrorBody, a 4xx's or a 5xx's), redacted as every connector call's is of the
	// path, the query and the connection's credential, with URLs kept to their host
	// (core.LoggedBody). It names no header.
	if status < http.StatusOK || status >= http.StatusMultipleChoices {
		line := []any{"connection", connection.ID, "connector", connection.ConnectorID,
			"provider_status", status, "provider_host", exchange.Host()}
		// Why: a provider may repeat the request in its error answer (an echo, a validation
		// error quoting a field), and a request body is never logged (security/go.md section 7),
		// nor a header that may carry a secret, and the query is forwarded too. Its echo can be any
		// part of it in any encoding (quoted, JSON-escaped), so no redaction could be shown to catch
		// it: a call that sent a body, a query, or a header outside plainHeaders, leaves no excerpt.
		if quotable {
			line = append(line, "provider_body", core.LoggedBody(exchange.ErrorBody()))
		}
		s.logger.Info("a direct call's provider answered with an error", line...)
	}
}

// connectionProxyURL is where the router at publicURL (ROUTER_PUBLIC_URL) takes direct calls
// for the connection id, with no slash at its end; empty when publicURL is.
func (s *Server) connectionProxyURL(id string) string {
	public := strings.TrimRight(s.publicURL, "/")
	if public == "" {
		return ""
	}
	return public + strings.TrimSuffix(strings.Replace(connectionProxyPath, "{id}", url.PathEscape(id), 1), "/")
}

// proxiedLinks is the Link header values (RFC 8288 section 3: link-values split by commas, each
// a URI-Reference in «<» and «>» and its parameters) with each target under base sent to proxy
// instead, at the same path under it and the same query. A provider pages by such links (GitHub
// REST: «link: <https://api.github.com/user/repos?page=2>; rel="next"», see github.yaml), and a
// client follows one as it is with the credential it has, which for a direct call is the
// router's: under base, the link would take that credential to the provider. A target
// elsewhere (another host, https://api.github.com.evil.example) is left as it is. A quoted
// parameter is skipped, so a «<» in it starts no target. With no proxy (no public URL) there is
// nowhere to send a link, so there are none.
//
// Example: base https://api.github.com, proxy https://router.example/v1/agents/connections/c1/proxy
// and <https://api.github.com/user/repos?page=2>; rel="next" is
// <https://router.example/v1/agents/connections/c1/proxy/user/repos?page=2>; rel="next".
func proxiedLinks(values []string, base, proxy string) []string {
	if proxy == "" {
		return nil
	}
	proxied := make([]string, 0, len(values))
	for _, value := range values {
		var out strings.Builder
		quoted, escaped := false, false
		for i := 0; i < len(value); i++ {
			c := value[i]
			switch {
			case escaped:
				escaped = false
			case quoted && c == '\\':
				escaped = true
			case c == '"':
				quoted = !quoted
			case !quoted && c == '<':
				end := strings.IndexByte(value[i:], '>')
				if end < 0 {
					break
				}
				target := value[i+1 : i+end]
				if rest, under := underBase(target, base); under {
					target = proxy + rest
				}
				out.WriteString("<" + target + ">")
				i += end
				continue
			}
			out.WriteByte(c)
		}
		proxied = append(proxied, out.String())
	}
	return proxied
}

// underBase is whether target is base or under it, and what follows base in it as written:
// the rest of the path, the query and the fragment. Under means base (scheme and host in any
// case) followed by nothing or by «/», «?» or «#», so a host or port that only starts with
// base's, or a userinfo naming it, is not under it (RFC 3986 section 3.2).
//
// Example: under https://api.github.com, https://api.github.com/user?page=2 is, with rest
// /user?page=2; https://api.github.com.evil.example/user, https://api.github.com@evil.example/
// and https://api.github.com:8443/user are not.
func underBase(target, base string) (string, bool) {
	base = strings.TrimSuffix(base, "/")
	if len(target) < len(base) || !strings.EqualFold(target[:len(base)], base) {
		return "", false
	}
	rest := target[len(base):]
	if rest != "" && !strings.ContainsRune("/?#", rune(rest[0])) {
		return "", false
	}
	return rest, true
}

// onlyPlainHeaders is whether every header of header, as the server read it (in canonical
// form), is one of plainHeaders.
//
// Example: Accept and User-Agent are; Accept and Cookie are not.
func onlyPlainHeaders(header http.Header) bool {
	for name := range header {
		if !slices.Contains(plainHeaders, name) {
			return false
		}
	}
	return true
}

// proxiedPath is the provider's path the request names, as it was escaped on the wire: what
// follows /v1/agents/connections/{id}/proxy/. The route matched, so the escaped path has the
// six segments before it.
func proxiedPath(r *http.Request) string {
	parts := strings.SplitN(r.URL.EscapedPath(), "/", 7)
	if len(parts) < 7 {
		return ""
	}
	return parts[6]
}

// proxyTarget is the provider URL a direct call goes to: base, then path, then query without
// the router's own parameters. path is kept as escaped, and follows base's own path after a
// slash, so it cannot name another host or a userinfo: «//evil.example» is a path on base's host.
// A dot segment, written or escaped, is refused: it removes part of the path where it is
// resolved (RFC 3986 sections 3.3 and 5.2.4), so it could leave base. So is a segment holding
// an escaped slash (%2F) or a backslash (%5C, or a raw one): the path is sent
// escaped, and a provider that decodes it before it resolves dot segments would read
// a/%2e%2e%2fx as a/../x (AI-958). The WHATWG URL parser ends a segment of an https URL at a
// backslash as at a slash («invalid-reverse-solidus», and takes %2e as a dot,
// https://url.spec.whatwg.org/#path-state, opened 2026-10-09).
//
// Example: base https://slack.com/api and path chat.postMessage is
// https://slack.com/api/chat.postMessage; paths ../oauth.v2.access, a%2F..%2Fx and a%5Cb are
// refused.
func proxyTarget(base, path, query string) (*url.URL, error) {
	for segment := range strings.SplitSeq(path, "/") {
		// A segment that does not unescape is refused by url.Parse below.
		decoded, _ := url.PathUnescape(segment)
		if decoded == "." || decoded == ".." {
			return nil, fmt.Errorf("the path %q has a dot segment, which would leave the connector's API", path)
		}
		if strings.ContainsAny(decoded, `/\`) {
			return nil, fmt.Errorf("the path %q has an escaped slash or a backslash, which a provider could read as a separator", path)
		}
	}
	// base is a rendered endpoint: an https URL with no query or fragment (core.Manifest.render).
	target, err := url.Parse(strings.TrimSuffix(base, "/") + "/" + path)
	if err != nil {
		return nil, fmt.Errorf("the path %q does not make a URL", path)
	}
	target.RawQuery = withoutRouterParams(query)
	return target, nil
}

// withoutRouterParams is query as it came, but for the router's own parameters, which may
// carry its credential. The others keep their order and spelling.
func withoutRouterParams(query string) string {
	if query == "" {
		return ""
	}
	kept := make([]string, 0, strings.Count(query, "&")+1)
	for pair := range strings.SplitSeq(query, "&") {
		name, _, _ := strings.Cut(pair, "=")
		if unescaped, err := url.QueryUnescape(name); err == nil && slices.Contains(routerParams, unescaped) {
			continue
		}
		kept = append(kept, pair)
	}
	return strings.Join(kept, "&")
}

// writeHandError answers a request served by hand with err: an APIError as it is, anything
// else as writeFailure does.
func writeHandError(w http.ResponseWriter, r *http.Request, err error) {
	var failure APIError
	if errors.As(err, &failure) {
		writeError(w, failure)
		return
	}
	writeFailure(w, r, err)
}

// forwardedHeader is the caller's header without the router's own, the hop-by-hop ones (those
// Connection names too) and the forwarding ones.
func forwardedHeader(from http.Header) http.Header {
	header := from.Clone()
	for _, listed := range from.Values("Connection") {
		for name := range strings.SplitSeq(listed, ",") {
			if name = textproto.TrimString(name); name != "" {
				header.Del(name)
			}
		}
	}
	for _, names := range [][]string{routerHeaders, hopHeaders, forwardingHeaders} {
		for _, name := range names {
			header.Del(name)
		}
	}
	return header
}

// waitSeconds is wait as Retry-After's delay-seconds (RFC 9110 section 10.2.3), rounded up so
// a caller that waits it is not refused again.
func waitSeconds(wait time.Duration) string {
	return strconv.Itoa(int(math.Ceil(wait.Seconds())))
}

// auditProxyCall records one direct call the connection's client sent (T44): the provider's
// status, nil when no answer came, the time to its answer and the host it reached
// (20261007170000_connector_invocations_and_audit.sql). The call happened whatever the row
// does, so a row that cannot be written is logged.
func (s *Server) auditProxyCall(ctx context.Context, connection store.ConnectorConnection, host string, status *int, latency int64) {
	ctx, cancel := detachedWrite(ctx)
	defer cancel()
	err := s.store.RecordConnectorAudit(ctx, &store.ConnectorAuditEvent{
		CustomerID: connection.CustomerID, ConnectionID: connection.ID, ConnectorID: connection.ConnectorID,
		OwnerType: connection.OwnerType, Action: store.AuditProxyCall, RequestID: core.CorrelationOf(ctx).RequestID,
		StatusCode: status, LatencyMs: &latency, Target: host,
	})
	if err != nil {
		s.logger.Error("could not record a connector audit row", "connection", connection.ID, "action", store.AuditProxyCall, "error", err)
	}
}
