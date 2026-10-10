//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/apikey"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Headers a test sends through the proxy to tell the echo how to answer. They reach it
// because the proxy forwards every header but the router's own.
const (
	echoStatusHeader     = "X-Echo-Status"
	echoRetryAfterHeader = "X-Echo-Retry-After"
	// echoRawHeader has the echo answer rawAnswer with rawServerTiming instead.
	echoRawHeader = "X-Echo-Raw"
)

// rawAnswer is a JSON object over 2 KiB, so net/http sends it chunked, with no Content-Length
// (bufferBeforeChunkingSize in net/http/server.go is 2048).
var rawAnswer = `{"ok":true,"pad":"` + strings.Repeat("x", 4096) + `"}`

const rawServerTiming = "provider;dur=7"

// echoed is what the echo provider received, as it answers it.
type echoed struct {
	Method string      `json:"method"`
	Path   string      `json:"path"`
	Query  string      `json:"query"`
	Header http.Header `json:"header"`
	Body   string      `json:"body"`
}

// ConnectionProxySuite sends direct calls through a connection (T44), end to end: the
// router's resolver and core.Transports over the suite's database, an echo provider that
// answers with what it received, and T5's fake provider for a credential the provider refuses
// and the router renews. Every connection's client reaches both on loopback.
type ConnectionProxySuite struct {
	RouterSuite
	provider *fakeprovider.Server
	echo     *httptest.Server
	// hits counts the requests the echo took.
	hits atomic.Int64
}

func TestConnectionProxySuite(t *testing.T) {
	runSuite(t, new(ConnectionProxySuite))
}

func (s *ConnectionProxySuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	s.echo = httptest.NewTLSServer(http.HandlerFunc(s.answer))
	s.T().Cleanup(s.echo.Close)
	clientCredentials, err := oauth2cc.New(oauth2cc.Config{HTTP: s.provider.Client(), PublicEndpoint: loopbackOrPublic})
	s.Require().NoError(err)
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{
		oauth2cc.Name: clientCredentials, bearer.Name: bearer.New(), apikey.Name: apikey.New()}}
	// One transport that trusts both servers' certificates.
	transport := s.provider.Client().Transport.(*http.Transport).Clone()
	roots := transport.TLSClientConfig.RootCAs.Clone()
	roots.AddCert(s.echo.Certificate())
	transport.TLSClientConfig.RootCAs = roots
	s.connectorHTTP = &http.Client{Transport: transport}
	// The router's default cap, as a deployment that sets none runs (AI-958). Every test but
	// the cap's own stays under it, so each asserts what base answered.
	s.proxyCallsPerMinute = config.Defaults().Connectors.ProxyCallsPerMinute
	s.RouterSuite.SetupSuite()
}

// TestCallsUpToTheDefaultCapAllReachTheProvider: the default cap's worth of calls on one
// connection in a row are all sent and all answered. Base 599298c6, which had no cap, answered
// 61 such calls with 61 200s and 61 hits (probe, <scratchpad>/pr-w8-proxy/probe-base.txt).
func (s *ConnectionProxySuite) TestCallsUpToTheDefaultCapAllReachTheProvider() {
	id := s.connected(s.connector(""), bearer.Name)
	before := s.hits.Load()

	ok := 0
	for range 60 {
		if status, _ := s.send(s.serverClient, http.MethodGet, proxy(id, "ping"), "", nil); status == http.StatusOK {
			ok++
		}
	}

	s.Equal(60, ok)
	s.Equal(before+60, s.hits.Load())
}

func (s *ConnectionProxySuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ConnectionProxySuite) TestOnlyTheAppsBackendMayCallAnAppOwnedConnection() {
	id := s.connected(s.connector(""), bearer.Name)

	// Two segments, which the spec's server-side pattern does not match.
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, proxy(id, "users/me"), nil, nil)
	})
}

func (s *ConnectionProxySuite) TestAnotherAppIsToldTheConnectionDoesNotExist() {
	id := s.connected(s.connector(""), bearer.Name)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, proxy(id, "ping"), nil, nil)
	})
}

// TestAUserConnectionIsReachedOnlyByABackendActingForThatUser: the owner's own backend call
// goes through; the backend acting for nobody or for another user is told it does not exist,
// and the owner's device is refused.
func (s *ConnectionProxySuite) TestAUserConnectionIsReachedOnlyByABackendActingForThatUser() {
	owner, other := s.data.createUser(), s.data.createUser()
	id := s.connectedFor(owner, s.connector(""))

	s.Equal(http.StatusOK, s.serverClient.actingFor(owner).do(http.MethodGet, proxy(id, "ping"), nil, nil))
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, proxy(id, "ping"), nil, nil))
	s.Equal(http.StatusNotFound, s.serverClient.actingFor(other).do(http.MethodGet, proxy(id, "ping"), nil, nil))
	s.Equal(http.StatusForbidden, owner.do(http.MethodGet, proxy(id, "ping"), nil, nil))
}

// TestTheProviderGetsTheRequestAsItCameWithTheConnectionsCredential: method, escaped path,
// query, headers and body arrive as sent, with the connection's key added and none of the
// router's own credentials or caller headers.
func (s *ConnectionProxySuite) TestTheProviderGetsTheRequestAsItCameWithTheConnectionsCredential() {
	key := "key-" + s.utils.uuid()
	id := s.connectedWithKey(s.connector(""), "X-Provider-Key", key)
	body := `{"text":"hello ` + s.utils.uuid() + `"}`

	status, answer := s.send(s.serverClient.actingFor(s.data.createUser()), http.MethodPatch,
		proxy(id, "v1/items/a%20b")+"?limit=5&api_key=router-key&token=router-token&cursor=c%20d&user_id=u",
		body, http.Header{"X-Custom": {"kept"}, "Connection": {"X-Hop"}, "X-Hop": {"dropped"},
			"X-Forwarded-For": {"192.0.2.1"}, "Proxy-Authorization": {"Basic cm91dGVyOmhvcA=="}})

	s.Require().Equal(http.StatusOK, status, string(answer))
	got := s.echoed(answer)
	s.Equal(http.MethodPatch, got.Method)
	s.Equal("/base/v1/items/a%20b", got.Path)
	s.Equal("limit=5&cursor=c%20d", got.Query)
	s.Equal(body, got.Body)
	s.True(got.Header.Get("X-Provider-Key") == key, "the provider got the connection's key")
	s.Equal("kept", got.Header.Get("X-Custom"))
	for _, name := range []string{"Authorization", auth.APIKeyHeader, auth.AuthTypeHeader, auth.UserHeader, "X-Hop", "X-Forwarded-For", "Proxy-Authorization"} {
		s.Empty(got.Header.Values(name), name)
	}
}

// TestNoneOfTheRoutersCredentialsOrCallerNamesReachTheProvider: every header and query
// parameter the router reads a credential or a caller from is stripped, a parameter name
// escaped on the wire as well, since the router reads names unescaped.
func (s *ConnectionProxySuite) TestNoneOfTheRoutersCredentialsOrCallerNamesReachTheProvider() {
	id := s.connected(s.connector(""), bearer.Name)
	router := []string{"Authorization", auth.APIKeyHeader, auth.AuthTypeHeader, auth.OrganizationHeader,
		auth.AppHeader, auth.CustomerHeader, auth.UserHeader, mintingKeyHeader}
	extra := http.Header{}
	for _, name := range router[3:] {
		extra.Set(name, "router-"+name)
	}

	status, answer := s.send(s.serverClient, http.MethodGet,
		proxy(id, "q")+"?keep=1&api%5Fkey=router-key&%74oken=router-token&customer_id=c&customer%5Fid=c&user%5Fid=u", "", extra)

	s.Require().Equal(http.StatusOK, status, string(answer))
	got := s.echoed(answer)
	s.Equal("keep=1", got.Query)
	for _, name := range router[1:] {
		s.Empty(got.Header.Values(name), name)
	}
	s.False(strings.HasPrefix(got.Header.Get("Authorization"), "Bearer ey"), "the router's token stayed with the router")
}

func (s *ConnectionProxySuite) TestABearerConnectionSendsItsTokenInsteadOfTheRouters() {
	token := "token-" + s.utils.uuid()
	id := s.connectedWithToken(s.connector(""), token)

	status, answer := s.send(s.serverClient, http.MethodGet, proxy(id, "me"), "", nil)

	s.Require().Equal(http.StatusOK, status, string(answer))
	s.True(s.echoed(answer).Header.Get("Authorization") == "Bearer "+token, "the provider got the connection's token")
}

// TestTheProvidersAnswerComesBackAsItCame: status, headers and body, but for the router's own
// request id, which the audit row names.
func (s *ConnectionProxySuite) TestTheProvidersAnswerComesBackAsItCame() {
	id := s.connected(s.connector(""), bearer.Name)

	response, answer := s.sendRaw(s.serverClient, http.MethodPost, proxy(id, "teapot"), "brew",
		http.Header{echoStatusHeader: {"418"}, RequestIDHeader: {"caller-" + s.utils.uuid()}})

	s.Equal(http.StatusTeapot, response.StatusCode)
	s.Equal("yes", response.Header.Get("X-Echo-Seen"))
	s.Empty(response.Header.Values("Keep-Alive"), "a hop-by-hop header stays with the hop")
	s.Equal("brew", s.echoed(answer).Body)
	s.True(strings.HasPrefix(response.Header.Get(RequestIDHeader), "caller-"), response.Header.Get(RequestIDHeader))
}

// TestAJSONAnswerComesBackByteForByte: a chunked JSON object, with no Content-Length, gets no
// duration field from the router, and the provider's Server-Timing is not replaced.
func (s *ConnectionProxySuite) TestAJSONAnswerComesBackByteForByte() {
	id := s.connected(s.connector(""), bearer.Name)

	response, answer := s.sendRaw(s.serverClient, http.MethodGet, proxy(id, "big"), "", http.Header{echoRawHeader: {"1"}})

	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{rawServerTiming}, response.Header.Values("Server-Timing"))
	s.Equal(rawAnswer, string(answer))
}

// TestAPathThatWouldLeaveTheAPIIsRefusedAndNothingIsSent: a dot segment, written or escaped,
// would take the path out from under api_base, and so would an escaped slash or a backslash
// at a provider that decodes the path before it resolves dots (AI-958).
func (s *ConnectionProxySuite) TestAPathThatWouldLeaveTheAPIIsRefusedAndNothingIsSent() {
	id := s.connected(s.connector(""), bearer.Name)
	before := s.hits.Load()

	for _, path := range []string{"../token", "%2e%2e/token", "a/./b", "a/%2E/b",
		"a/%2e%2e%2fx", "a%2F..%2F..%2Fx", "a/.%2e/x", "a/%2E%2e/x", "a%5C..%5Cx", "a%5cb", "%2F%2Fevil.example%2Fx"} {
		status, answer := s.send(s.serverClient, http.MethodGet, proxy(id, path), "", nil)
		s.Equal(http.StatusBadRequest, status, path+": "+string(answer))
	}
	s.Equal(before, s.hits.Load(), "nothing reached the provider")
}

// TestAPathCannotNameAnotherHost: whatever the path spells, it stays a path on api_base's host.
func (s *ConnectionProxySuite) TestAPathCannotNameAnotherHost() {
	id := s.connected(s.connector(""), bearer.Name)

	for path, arrived := range map[string]string{
		"//evil.example/x":       "/base///evil.example/x",
		"https://evil.example/x": "/base/https://evil.example/x",
		"@evil.example/x":        "/base/@evil.example/x",
	} {
		status, answer := s.send(s.serverClient, http.MethodGet, proxy(id, path), "", nil)
		s.Require().Equal(http.StatusOK, status, path+": "+string(answer))
		s.Equal(arrived, s.echoed(answer).Path, path)
	}
}

func (s *ConnectionProxySuite) TestAConnectorWithoutAnAPIBaseTakesNoDirectCalls() {
	id := s.connected(s.connectorWithout(), bearer.Name)

	status, message := s.serverClient.failure(http.MethodGet, proxy(id, "ping"), nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, "api_base")
}

func (s *ConnectionProxySuite) TestAConnectionNotConnectedYetIsAConflictAndNothingIsSent() {
	id := s.pending(s.connector(""), bearer.Name)
	before := s.hits.Load()

	status, message := s.serverClient.failure(http.MethodGet, proxy(id, "ping"), nil)

	s.Equal(http.StatusConflict, status)
	s.Contains(message, store.ConnectionPending)
	s.Equal(before, s.hits.Load())
}

func (s *ConnectionProxySuite) TestABodyOverTheCapIsRefusedAndNothingIsSent() {
	id := s.connected(s.connector(""), bearer.Name)
	before := s.hits.Load()

	status, _ := s.send(s.serverClient, http.MethodPost, proxy(id, "upload"), strings.Repeat("a", maxProxyBody+1), nil)

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Equal(before, s.hits.Load())
}

// TestAProvidersRateLimitPassesThroughAndHoldsTheConnectionsCalls: the provider's 429 and
// Retry-After come back as they came, and the next call is refused here, unsent, until then.
func (s *ConnectionProxySuite) TestAProvidersRateLimitPassesThroughAndHoldsTheConnectionsCalls() {
	id := s.connected(s.connector("rate_limit:\n  per: app\n"), bearer.Name)

	limited, answer := s.sendRaw(s.serverClient, http.MethodGet, proxy(id, "busy"), "",
		http.Header{echoStatusHeader: {"429"}, echoRetryAfterHeader: {"30"}})
	s.Require().Equal(http.StatusTooManyRequests, limited.StatusCode)
	s.Equal("30", limited.Header.Get("Retry-After"))
	s.Equal("/base/busy", s.echoed(answer).Path, "the provider's own body came back")
	before := s.hits.Load()

	held, body := s.sendRaw(s.serverClient, http.MethodGet, proxy(id, "busy"), "", nil)

	s.Equal(http.StatusTooManyRequests, held.StatusCode)
	wait, err := strconv.Atoi(held.Header.Get("Retry-After"))
	s.Require().NoError(err)
	s.True(wait > 0 && wait <= 30, wait)
	s.Contains(string(body), `"rate_limited"`)
	s.Equal(before, s.hits.Load(), "the held call was not sent")
}

// TestA429IsNotHeldForAConnectorThatNamesNoRateLimitScope: such a connector is never limited
// by the router (core.ResolvedManifest.RateLimitKey).
func (s *ConnectionProxySuite) TestA429IsNotHeldForAConnectorThatNamesNoRateLimitScope() {
	id := s.connected(s.connector(""), bearer.Name)
	s.send(s.serverClient, http.MethodGet, proxy(id, "busy"), "", http.Header{echoStatusHeader: {"429"}, echoRetryAfterHeader: {"30"}})
	before := s.hits.Load()

	status, _ := s.send(s.serverClient, http.MethodGet, proxy(id, "busy"), "", nil)

	s.Equal(http.StatusOK, status)
	s.Equal(before+1, s.hits.Load())
}

// TestARefusedCredentialIsRenewedAndTheBodySentOnceMore: the provider no longer takes the
// token the router holds (its clock moved past it), so the router mints a new one and sends
// the same request again, body and all.
func (s *ConnectionProxySuite) TestARefusedCredentialIsRenewedAndTheBodySentOnceMore() {
	id := s.connectedByClientCredentials()
	grants := s.provider.ClientCredentialsGrants()
	s.provider.Advance(2 * fakeprovider.AccessTTL)

	status, answer := s.send(s.serverClient, http.MethodPost, proxy(id, "mcp"), `{"jsonrpc":"2.0","id":7,"method":"ping"}`, nil)

	s.Require().Equal(http.StatusOK, status, string(answer))
	s.JSONEq(`{"jsonrpc":"2.0","id":7,"result":{}}`, string(answer))
	s.Equal(grants+1, s.provider.ClientCredentialsGrants(), "one new token was minted")
}

// TestEachCallSentLeavesOneAuditRow: its status, how long it took and the host, under the
// request's id; a call refused before it was sent leaves none.
func (s *ConnectionProxySuite) TestEachCallSentLeavesOneAuditRow() {
	id := s.connected(s.connector(""), bearer.Name)
	requestID := "proxy-" + s.utils.uuid()
	s.send(s.serverClient, http.MethodGet, proxy(id, "../refused"), "", nil)

	status, _ := s.send(s.serverClient, http.MethodGet, proxy(id, "created"), "", http.Header{
		echoStatusHeader: {"201"}, RequestIDHeader: {requestID}})

	s.Require().Equal(http.StatusCreated, status)
	calls := s.proxyCalls(id)
	s.Require().Len(calls, 1)
	echoURL, err := url.Parse(s.echo.URL)
	s.Require().NoError(err)
	s.Equal(requestID, calls[0].RequestID)
	s.Equal(echoURL.Host, calls[0].Target)
	s.Require().NotNil(calls[0].StatusCode)
	s.Equal(http.StatusCreated, *calls[0].StatusCode)
	s.Require().NotNil(calls[0].LatencyMs)
	s.GreaterOrEqual(*calls[0].LatencyMs, int64(0))
}

// TestACallThatGetsNoAnswerIsUnavailableAndLeavesOneRow: api_base is a closed port, so the
// call is sent and nothing answers. Its row has no status.
func (s *ConnectionProxySuite) TestACallThatGetsNoAnswerIsUnavailableAndLeavesOneRow() {
	closed := httptest.NewTLSServer(http.NotFoundHandler())
	closed.Close()
	connector := s.storeConnector("endpoints:\n  api_base: " + closed.URL + "/base\nschemes: [bearer]\n")
	id := s.connected(connector, bearer.Name)

	status, body := s.serverClient.call(http.MethodGet, proxy(id, "ping"), nil)

	s.Equal(http.StatusServiceUnavailable, status)
	s.Contains(string(body), `"type":"unavailable"`)
	calls := s.proxyCalls(id)
	s.Require().Len(calls, 1)
	s.Nil(calls[0].StatusCode)
	s.Equal(strings.TrimPrefix(closed.URL, "https://"), calls[0].Target)
}

// proxyCalls is the connection's proxy_call audit rows.
func (s *ConnectionProxySuite) proxyCalls(id string) []ConnectorAuditEvent {
	var page ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+id, nil, &page))
	var calls []ConnectorAuditEvent
	for _, row := range page.Items {
		if row.Action == ConnectorAuditAction(store.AuditProxyCall) {
			calls = append(calls, row)
		}
	}
	return calls
}

// answer is the echo provider: it answers with what it received, with the status and
// Retry-After the test asked for, or with rawAnswer when asked.
func (s *ConnectionProxySuite) answer(w http.ResponseWriter, r *http.Request) {
	s.hits.Add(1)
	if r.Header.Get(echoRawHeader) != "" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Server-Timing", rawServerTiming)
		_, _ = io.WriteString(w, rawAnswer)
		return
	}
	body, _ := io.ReadAll(r.Body)
	status := http.StatusOK
	if asked := r.Header.Get(echoStatusHeader); asked != "" {
		status, _ = strconv.Atoi(asked)
	}
	if wait := r.Header.Get(echoRetryAfterHeader); wait != "" {
		w.Header().Set("Retry-After", wait)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Echo-Seen", "yes")
	w.Header().Set("Keep-Alive", "timeout=5")
	w.Header().Set(RequestIDHeader, "provider-request")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(echoed{Method: r.Method, Path: r.URL.EscapedPath(), Query: r.URL.RawQuery,
		Header: r.Header, Body: string(body)})
}

// proxy is the proxy path of connection id for the provider's path.
func proxy(id, path string) string {
	return "/v1/agents/connections/" + id + "/proxy/" + path
}

// send is a request through as with body as it is and extra headers, its path sent as written.
func (s *ConnectionProxySuite) send(as *testClient, method, path, body string, extra http.Header) (int, []byte) {
	response, answer := s.sendRaw(as, method, path, body, extra)
	return response.StatusCode, answer
}

func (s *ConnectionProxySuite) sendRaw(as *testClient, method, path, body string, extra http.Header) (*http.Response, []byte) {
	request, err := http.NewRequest(method, s.server.URL+path, bytes.NewReader([]byte(body)))
	s.Require().NoError(err)
	request.Header = as.header.Clone()
	for name, values := range extra {
		request.Header[name] = values
	}
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answer, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response, answer
}

func (s *ConnectionProxySuite) echoed(answer []byte) echoed {
	var got echoed
	s.Require().NoError(json.Unmarshal(answer, &got), string(answer))
	return got
}

// connector stores a connector of the suite's app whose api_base is the echo's /base, taking
// bearer and api_key; more is more YAML at its top level.
func (s *ConnectionProxySuite) connector(more string) string {
	return s.storeConnector("endpoints:\n  api_base: " + s.echo.URL + "/base\nschemes: [bearer, api_key]\n" + more)
}

// connectorWithout stores a connector with no api_base.
func (s *ConnectionProxySuite) connectorWithout() string {
	return s.storeConnector("endpoints:\n  mcp: " + s.echo.URL + "/mcp\nschemes: [bearer]\n")
}

func (s *ConnectionProxySuite) storeConnector(body string) string {
	id := "custom_proxy" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	s.storeConnectorAs(id, body)
	return id
}

// storeConnectorAs stores a connector of id for the suite's current app.
func (s *ConnectionProxySuite) storeConnectorAs(id, body string) {
	manifest, err := core.ParseManifest([]byte("id: " + id + "\nrevision: 1\nname: Proxy\n" + body))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
}

// pending is an app-owned connection to connector with scheme and no credentials yet.
func (s *ConnectionProxySuite) pending(connector, scheme string) string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(connector), scheme), &created))
	return created.ID
}

// connected is an app-owned bearer connection given a synthetic token.
func (s *ConnectionProxySuite) connected(connector, scheme string) string {
	return s.connectedWithToken(connector, "token-"+s.utils.uuid())
}

func (s *ConnectionProxySuite) connectedWithToken(connector, token string) string {
	id := s.pending(connector, bearer.Name)
	s.credentials(s.serverClient, id, map[string]string{bearer.SuppliedToken: token})
	return id
}

func (s *ConnectionProxySuite) connectedWithKey(connector, header, key string) string {
	id := s.pending(connector, apikey.Name)
	s.credentials(s.serverClient, id, map[string]string{apikey.SuppliedKey: key, apikey.SuppliedHeader: header})
	return id
}

// connectedFor is a bearer connection of owner's, made by the backend acting for them.
func (s *ConnectionProxySuite) connectedFor(owner *testClient, connector string) string {
	acting := s.serverClient.actingFor(owner)
	var created Connection
	s.Require().Equal(http.StatusCreated, acting.do(http.MethodPost, "/v1/agents/connections",
		withScheme(userOwned(connector, owner), bearer.Name), &created))
	s.credentials(acting, created.ID, map[string]string{bearer.SuppliedToken: "token-" + s.utils.uuid()})
	return created.ID
}

// connectedByClientCredentials is a connection to a connector whose api_base and token
// endpoint are the fake's, holding a token the fake minted for its client.
func (s *ConnectionProxySuite) connectedByClientCredentials() string {
	connector := s.storeConnector("endpoints:\n  token: " + s.provider.URL + fakeprovider.PathToken +
		"\n  api_base: " + s.provider.URL + "\nschemes: [" + oauth2cc.Name + "]\n")
	id := s.pending(connector, oauth2cc.Name)
	s.credentials(s.serverClient, id, map[string]string{"client_id": s.provider.ClientID, "client_secret": s.provider.ClientSecret})
	return id
}

func (s *ConnectionProxySuite) credentials(as *testClient, id string, values map[string]string) {
	status, body := as.call(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": values})
	s.Require().Equal(http.StatusOK, status, string(body))
}

// TestACallOverTheCapIsRefusedUntilTheMinuteEndsAndNotSent: the default cap is 60 direct calls
// a minute for one customer's calls to one connector (AI-958). The 61st is a 429 with Retry-After in the
// APIError envelope, and the provider never sees the call.
func (s *ConnectionProxySuite) TestACallOverTheCapIsRefusedUntilTheMinuteEndsAndNotSent() {
	connector := s.connector("")
	id := s.connected(connector, bearer.Name)
	s.spendTheCap(id)
	before := s.hits.Load()

	refused, body := s.sendRaw(s.serverClient, http.MethodGet, proxy(id, "ping"), "", nil)

	s.Equal(http.StatusTooManyRequests, refused.StatusCode)
	wait, err := strconv.Atoi(refused.Header.Get("Retry-After"))
	s.Require().NoError(err)
	s.True(wait > 0 && wait <= 60, wait)
	s.Contains(string(body), "over 60 a minute")
	var failure struct {
		Error struct{ Type, Code, Message string } `json:"error"`
	}
	s.Require().NoError(json.Unmarshal(body, &failure), string(body))
	s.Equal(string(ErrorTypeRateLimited), failure.Error.Type)
	s.Contains(failure.Error.Message, connector)
	s.Equal(before, s.hits.Load(), "the refused call was not sent")
}

// TestTwoConnectionsToOneConnectorShareTheCap: the cap is the customer's for the connector,
// not one connection's, so a second connection does not double it.
func (s *ConnectionProxySuite) TestTwoConnectionsToOneConnectorShareTheCap() {
	connector := s.connector("")
	first, second := s.connected(connector, bearer.Name), s.connected(connector, bearer.Name)
	s.spendTheCap(first)

	status, _ := s.send(s.serverClient, http.MethodGet, proxy(second, "ping"), "", nil)

	s.Equal(http.StatusTooManyRequests, status)
}

func (s *ConnectionProxySuite) TestAnotherConnectorOfTheAppHasACapOfItsOwn() {
	s.spendTheCap(s.connected(s.connector(""), bearer.Name))

	status, _ := s.send(s.serverClient, http.MethodGet, proxy(s.connected(s.connector(""), bearer.Name), "ping"), "", nil)

	s.Equal(http.StatusOK, status)
}

// TestAnotherAppsCallsGoWhileOneAppIsOverItsCap: what the cap is for. One app spending its
// calls to a connector leaves another app's calls to a connector of the same id going.
func (s *ConnectionProxySuite) TestAnotherAppsCallsGoWhileOneAppIsOverItsCap() {
	connector := s.connector("")
	s.spendTheCap(s.connected(connector, bearer.Name))
	backend, id := s.connectedInAnotherApp(connector)

	status, answer := s.send(backend, http.MethodGet, proxy(id, "ping"), "", nil)

	s.Equal(http.StatusOK, status, string(answer))
}

// TestARefusedPathIsNotCounted: a call refused before it could be sent spends none of the cap.
func (s *ConnectionProxySuite) TestARefusedPathIsNotCounted() {
	id := s.connected(s.connector(""), bearer.Name)
	awayFromAMinuteBoundary()
	for range s.proxyCallsPerMinute {
		status, _ := s.send(s.serverClient, http.MethodGet, proxy(id, "a%2Fb"), "", nil)
		s.Require().Equal(http.StatusBadRequest, status)
	}

	status, _ := s.send(s.serverClient, http.MethodGet, proxy(id, "ping"), "", nil)

	s.Equal(http.StatusOK, status)
}

// spendTheCap sends the default cap's calls on connection id, all answered, in one minute.
func (s *ConnectionProxySuite) spendTheCap(id string) {
	awayFromAMinuteBoundary()
	for range s.proxyCallsPerMinute {
		status, answer := s.send(s.serverClient, http.MethodGet, proxy(id, "ping"), "", nil)
		s.Require().Equal(http.StatusOK, status, string(answer))
	}
}

// connectedInAnotherApp is a backend of a new app and its app-owned connection to that app's
// own connector of id, at the echo, as the built-ins share one id across apps.
func (s *ConnectionProxySuite) connectedInAnotherApp(id string) (*testClient, string) {
	mine := s.app
	defer func() { s.app = mine }()
	s.app = s.data.createApp()
	backend := s.signedIn(jwt.MapClaims{"server": true}, auth.AuthTypeServer, server, "")
	s.storeConnectorAs(id, "endpoints:\n  api_base: "+s.echo.URL+"/base\nschemes: [bearer]\n")
	var created Connection
	s.Require().Equal(http.StatusCreated, backend.do(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(id), bearer.Name), &created))
	s.credentials(backend, created.ID, map[string]string{bearer.SuppliedToken: "token-" + s.utils.uuid()})
	return backend, created.ID
}

// awayFromAMinuteBoundary waits out a minute that ends in under five seconds, so the calls a
// test counts next fall in one of the cap's windows (core.Limiter.Take).
func awayFromAMinuteBoundary() {
	if left := time.Until(time.Now().Truncate(time.Minute).Add(time.Minute)); left < 5*time.Second {
		time.Sleep(left)
	}
}

// ConnectionProxyOffSuite is the router with connectors off, as staging runs: no transports,
// no limiter (cmd/router newConnectorTransports, newConnectorLimiter).
type ConnectionProxyOffSuite struct {
	RouterSuite
}

func TestConnectionProxyOffSuite(t *testing.T) {
	suite := new(ConnectionProxyOffSuite)
	suite.connectorsOff = true
	runSuite(t, suite)
}

func (s *ConnectionProxyOffSuite) SetupTest() {
	s.useFixture("standard")
}

// TestTheConnectionEndpointsAnswerAsOnBase: the answers base ad3fffd0 gave with connectors off
// (probe, <scratchpad>/pr-w3d-t44/probe-base.txt): a missing connection is connection_not_found
// and a list without owner_type a validation failure.
func (s *ConnectionProxyOffSuite) TestTheConnectionEndpointsAnswerAsOnBase() {
	status, body := s.serverClient.call(http.MethodGet, "/v1/agents/connections/x", nil)
	s.Equal(http.StatusNotFound, status)
	s.JSONEq(`{"error":{"message":"no such connection","type":"not_found","code":"connection_not_found","doc_url":"https://getstream.io/agents/docs/api/errors/#connection_not_found"}}`, withoutDuration(body))

	status, body = s.serverClient.call(http.MethodGet, "/v1/agents/connections", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"validation_failed"`)
}

// TestTheProxyIsNoRouteAsOnBase: base fd4b4405 with connectors off answered every method on
// a proxy path with 404 "no such route", for a backend and a device, a missing and a stored
// connection, one segment or two (probe, <scratchpad>/pr-w3d-t44/fx-probe-base.txt).
func (s *ConnectionProxyOffSuite) TestTheProxyIsNoRouteAsOnBase() {
	ctx := context.Background()
	connectorID := "custom_proxy" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte("id: " + connectorID + "\nrevision: 1\nname: Proxy\n" +
		"endpoints:\n  api_base: https://api.example.com\nschemes: [bearer]\n"))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(ctx, s.customerID(), manifest)
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{CustomerID: s.customerID(), ConnectorID: connectorID,
		OwnerType: store.OwnerApp, AuthScheme: bearer.Name, DefinitionRevision: 1}
	registry := core.Registry{Schemes: map[string]core.Scheme{bearer.Name: bearer.New()}}
	s.Require().NoError(s.store.CreateConnectorConnection(ctx, registry, connection))
	device := s.data.createUser()

	for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut, http.MethodPatch,
		http.MethodDelete, http.MethodOptions, http.MethodTrace} {
		for _, id := range []string{"x", connection.ID} {
			for _, as := range []*testClient{s.serverClient, device} {
				for _, path := range []string{"auth.test", "repos/octo/hello"} {
					status, body := as.call(method, proxy(id, path), nil)
					s.Equal(http.StatusNotFound, status, method+" "+path)
					if method != http.MethodHead {
						s.JSONEq(`{"error":{"message":"no such route","type":"not_found","code":"not_found","doc_url":"https://getstream.io/agents/docs/api/errors/#not_found"}}`, withoutDuration(body), method+" "+path)
					}
				}
			}
		}
	}
}
