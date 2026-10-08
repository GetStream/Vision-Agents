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

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
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
)

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
	s.RouterSuite.SetupSuite()
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
		proxy(id, "v1/items/a%2Fb")+"?limit=5&api_key=router-key&token=router-token&cursor=c%20d&user_id=u",
		body, http.Header{"X-Custom": {"kept"}, "Connection": {"X-Hop"}, "X-Hop": {"dropped"},
			"X-Forwarded-For": {"192.0.2.1"}, "Proxy-Authorization": {"Basic cm91dGVyOmhvcA=="}})

	s.Require().Equal(http.StatusOK, status, string(answer))
	got := s.echoed(answer)
	s.Equal(http.MethodPatch, got.Method)
	s.Equal("/base/v1/items/a%2Fb", got.Path)
	s.Equal("limit=5&cursor=c%20d", got.Query)
	s.Equal(body, got.Body)
	s.True(got.Header.Get("X-Provider-Key") == key, "the provider got the connection's key")
	s.Equal("kept", got.Header.Get("X-Custom"))
	for _, name := range []string{"Authorization", auth.APIKeyHeader, auth.AuthTypeHeader, auth.UserHeader, "X-Hop", "X-Forwarded-For", "Proxy-Authorization"} {
		s.Empty(got.Header.Values(name), name)
	}
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

// TestAPathThatWouldLeaveTheAPIIsRefusedAndNothingIsSent: a dot segment, written or escaped,
// would take the path out from under api_base.
func (s *ConnectionProxySuite) TestAPathThatWouldLeaveTheAPIIsRefusedAndNothingIsSent() {
	id := s.connected(s.connector(""), bearer.Name)
	before := s.hits.Load()

	for _, path := range []string{"../token", "%2e%2e/token", "a/./b", "a/%2E/b"} {
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
		"%2F%2Fevil.example%2Fx": "/base/%2F%2Fevil.example%2Fx",
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
	var page ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+id, nil, &page))
	var calls []ConnectorAuditEvent
	for _, row := range page.Items {
		if row.Action == ConnectorAuditAction(store.AuditProxyCall) {
			calls = append(calls, row)
		}
	}
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

// answer is the echo provider: it answers with what it received, with the status and
// Retry-After the test asked for.
func (s *ConnectionProxySuite) answer(w http.ResponseWriter, r *http.Request) {
	s.hits.Add(1)
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
	manifest, err := core.ParseManifest([]byte("id: " + id + "\nrevision: 1\nname: Proxy\n" + body))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
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

// TestAMissingConnectionIsNotFoundAsGetConnectionAnswers: the same answer getConnection gives.
func (s *ConnectionProxyOffSuite) TestAMissingConnectionIsNotFoundAsGetConnectionAnswers() {
	_, read := s.serverClient.call(http.MethodGet, "/v1/agents/connections/x", nil)

	status, body := s.serverClient.call(http.MethodPost, proxy("x", "chat.postMessage"), nil)

	s.Equal(http.StatusNotFound, status)
	s.JSONEq(withoutDuration(read), withoutDuration(body))
}

// TestAConnectionLeftFromBeforeIsNotCalled: a stored connection is refused as not
// configured, before anything could be sent.
func (s *ConnectionProxyOffSuite) TestAConnectionLeftFromBeforeIsNotCalled() {
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

	status, body := s.serverClient.call(http.MethodGet, proxy(connection.ID, "auth.test"), nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"not_configured"`)
}
