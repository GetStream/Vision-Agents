//go:build integration

package api

import (
	"crypto/rand"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/golang-jwt/jwt/v5"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// appLogSuite is a router whose log a test reads back.
type appLogSuite struct {
	RouterSuite

	logged *lockedLog
}

func (s *appLogSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

// LogAppSuite reads what the router logs for a request and finds the app and public key on
// each line of an authenticated one.
type LogAppSuite struct {
	appLogSuite
}

func TestLogAppSuite(t *testing.T) {
	runSuite(t, new(LogAppSuite))
}

// testRelease is the version LogAppSuite's router says it was built from.
const testRelease = "v0.0.0-logtest"

// loggedCap is how many bytes of a value the caller chose the access line keeps, as
// acceleration/README.md documents it. Not maxLoggedValue, so a changed cap fails here.
const loggedCap = 256

func (s *LogAppSuite) SetupSuite() {
	s.release = testRelease
	s.appLogSuite.SetupSuite()
}

func (s *LogAppSuite) SetupTest() {
	s.useFixture("standard")
}

// Defect: the path logged as the operation, which spreads one route over one value per id
// (and a token route's over its secrets), or the route left off.
func (s *LogAppSuite) TestTheAccessLineNamesTheRouteItServedRatherThanThePath() {
	id := s.utils.uuid()

	line := s.sent(s.serverClient, http.MethodGet, "/v1/agents/sessions/"+id, nil)

	s.Equal("/v1/agents/sessions/{id}", s.field(line, "operation_type"))
	s.Equal(1, strings.Count(line, "operation_type="))
}

// Defect: a refusal before the route (withServerSide) logged without its route.
func (s *LogAppSuite) TestTheAccessLineOfARequestRefusedBeforeItsRouteNamesTheRoute() {
	line := s.sent(s.client, http.MethodGet, "/v1/usage/events", nil)

	s.Contains(line, "status=403")
	s.Equal("/v1/usage/events", s.field(line, "operation_type"))
}

// Defect: "unmatched", the metrics' bucket, logged as if it were a route.
func (s *LogAppSuite) TestTheAccessLineOfAPathNoRouteServesNamesNoOperation() {
	line := s.sent(s.serverClient, http.MethodGet, "/v1/no-such-route/"+s.utils.uuid(), nil)

	s.Contains(line, "status=404")
	s.NotContains(line, "operation_type")
}

// Defect: the code left off a refusal, or another code than the one the caller was told.
func (s *LogAppSuite) TestTheAccessLineOfARefusalNamesTheCodeTheCallerWasTold() {
	for _, refused := range []struct {
		client *testClient
		path   string
	}{
		{s.unauthenticatedClient, "/v1/account"},
		{s.client, "/v1/usage/events"},
	} {
		id := s.utils.uuid()
		status, payload := refused.client.with(RequestIDHeader, id).call(http.MethodGet, refused.path, nil)
		s.Require().GreaterOrEqual(status, http.StatusBadRequest, refused.path)

		s.Equal(s.errorCode(payload), s.field(s.lineOf(id), "error_code"), refused.path)
	}
}

func (s *LogAppSuite) TestTheAccessLineOfASuccessNamesNoErrorCode() {
	line := s.sent(s.serverClient, http.MethodGet, "/v1/agents/skills", nil)

	s.Contains(line, "status=200")
	s.NotContains(line, "error_code")
}

// Defect: a header logged under another name, or not at all.
func (s *LogAppSuite) TestTheAccessLineNamesTheClientItsAuthTypeAndItsUserAgent() {
	caller := s.serverClient.with(clientHeader, "stream-cli 1.2").with("User-Agent", "probe agent/1.0")

	line := s.servedAs(caller)

	s.Equal("stream-cli 1.2", s.field(line, "x_stream_client"))
	s.Equal(auth.AuthTypeServer, s.field(line, "stream_auth_type"))
	s.Equal("probe agent/1.0", s.field(line, "user_agent"))
}

// Defect: an absent header logged empty.
func (s *LogAppSuite) TestTheAccessLineOfARequestWithoutThoseHeadersNamesNone() {
	caller := s.unauthenticatedClient.with(clientHeader, "")
	// An empty User-Agent is how net/http is told to send none rather than its own.
	caller.header["User-Agent"] = []string{""}

	line := s.servedAs(caller)

	s.NotContains(line, "x_stream_client")
	s.NotContains(line, "stream_auth_type")
	s.NotContains(line, "user_agent")
}

// Defect: a header the caller chose logged whole, so one request can write a megabyte line;
// or cut inside a character, which the log then shows as bytes.
func (s *LogAppSuite) TestAHeaderTheCallerChoseIsCutOnTheAccessLine() {
	long := strings.Repeat("a", loggedCap+44)
	// A two-byte character straddling the cap, which is cut before rather than in half.
	straddling := strings.Repeat("b", loggedCap-1) + "éé"

	line := s.servedAs(s.serverClient.with(clientHeader, long).with("User-Agent", straddling))

	s.Equal(strings.Repeat("a", loggedCap), s.field(line, "x_stream_client"))
	s.Equal(strings.Repeat("b", loggedCap-1), s.field(line, "user_agent"))
}

// Defect: the body's size left off, or the response's logged as it.
func (s *LogAppSuite) TestTheAccessLineNamesTheSizeOfTheBodyItWasSent() {
	body := `{"probe":"` + s.utils.uuid() + `"}`

	line := s.sent(s.serverClient, http.MethodPost, "/v1/agents/sessions", strings.NewReader(body))

	s.Equal(strconv.Itoa(len(body)), s.field(line, "request_size"))
}

// Defect: a body of unknown length logged as -1, or a request with no body as 0.
func (s *LogAppSuite) TestTheAccessLineOfABodyOfUnknownLengthOrNoneNamesNoSize() {
	// A reader whose length net/http cannot know, which it sends chunked.
	chunked := io.MultiReader(strings.NewReader(`{"probe":"` + s.utils.uuid() + `"}`))

	sentChunked := s.sent(s.serverClient, http.MethodPost, "/v1/agents/sessions", chunked)
	sentNone := s.servedAs(s.serverClient)

	s.NotContains(sentChunked, "request_size")
	s.NotContains(sentNone, "request_size")
}

// Defect: the user of an end user's request left off.
func (s *LogAppSuite) TestTheAccessLineOfAnEndUserNamesTheUser() {
	s.Require().NotEmpty(s.client.userID)
	s.Require().NotEmpty(s.anonymousClient.userID)

	s.Equal(s.client.userID, s.field(s.servedAs(s.client), "user_id"))
	s.Equal(s.anonymousClient.userID, s.field(s.servedAs(s.anonymousClient), "user_id"))
}

// Defect: a backend's request logged as the user it says it acts for, which it made for itself.
func (s *LogAppSuite) TestTheAccessLineOfABackendNamesNoUserEvenActingForOne() {
	s.NotContains(s.servedAs(s.serverClient), "user_id")
	s.NotContains(s.servedAs(s.serverClient.actingFor(s.client)), "user_id")
	s.NotContains(s.servedAs(s.unauthenticatedClient), "user_id")
}

// Defect: a user id the caller chose logged whole.
func (s *LogAppSuite) TestAUserIDTheCallerChoseIsCutOnTheAccessLine() {
	anonymous := s.data.claiming(strings.Repeat("u", loggedCap+10))

	s.Equal(strings.Repeat("u", loggedCap), s.field(s.servedAs(anonymous), "user_id"))
}

// Defect: an end user who named nobody logged with an empty user id.
func (s *LogAppSuite) TestTheAccessLineOfAnEndUserWhoNamedNobodyNamesNoUser() {
	nameless := s.signedIn(jwt.MapClaims{}, auth.AuthTypeAnonymous, anonymous, "")

	line := s.servedAs(nameless)

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.NotContains(line, "user_id")
}

func (s *LogAppSuite) TestTheAccessLineNamesTheReleaseTheRouterWasBuiltFrom() {
	line := s.servedAs(s.unauthenticatedClient)

	s.Equal(testRelease, s.field(line, "build_version"))
}

// Defect: the trace left off, or another id than the request's span's.
func (s *LogAppSuite) TestTheAccessLineNamesTheTraceOfTheRequestsSpan() {
	var traceID trace.TraceID
	var spanID trace.SpanID
	_, _ = rand.Read(traceID[:])
	_, _ = rand.Read(spanID[:])
	span := trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID, TraceFlags: trace.FlagsSampled})

	line := s.servedWithin(func(r *http.Request) *http.Request {
		return r.WithContext(trace.ContextWithSpanContext(r.Context(), span))
	})

	s.Equal(traceID.String(), s.field(line, "trace_id"))
}

// Defect: a request with tracing off logged with an empty or all-zero trace id.
func (s *LogAppSuite) TestTheAccessLineOfARequestWithNoSpanNamesNoTrace() {
	line := s.servedWithin(func(r *http.Request) *http.Request { return r })

	s.NotContains(line, "trace_id")
}

// Defect: the sampler dropped the span, and the line named a trace that was never exported.
func (s *LogAppSuite) TestTheAccessLineOfAnUnsampledRequestNamesNoTrace() {
	var traceID trace.TraceID
	var spanID trace.SpanID
	traceID[0], spanID[0] = 1, 1
	unsampled := trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID})
	s.Require().True(unsampled.IsValid())

	line := s.servedWithin(func(r *http.Request) *http.Request {
		return r.WithContext(trace.ContextWithSpanContext(r.Context(), unsampled))
	})

	s.NotContains(line, "trace_id")
}

// servedWithin is the access line withRequestLog writes for a GET whose request was prepared
// by within first, as withTrace prepares it with the request's span.
func (s *LogAppSuite) servedWithin(within func(*http.Request) *http.Request) string {
	path := "/probe-" + s.utils.uuid()
	routes := chi.NewRouter()
	routes.Get(path, func(http.ResponseWriter, *http.Request) {})
	logged := s.router.withRequestLog(routes, routes)
	node := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		logged.ServeHTTP(w, within(r))
	}))
	s.T().Cleanup(node.Close)

	response, err := node.Client().Get(node.URL + path)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	var line string
	s.Require().Eventually(func() bool {
		for l := range strings.Lines(s.logged.String()) {
			if strings.Contains(l, `msg="served a request"`) && strings.Contains(l, " path="+path+" ") {
				line = l
				return true
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond)
	return line
}

// errorCode is the code of the failure payload answers with.
func (s *LogAppSuite) errorCode(payload []byte) string {
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	s.Require().NotEmpty(answered.Error.Code, string(payload))
	return answered.Error.Code
}

func (s *LogAppSuite) TestTheAccessLineOfABackendNamesTheAppAndItsPublicKey() {
	line := s.servedAs(s.serverClient)

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
	s.Equal(1, strings.Count(line, "app_id="))
	s.Equal(1, strings.Count(line, "api_key="))
}

func (s *LogAppSuite) TestTheAccessLineOfAnEndUserNamesTheAppAndItsPublicKey() {
	line := s.servedAs(s.client)

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

func (s *LogAppSuite) TestNoLineNamesTheKeysSecret() {
	s.servedAs(s.serverClient)
	s.servedAs(s.client)

	s.NotContains(s.logged.String(), s.app.secret)
}

func (s *LogAppSuite) TestTheAccessLineStillNamesTheCustomer() {
	line := s.servedAs(s.serverClient)

	s.Contains(line, " customer="+s.customerID()+" ")
}

func (s *LogAppSuite) TestTheAccessLineOfARequestNobodyAuthenticatedNamesNeither() {
	line := s.servedAs(s.unauthenticatedClient)

	s.Contains(line, "status=401")
	s.NotContains(line, "app_id")
	s.NotContains(line, "api_key")
}

func (s *LogAppSuite) TestTheAccessLineOfAKeyWithABadTokenNamesNeither() {
	forged := &testClient{suite: &s.RouterSuite, header: http.Header{}}
	forged.header.Set(auth.APIKeyHeader, s.app.key)
	forged.header.Set(auth.CustomerHeader, s.customerID())
	forged.header.Set("Authorization", "Bearer not-a-token")

	line := s.servedAs(forged)

	s.Contains(line, "status=401")
	s.NotContains(line, "app_id")
	s.NotContains(line, "api_key")
}

func (s *LogAppSuite) TestTheAccessLineOfAHookNamesNeitherEvenWhenTheCallerSendsAnApp() {
	hostile := *s.serverClient
	hostile.header = s.serverClient.header.Clone()
	hostile.header.Set(auth.CustomerHeader, s.customerID())
	id := s.utils.uuid()
	hostile.header.Set(RequestIDHeader, id)

	hostile.call(http.MethodPost, "/v1/phone/hooks/stream", map[string]any{})

	line := s.lineOf(id)
	s.Contains(line, "path=/v1/phone/hooks/stream")
	s.NotContains(line, "app_id")
	s.NotContains(line, "api_key")
}

func (s *LogAppSuite) TestALineAHandlerLogsWithTheRequestsContextNamesTheAppAndItsPublicKey() {
	handler := s.router.withCustomer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		s.router.logger.InfoContext(r.Context(), "probe from a handler")
		s.router.logger.Info("probe without the context")
	}))
	node := httptest.NewServer(handler)
	s.T().Cleanup(node.Close)
	request, err := http.NewRequest(http.MethodGet, node.URL+"/v1/agents/sessions", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()

	response, err := node.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), "probe without the context") },
		5*time.Second, 20*time.Millisecond)
	var with, without string
	for line := range strings.Lines(s.logged.String()) {
		switch {
		case strings.Contains(line, "probe from a handler"):
			with = line
		case strings.Contains(line, "probe without the context"):
			without = line
		}
	}
	s.Contains(with, " app_id="+s.customerID())
	s.Contains(with, " api_key="+s.app.key)
	s.NotContains(without, "app_id")
}

func (s *LogAppSuite) TestThePanicLineNamesTheAppAndItsPublicKey() {
	routes := chi.NewRouter()
	routes.Get("/probe", func(http.ResponseWriter, *http.Request) { panic("probe panic") })
	node := httptest.NewUnstartedServer(s.router.withCustomer(s.router.withRequestLog(routes, routes)))
	node.Config.ErrorLog = log.New(io.Discard, "", 0)
	node.Start()
	s.T().Cleanup(node.Close)
	request, err := http.NewRequest(http.MethodGet, node.URL+"/probe", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()

	// The panic goes on to net/http after the 500, which drops the connection.
	if response, err := node.Client().Do(request); err == nil {
		s.Require().NoError(response.Body.Close())
	}

	var line string
	s.Require().Eventually(func() bool {
		for l := range strings.Lines(s.logged.String()) {
			if strings.Contains(l, `msg="a request panicked"`) && strings.Contains(l, " path=/probe ") {
				line = l
				return true
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond)
	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

func (s *LogAppSuite) TestTheAccessLineOfAServerErrorNamesTheAppAndItsPublicKey() {
	routes := chi.NewRouter()
	routes.Get("/probe-500", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusInternalServerError) })
	node := httptest.NewServer(s.router.withCustomer(s.router.withRequestLog(routes, routes)))
	s.T().Cleanup(node.Close)
	request, err := http.NewRequest(http.MethodGet, node.URL+"/probe-500", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()

	response, err := node.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	var line string
	s.Require().Eventually(func() bool {
		for l := range strings.Lines(s.logged.String()) {
			if strings.Contains(l, `msg="served a request"`) && strings.Contains(l, " path=/probe-500 ") {
				line = l
				return true
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond)
	s.Contains(line, "level=ERROR")
	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

// servedAs is the access line of a GET by client.
func (s *appLogSuite) servedAs(client *testClient) string {
	id := s.utils.uuid()
	named := *client
	named.header = client.header.Clone()
	named.header.Set(RequestIDHeader, id)
	named.call(http.MethodGet, "/v1/account", nil)
	return s.lineOf(id)
}

// sent is the access line of a request by client sending body as is, whatever it answered.
func (s *appLogSuite) sent(client *testClient, method, path string, body io.Reader) string {
	id := s.utils.uuid()
	request, err := http.NewRequest(method, s.server.URL+path, body)
	s.Require().NoError(err)
	request.Header = client.header.Clone()
	request.Header.Set(RequestIDHeader, id)
	request.Header.Set("Content-Type", "application/json")
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return s.lineOf(id)
}

// field is the value line gives key, unquoted, or "" when it gives none.
func (s *appLogSuite) field(line, key string) string {
	at := strings.Index(line, " "+key+"=")
	if at < 0 {
		return ""
	}
	rest := strings.TrimSuffix(line[at+len(key)+2:], "\n")
	if strings.HasPrefix(rest, `"`) {
		quoted, err := strconv.QuotedPrefix(rest)
		s.Require().NoError(err, line)
		value, err := strconv.Unquote(quoted)
		s.Require().NoError(err, line)
		return value
	}
	if end := strings.IndexByte(rest, ' '); end >= 0 {
		return rest[:end]
	}
	return rest
}

// lineOf is the access line of the request that carried id.
func (s *appLogSuite) lineOf(id string) string {
	var found string
	s.Require().Eventually(func() bool {
		for line := range strings.Lines(s.logged.String()) {
			if strings.Contains(line, `msg="served a request"`) && strings.Contains(line, " request_id="+id) {
				found = line
				return true
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond, "the access line of request %s", id)
	return found
}

// LogAppProxiedSuite is the same behind the proxy, which names the key the caller used.
type LogAppProxiedSuite struct {
	appLogSuite
}

func TestLogAppProxiedSuite(t *testing.T) {
	proxied := new(LogAppProxiedSuite)
	proxied.proxied = true
	runSuite(t, proxied)
}

func (s *LogAppProxiedSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *LogAppProxiedSuite) TestTheAccessLineNamesTheKeyTheProxySaidTheCallerUsed() {
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)
	header.Set(auth.APIKeyIDHeader, s.app.key)

	line := s.servedAs(&testClient{suite: &s.RouterSuite, header: header})

	s.Contains(line, " app_id="+s.customerID()+" ")
	s.Contains(line, " api_key="+s.app.key)
}

// The control: a router built without a release (a local build) names none.
func (s *LogAppProxiedSuite) TestTheAccessLineOfARouterWithNoReleaseNamesNone() {
	line := s.servedAs(s.unauthenticatedClient)

	s.NotContains(line, "build_version")
}

// The proxy says the caller's user in a header, which is the user the line names.
func (s *LogAppProxiedSuite) TestTheAccessLineNamesTheUserTheProxySaidTheCallerIs() {
	user := s.utils.uuid()
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)
	header.Set(auth.UserHeader, user)

	line := s.servedAs(&testClient{suite: &s.RouterSuite, header: header})

	s.Equal(user, s.field(line, "user_id"))
	s.Equal(auth.AuthTypeJWT, s.field(line, "stream_auth_type"))
}

func (s *LogAppProxiedSuite) TestTheAccessLineOfACallerTheProxyNamedNoKeyForNamesNoKey() {
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)

	line := s.servedAs(&testClient{suite: &s.RouterSuite, header: header})

	s.Contains(line, " app_id="+s.customerID())
	s.NotContains(line, "api_key")
}
