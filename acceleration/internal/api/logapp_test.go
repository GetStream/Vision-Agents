//go:build integration

package api

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/golang-jwt/jwt/v5"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
// (and a token route's over its secrets), or the route pattern logged where the operation's
// name is wanted.
func (s *LogAppSuite) TestTheAccessLineNamesTheOperationItServedRatherThanThePath() {
	id := s.utils.uuid()

	line := s.sent(s.serverClient, http.MethodGet, "/v1/agents/sessions/"+id, nil)

	s.Equal("GetSession", s.field(line, "operation_type"))
	s.Equal(1, strings.Count(line, "operation_type="))
	s.Equal("/v1/agents/sessions/"+id, s.field(line, "path"))
}

// Defect: a refusal before the route (withServerSide) logged without its operation.
func (s *LogAppSuite) TestTheAccessLineOfARequestRefusedBeforeItsRouteNamesTheOperation() {
	line := s.sent(s.client, http.MethodGet, "/v1/usage/events", nil)

	s.Contains(line, "status=403")
	s.Equal("ListUsageEvents", s.field(line, "operation_type"))
}

// Defect: a route served by hand logged with its pattern, or with none, as it is no huma.Register.
func (s *LogAppSuite) TestTheAccessLineOfASocketNamesTheOperationItDeclares() {
	events := s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/agents/sessions/"+s.utils.uuid()+"/events", nil)
	socket := s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/agents/socket", nil)

	s.Equal("WatchSession", s.field(events, "operation_type"))
	s.Equal("OpenSocketSession", s.field(socket, "operation_type"))
}

// Defect: the operation looked up by the path as sent, so a token route logged its secret.
func (s *LogAppSuite) TestTheAccessLineOfATokenRouteNamesTheOperationAndNoSecret() {
	token := s.utils.uuid()

	line := s.sent(s.unauthenticatedClient, http.MethodPost, "/v1/phone/status/"+token, nil)

	s.Equal("ReceiveCallStatus", s.field(line, "operation_type"))
	s.NotContains(line, token)
}

// Defect: "unmatched", the metrics' bucket, or a route pattern, logged as if it were an operation.
func (s *LogAppSuite) TestTheAccessLineOfARequestNoOperationServedNamesNoOperation() {
	missing := s.sent(s.serverClient, http.MethodGet, "/v1/no-such-route/"+s.utils.uuid(), nil)
	wrongMethod := s.sent(s.serverClient, http.MethodDelete, "/v1/account", nil)

	s.Contains(missing, "status=404")
	s.NotContains(missing, "operation_type")
	s.Contains(wrongMethod, "status=405")
	s.NotContains(wrongMethod, "operation_type")
}

// Defect: a route registered without its name (mux.HandleFunc rather than handle), which logs no
// operation_type, or a hand route named otherwise than the id the spec declares for it.
func (s *LogAppSuite) TestEveryRouteHasAnOperationName() {
	declared := map[string]string{}
	for _, operation := range specifiedOperations((&Server{}).newAPI(chi.NewRouter()).OpenAPI()) {
		// chi matches the connection proxy as "*" where its operation says {path}.
		path := strings.Replace(operation.path, connectionProxyPath+"{path}", connectionProxyPath+"*", 1)
		declared[operation.method+" "+path] = pascalCase(operation.id)
	}
	parameter := regexp.MustCompile(`\{[^}]*\}`)
	s.Require().NoError(chi.Walk(s.router.mux, func(method, route string, _ http.Handler, _ ...func(http.Handler) http.Handler) error {
		// A probe that is answered is not logged (answeredHealthy).
		if route == healthPath {
			return nil
		}
		path := strings.ReplaceAll(parameter.ReplaceAllString(route, "probe"), "*", "probe")
		name := s.field(s.sent(s.unauthenticatedClient, method, path, nil), "operation_type")
		key := method + " " + route
		s.NotEmpty(name, key)
		if want, ok := declared[key]; ok {
			s.Equal(want, name, key)
		}
		return nil
	}))
}

// Defect: a route the spec leaves out (the TeXML answer, a vendor's hook) logged no operation,
// or with its pattern as the name, or with its token.
func (s *LogAppSuite) TestTheAccessLineOfARouteWithNoOperationNamesItsLogOnlyName() {
	token := s.utils.uuid()

	answer := s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/phone/answer/"+token, nil)
	hook := s.sent(s.unauthenticatedClient, http.MethodPost, "/v1/phone/hooks/stream/"+s.utils.uuid(), nil)

	chatHook := s.sent(s.unauthenticatedClient, http.MethodPost, "/v1/chat/hooks/stream", nil)

	s.Contains(answer, "path=/v1/phone/answer/{token}")
	s.Equal("PhoneAnswer", s.field(answer, "operation_type"))
	s.NotContains(answer, token)
	s.Equal("PhoneHook", s.field(hook, "operation_type"))
	s.Equal("ChatHook", s.field(chatHook, "operation_type"))

	// Defect: a peer route logging another peer's name, which a non-empty check passes.
	for path, want := range map[string]string{
		scheduledPeerPath:   "ScheduledTaskDeliver",
		chatInboundPeerPath: "ChatInboundAnswer",
		unansweredPeerPath:  "PhoneEndUnanswered",
		machinePeerPath:     "PhoneMachine",
	} {
		line := s.sent(s.unauthenticatedClient, http.MethodPost, path, nil)
		s.Equal(want, s.field(line, "operation_type"), path)
	}
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

// Defect: a route whose refusal leaves the access line without a code, or with another one
// than its answer names: one answering outside the envelope (a vendor's or gorilla's shape) or
// one added later that answers past writeError and Huma. Every route is asked by a caller of
// each kind, so a refusal before the handler and one in it are both read.
func (s *LogAppSuite) TestEveryRefusalOfEveryRouteNamesItsCodeOnTheAccessLine() {
	// The code a failure of each status has when nothing names it better, for an answer with no
	// envelope to read it from: as ErrorDetail's doc lists them.
	typeCodes := map[int]string{
		http.StatusBadRequest: "invalid_request",
		http.StatusNotFound:   "not_found",
		http.StatusGone:       "gone",
	}
	parameter := regexp.MustCompile(`\{[^}]*\}`)
	refusals := 0
	for _, client := range []*testClient{s.unauthenticatedClient, s.client, s.serverClient} {
		s.Require().NoError(chi.Walk(s.router.mux, func(method, route string, _ http.Handler, _ ...func(http.Handler) http.Handler) error {
			path := strings.ReplaceAll(parameter.ReplaceAllString(route, "probe"), "*", "probe")
			id := s.utils.uuid()
			named := client.with(RequestIDHeader, id)
			status, payload := named.call(method, path, map[string]any{})
			if status < http.StatusBadRequest {
				return nil
			}
			refusals++
			var answered ErrorResponse
			want := typeCodes[status]
			if json.Unmarshal(payload, &answered) == nil && answered.Error.Code != "" {
				want = answered.Error.Code
			}
			key := fmt.Sprintf("%s %s answered %d %s", method, route, status, payload)
			s.Require().NotEmpty(want, key)
			s.Equal(want, s.field(s.lineOf(id), "error_code"), key)
			return nil
		}))
	}
	// The walk read refusals of every kind of caller, not none.
	s.Greater(refusals, 100)
}

// Defect: a refusal withCustomer makes itself, before withRequestLog, written with no access
// line at all.
func (s *LogAppSuite) TestARequestNamingUsageCategoriesThatBreakTheRuleHasAnAccessLineWithItsCode() {
	id := s.utils.uuid()

	status, payload := s.serverClient.with(auth.UsageCategoriesHeader, "not a category!").with(RequestIDHeader, id).
		call(http.MethodGet, "/v1/agents/skills", nil)

	s.Require().Equal(http.StatusBadRequest, status)
	line := s.lineOf(id)
	s.Equal(codeInvalidUsageCategories, s.errorCode(payload))
	s.Equal(codeInvalidUsageCategories, s.field(line, "error_code"))
	s.Contains(line, " status=400 ")
	s.Contains(line, " path=/v1/agents/skills ")
}

// Defect: a caller of a level its app turns away refused with no access line, or with one that
// does not name the app it proved it is of.
func (s *LogAppSuite) TestACallerOfALevelItsAppTurnsAwayHasAnAccessLineNamingTheApp() {
	no := false
	s.useApp(s.data.createAppAdmitting(store.AppSettings{AllowAnonymous: &no}))
	id := s.utils.uuid()

	status, _ := s.anonymousClient.with(RequestIDHeader, id).call(http.MethodGet, "/v1/agents/skills", nil)

	s.Require().Equal(http.StatusForbidden, status)
	line := s.lineOf(id)
	s.Equal("forbidden", s.field(line, "error_code"))
	s.Equal(s.customerID(), s.field(line, "app_id"))
	s.Equal(s.app.key, s.field(line, "api_key"))
	// It is no customer here: nothing it asked for is the app's.
	s.Empty(s.field(line, "customer"))
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

	line := s.servedAs(s.serverClient.with(clientHeader, long).with("User-Agent", straddling).with(auth.AuthTypeHeader, long))

	s.Equal(strings.Repeat("a", loggedCap), s.field(line, "x_stream_client"))
	s.Equal(strings.Repeat("b", loggedCap-1), s.field(line, "user_agent"))
	s.Equal(strings.Repeat("a", loggedCap), s.field(line, "stream_auth_type"))
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

// Defect: a socket declares its auth type in the query, which the line read only from the header.
func (s *LogAppSuite) TestTheAccessLineNamesTheAuthTypeASocketDeclaresInItsQuery() {
	s.Equal("jwt", s.field(s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/account?stream-auth-type=jwt", nil), "stream_auth_type"))
	s.Equal("anonymous", s.field(s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/account?stream-auth-type=anonymous", nil), "stream_auth_type"))
	s.NotContains(s.sent(s.unauthenticatedClient, http.MethodGet, "/v1/account?stream-auth-type=server", nil), "stream_auth_type")
}

// Defect: the load balancer's and kubelet's probes, every few seconds, wrote a line each and
// crowded the log's real requests out.
func (s *LogAppSuite) TestASuccessfulHealthProbeWritesNoAccessLine() {
	server, logged := s.healthServer(false)

	status := s.probed(server, http.MethodGet, "/health")

	s.Equal(http.StatusOK, status)
	s.NotContains(logged.String(), "served a request")
}

// Defect: the quiet rule swallowed a failing probe too, which a failing dependency shows up in.
func (s *LogAppSuite) TestAFailingHealthProbeWritesItsAccessLine() {
	server, logged := s.healthServer(true)

	status := s.probed(server, http.MethodGet, "/health")

	s.Equal(http.StatusServiceUnavailable, status)
	line := s.lineIn(logged.String())
	s.Contains(line, "level=ERROR")
	s.Contains(line, " path=/health ")
	s.Contains(line, " status=503 ")
}

// Defect: the quiet rule went by the status only, and a health probe answered below 400
// whose handler recorded an error went unlogged.
func (s *LogAppSuite) TestAHealthProbeWithARecordedErrorWritesItsAccessLine() {
	line := s.servedAt("/health", func(w http.ResponseWriter, r *http.Request) {
		recordFailure(r.Context(), errors.New("probe broke"), "")
	})

	s.Contains(line, " status=200 ")
	s.Equal("probe broke", s.field(line, "error"))
}

// Defect: the quiet rule's limit was a server error, not 400, as chat's and the ticket's are.
func (s *LogAppSuite) TestAHealthProbeAnsweredWithAClientErrorWritesItsAccessLine() {
	line := s.servedAt("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusBadRequest) })

	s.Contains(line, " status=400 ")
}

// Defect: the rule matched by prefix or suffix, or on every path, so a route that only looks
// like the health check, or a method it does not serve, went unlogged.
func (s *LogAppSuite) TestOnlyAnswersFromTheHealthRouteAreQuiet() {
	for _, path := range []string{"/healthz", "/v1/agents/health", "/health/live", "/ready"} {
		line := s.servedAt(path, func(http.ResponseWriter, *http.Request) {})

		s.Contains(line, " path="+path+" ")
		s.Contains(line, " status=200 ")
	}
	server, logged := s.healthServer(false)

	s.Equal(http.StatusMethodNotAllowed, s.probed(server, http.MethodPost, "/health"))

	s.Contains(s.lineIn(logged.String()), " status=405 ")
}

// Control: a request that is not a health probe is still logged when it succeeds.
func (s *LogAppSuite) TestARequestThatIsNoHealthProbeIsLoggedWhenItSucceeds() {
	line := s.sent(s.serverClient, http.MethodGet, "/v1/agents/skills", nil)

	s.Contains(line, " path=/v1/agents/skills ")
	s.Contains(line, " status=200 ")
}

// healthServer is a router with nothing but its health check, whose store is closed when
// failing, as an unreachable database is, and the log it writes.
func (s *LogAppSuite) healthServer(failing bool) (*Server, *lockedLog) {
	logged := &lockedLog{}
	options := Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Logger:  slog.New(NewLogHandler(slog.NewTextHandler(logged, nil))),
	}
	if failing {
		closed, err := store.Open(testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN")))
		s.Require().NoError(err)
		s.Require().NoError(closed.Close())
		options.Store = closed
	}
	server, err := NewServer(options)
	s.Require().NoError(err)
	return server, logged
}

// probed is the status server answers a method request for path with. The line is written
// by the time it returns.
func (s *LogAppSuite) probed(server *Server, method, path string) int {
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, httptest.NewRequest(method, path, nil))
	return recorder.Code
}

// servedAt is the access line withRequestLog writes for a GET of path that handler answers.
func (s *LogAppSuite) servedAt(path string, handler http.HandlerFunc) string {
	routes := chi.NewRouter()
	routes.Get(path, handler)
	id := s.utils.uuid()
	request := httptest.NewRequest(http.MethodGet, path, nil)
	request.Header.Set(RequestIDHeader, id)

	withRequestID(s.router.withRequestLog(routes, routes)).ServeHTTP(httptest.NewRecorder(), request)

	return s.lineOf(id)
}

// lineIn is the one access line in written, which fails the test when there is none or more.
func (s *LogAppSuite) lineIn(written string) string {
	var lines []string
	for line := range strings.Lines(written) {
		if strings.Contains(line, `msg="served a request"`) {
			lines = append(lines, line)
		}
	}
	s.Require().Len(lines, 1, written)
	return lines[0]
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
	// Defect (AI-3349): the customer the caller's credential names, on a hook nobody checked.
	s.Contains(line, ` customer="" `)
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

// A handler's verified hook names its app on the lines withRequestLog writes after it: a 5xx
// and a panic as well as a success, and on the lines the handler logs with its context. The key
// whose secret signed it is no key the request was made with, so none of them names it. The
// customer the handler noted for it (activity.NoteCustomer) is the lines' customer (AI-3349).
func (s *LogAppSuite) TestTheLinesAfterAHandlerVerifiedAHookNameItsApp() {
	customer := s.utils.uuid()
	for msg, answer := range map[string]func(http.ResponseWriter){
		`msg="served a request" method=POST path=/probe-ok `:      func(http.ResponseWriter) {},
		`msg="served a request" method=POST path=/probe-500 `:     func(w http.ResponseWriter) { w.WriteHeader(http.StatusInternalServerError) },
		`msg="a request panicked" method=POST path=/probe-panic `: func(http.ResponseWriter) { panic("probe panic") },
	} {
		path := strings.Fields(strings.SplitAfter(msg, "path=")[1])[0]
		routes := chi.NewRouter()
		routes.Post(path, func(w http.ResponseWriter, r *http.Request) {
			marked := loggedHook(r, hookOrigin{app: 1234567, apiKey: "the-signing-key"})
			activity.NoteCustomer(r.Context(), customer, activity.AreaHooksIn)
			s.router.logger.InfoContext(marked.Context(), "probe from a hook at "+path)
			answer(w)
		})
		node := httptest.NewUnstartedServer(s.router.withRequestLog(routes, routes))
		node.Config.ErrorLog = log.New(io.Discard, "", 0)
		node.Start()
		s.T().Cleanup(node.Close)

		// A panic goes on to net/http after the 500, which drops the connection.
		if response, err := node.Client().Post(node.URL+path, "application/json", nil); err == nil {
			s.Require().NoError(response.Body.Close())
		}

		var line string
		s.Require().Eventually(func() bool {
			for l := range strings.Lines(s.logged.String()) {
				if strings.Contains(l, msg) {
					line = l
					return true
				}
			}
			return false
		}, 5*time.Second, 20*time.Millisecond, msg)
		s.Equal("1234567", s.field(line, "app_id"), msg)
		s.Equal(customer, s.field(line, "customer"), msg)
		s.NotContains(line, "api_key", msg)
		var probe string
		for l := range strings.Lines(s.logged.String()) {
			if strings.Contains(l, "probe from a hook at "+path) {
				probe = l
			}
		}
		s.Equal("1234567", s.field(probe, "app_id"), path)
		s.NotContains(probe, "api_key", path)
	}
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

// LogAppHookSuite is a router in app mode, where the gateway vouches for each app's own hooks
// once it checked their signature with the app's secret, and what the lines of a hook name.
type LogAppHookSuite struct {
	appLogSuite

	apiKey string
}

func TestLogAppHookSuite(t *testing.T) {
	hooks := new(LogAppHookSuite)
	hooks.appMode = true
	runSuite(t, hooks)
}

func (s *LogAppHookSuite) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.apiKey = "own-key-" + s.utils.uuid()
	s.answerAs(s.apiKey, s.appID())
	s.forwarded(s.apiKey)
}

// The line of a hook to an app's own path, as Stream delivers them in app mode, names the app.
func (s *LogAppHookSuite) TestTheAccessLineOfAHookTheGatewayVouchedForNamesItsApp() {
	for _, hook := range []string{phone.CallHookPath, chat.MessageHookPath} {
		request := s.hookTo(hook + "/" + s.customerID())
		vouched(request, s.appID(), s.apiKey)

		status, line := s.hookLine(request)

		s.Equal(http.StatusOK, status, hook)
		s.Equal(s.customerID(), s.field(line, "app_id"), hook)
		s.NotContains(line, "api_key", hook)
	}
}

// A hook nobody verified names no app, whichever its path names.
func (s *LogAppHookSuite) TestTheAccessLineOfAHookNobodyVouchedForNamesNoApp() {
	for _, hook := range []string{phone.CallHookPath, chat.MessageHookPath} {
		path := hook + "/" + s.customerID()
		unvouched := s.hookTo(path)
		forAnother := s.hookTo(path)
		vouched(forAnother, s.appID()+1, s.apiKey)

		for name, request := range map[string]*http.Request{"no mark": unvouched, "another app's mark": forAnother} {
			status, line := s.hookLine(request)

			s.Equal(http.StatusUnauthorized, status, hook, name)
			s.NotContains(line, "app_id", hook, name)
		}
	}
}

// A hook to the path naming no app is the deployment's, signed with its secret.
func (s *LogAppHookSuite) TestTheAccessLineOfTheDeploymentsHookNamesTheDeploymentsApp() {
	status, line := s.hookLine(s.hookTo(chat.MessageHookPath))

	s.Equal(http.StatusOK, status)
	s.Equal(strconv.Itoa(suiteStreamApp), s.field(line, "app_id"))
	s.NotContains(line, "api_key")
}

// hookTo is an event the hooks do not act on, posted to path and signed with the deployment's
// secret, which no app's own hook is checked with.
func (s *LogAppHookSuite) hookTo(path string) *http.Request {
	body := `{"type": "reaction.new"}`
	return s.request(path, body, sign(body, suiteStreamSecret))
}

// hookLine sends request and is its status and its access line.
func (s *LogAppHookSuite) hookLine(request *http.Request) (int, string) {
	id := s.utils.uuid()
	request.Header.Set(RequestIDHeader, id)
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode, s.lineOf(id)
}
