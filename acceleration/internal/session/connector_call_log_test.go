//go:build integration

package session

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// CallLogSuite is the "connector call failed" line a session's dispatcher logs for a call that
// failed or was refused (AI-1242): the invocation row's names and id, the provider's status and
// what the provider or the refusal said, and no line for a call that answered. It also holds
// the control: what the model reads of each call, and its row's error_type, are what they were
// on the base sha.
type CallLogSuite struct {
	connectorFixture
	address string
}

func TestCallLogSuite(t *testing.T) {
	suite.Run(t, new(CallLogSuite))
}

func (s *CallLogSuite) SetupSuite() {
	s.address = os.Getenv("ROUTER_REDIS_ADDR")
	if s.address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	s.connectorFixture.SetupSuite()
}

func (s *CallLogSuite) TestAnIsErrorAnswerLogsTheToolsWordsWithTheRowsNamesAndID() {
	app := s.connection("", "primary")
	spec := s.spec(s.config(s.fixed("crm", app, "fails")), "", nil)
	spec.ID = uuid.NewString()
	d, logged := s.logging(spec)

	_, err := s.call(d, "crm__fails", "{}")

	s.Require().Error(err)
	row := s.row(app)
	line := s.onlyFailedLine(logged)
	for _, field := range []string{"level=WARN", "customer=" + s.customerID, "connection=" + app,
		"connector=" + s.connectorID, "binding=crm", "tool=fails", "session=" + spec.ID, "invocation=" + row.ID,
		"error_type=external_server", fmt.Sprintf("latency_ms=%d ", row.LatencyMs), "provider_status=200",
		`provider_message="the record is locked"`, "provider_host=" + s.providerHost(), `provider_body=""`} {
		s.Contains(line, field)
	}
	s.NotContains(line, "www_authenticate=")
	s.NotContains(line, "retry_after=")
}

// TestA403LogsTheHostTheBodyAndTheChallenge: the provider refuses with a JSON body and a
// cookie; the line has the body and the challenge, and neither the cookie nor the token
// (AI-1243).
func (s *CallLogSuite) TestA403LogsTheHostTheBodyAndTheChallenge() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.provider.answer("primary", http.StatusForbidden, http.Header{"Content-Type": {"application/json"},
		"Www-Authenticate": {`Bearer realm="crm"`}, "Set-Cookie": {"session=" + setCookieSecret}},
		`{"error":"forbidden","message":"this app may not read contacts"}`)

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	for _, field := range []string{"error_type=customer_auth", "provider_status=403", "provider_host=" + s.providerHost() + " ",
		`provider_body="{\"error\":\"forbidden\",\"message\":\"this app may not read contacts\"}"`,
		`www_authenticate="Bearer realm=\"crm\""`} {
		s.Contains(line, field)
	}
	s.NotContains(line, "retry_after=")
	for _, secret := range []string{setCookieSecret, tokenOf("primary"), "/primary/mcp"} {
		s.NotContains(logged.String(), secret)
	}
}

// TestA5xxsBodyIsCutAtARuneAndItsURLsKeepTheirHost: a 502 with a 3 KiB page whose kept KiB
// ends inside a rune, a status page URL with a token, and a Retry-After.
func (s *CallLogSuite) TestA5xxsBodyIsCutAtARuneAndItsURLsKeepTheirHost() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	said := "bad gateway, see https://status.crm.example/incidents?token=" + queryToken + " "
	s.Require().NotZero((providerMessageBytes-len(said))%3, "the KiB ends inside a 3-byte rune")
	s.provider.answer("primary", http.StatusBadGateway, http.Header{"Retry-After": {"120"}}, said+strings.Repeat("€", 1000))

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	kept := "bad gateway, see https://status.crm.example " + strings.Repeat("€", (providerMessageBytes-len(said))/3)
	for _, field := range []string{"error_type=external_server", "provider_status=502",
		"provider_host=" + s.providerHost() + " ", `provider_body="` + kept + `"`, "retry_after=120"} {
		s.Contains(line, field)
	}
	s.NotContains(line, "www_authenticate=")
	s.NotContains(logged.String(), queryToken)
}

func (s *CallLogSuite) TestARefusedCallLogsTheReason() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	d, logged := s.logging(s.spec(config, "", nil))
	s.rebind(config, s.fixed("crm", app))

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=denied")
	s.Contains(line, "provider_status=0")
	s.Contains(line, `provider_message="session: connector \"crm\": the tool is no longer granted, so crm__whoami was not called"`)
	s.Zero(s.provider.calls("primary"))
}

func (s *CallLogSuite) TestACallThatAnswersLogsNothing() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Empty(s.failedLines(logged))
}

// TestTheLineNamesNoCredential: the provider refuses the bearer token it was sent, and the
// model put another in the call's arguments; neither, nor the Authorization header, is logged.
func (s *CallLogSuite) TestTheLineNamesNoCredential() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.sendWrongToken(app)

	_, err := s.call(d, "crm__whoami", fmt.Sprintf(`{"api_key": %q}`, tokenOf("moved")))

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=customer_auth")
	s.Contains(line, "provider_status=401")
	// The provider's challenge, `Bearer error="invalid_token"`, is logged (AI-1243); the
	// Authorization the router sent is not.
	s.Contains(line, `www_authenticate="Bearer error=\"invalid_token\""`)
	for _, secret := range []string{tokenOf("secondary"), tokenOf("primary"), tokenOf("moved"), "Bearer token"} {
		s.NotContains(logged.String(), secret)
	}
}

// TestATimedOutCallLogsNoMessage: outcome_unknown returns no error, so nothing is said for it.
// The provider streams, so its 200 reached the router before the timeout: the line has it.
func (s *CallLogSuite) TestATimedOutCallLogsNoMessage() {
	s.provider.streamFirst()
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 200
	d, logged := s.logging(s.spec(s.config(binding), "", nil))

	_, err := s.call(d, "crm__slow", "{}")

	s.Require().NoError(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=outcome_unknown")
	s.Contains(line, "provider_status=200")
	s.Contains(line, `provider_message=""`)
	s.Contains(line, "provider_host="+s.providerHost())
}

// TestARateLimitedCallLogsTheStatusAndNoMessage: the 429 the limiter holds is answered as
// connector_rate_limited, with no error.
func (s *CallLogSuite) TestARateLimitedCallLogsTheStatusAndNoMessage() {
	manager, logged := s.loggingManager()
	manager.options.Connectors.Limiter = s.limiter()
	s.limitPerApp()
	app := s.connection("", "primary")
	d := s.attachOn(manager, s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.provider.limit("primary", "30")
	defer s.provider.lift()

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Contains(said, "connector_rate_limited")
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=external_server")
	s.Contains(line, "provider_status=429")
	s.Contains(line, `provider_message=""`)
}

// TestAnIncognitoSessionsLineNamesNoSessionButLogsTheMessage: the owner decided (AI-1242,
// 2026-10-14) that the message is logged for an incognito session too; the session id stays
// out, as it does of the row (Spec.Incognito).
func (s *CallLogSuite) TestAnIncognitoSessionsLineNamesNoSessionButLogsTheMessage() {
	app := s.connection("", "primary")
	spec := s.spec(s.config(s.fixed("crm", app, "fails")), "", nil)
	spec.ID, spec.Incognito = uuid.NewString(), true
	d, logged := s.logging(spec)

	_, err := s.call(d, "crm__fails", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=external_server")
	s.Contains(line, `session="" `)
	s.Contains(line, `provider_message="the record is locked"`)
	s.NotContains(logged.String(), spec.ID)
}

// TestWhatTheModelReadsAndEachErrorTypeAreTheBases is the control: every value is what the
// same calls gave on e88504e4, before the log line (probe/base-e88504e4.txt in the AI-1242
// ledger, by zz_probe_test.go there).
func (s *CallLogSuite) TestWhatTheModelReadsAndEachErrorTypeAreTheBases() {
	s.Run("success", func() {
		app := s.connection("", "primary")
		d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami", "fails", "echo")), "", nil))
		s.outcome(app, "primary", "", "")(s.call(d, "crm__whoami", "{}"))
	})
	s.Run("isError", func() {
		app := s.connection("", "primary")
		d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "fails")), "", nil))
		s.outcome(app, "", "the record is locked", "external_server")(s.call(d, "crm__fails", "{}"))
	})
	s.Run("refused", func() {
		app := s.connection("", "primary")
		config := s.config(s.fixed("crm", app, "whoami"))
		d, _ := s.logging(s.spec(config, "", nil))
		s.rebind(config, s.fixed("crm", app))
		s.outcome(app, "", `session: connector "crm": the tool is no longer granted, so crm__whoami was not called`,
			"denied")(s.call(d, "crm__whoami", "{}"))
	})
	s.Run("arguments the schema refuses", func() {
		app := s.connection("", "primary")
		d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "echo")), "", nil))
		s.outcome(app, "", "mcp: crm__echo: the arguments do not match the tool's input schema",
			"denied")(s.call(d, "crm__echo", `{"note": 42}`))
	})
	s.Run("401", func() {
		app := s.connection("", "primary")
		d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
		s.sendWrongToken(app)
		s.outcome(app, "", `mcp: crm__whoami: calling "tools/call": sending "tools/call": rejected by transport: `+
			"mcp: the server refused the request with 401 Unauthorized", "customer_auth")(s.call(d, "crm__whoami", "{}"))
	})
	s.Run("timeout", func() {
		app := s.connection("", "primary")
		binding := s.fixed("crm", app, "slow")
		binding.TimeoutMs = 200
		d, _ := s.logging(s.spec(s.config(binding), "", nil))
		s.outcome(app, "outcome_unknown: crm__slow did not answer in time. It may or may not have done what was "+
			"asked; check before calling it again.", "", "outcome_unknown")(s.call(d, "crm__slow", "{}"))
	})
	s.Run("429 without a limiter", func() {
		app := s.connection("", "primary")
		d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
		s.provider.limit("primary", "30")
		defer s.provider.lift()
		s.outcome(app, "", `mcp: crm__whoami: calling "tools/call": rejected by transport: sending "tools/call": `+
			"Too Many Requests", "external_server")(s.call(d, "crm__whoami", "{}"))
	})
	s.Run("429 held by the limiter", func() {
		manager, _ := s.loggingManager()
		manager.options.Connectors.Limiter = s.limiter()
		s.limitPerApp()
		app := s.connection("", "primary")
		d := s.attachOn(manager, s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
		s.provider.limit("primary", "30")
		defer s.provider.lift()
		s.outcome(app, "connector_rate_limited: the provider limits how often crm__whoami may be called, and it "+
			"was not run. retry_after_seconds: 30. Do not call it again before then.", "",
			"external_server")(s.call(d, "crm__whoami", "{}"))
	})
}

// TestWhatTheModelReadsOfAnErrorAnswerIsTheBases is the control for AI-1243, which reads the
// start of a 4xx's or a 5xx's body before the MCP client does: every value is what the same
// calls gave on 11102295 (probe/base-11102295.txt in the AI-1243 ledger, by
// probe/zz_probe_test.go there). The 400's JSON-RPC error is over 1 KiB, and the client reads
// its message only from the whole body.
func (s *CallLogSuite) TestWhatTheModelReadsOfAnErrorAnswerIsTheBases() {
	jsonRPC := `{"jsonrpc":"2.0","id":2,"error":{"code":-32000,"message":"the CRM is down","data":"` +
		strings.Repeat("x", 3000) + `"}}`
	cases := []struct {
		name, failed, errorType string
		status                  int
		header                  http.Header
		body                    string
	}{
		{"403 json", `mcp: crm__whoami: calling "tools/call": sending "tools/call": rejected by transport: ` +
			"mcp: the server refused the request with 403 Forbidden", "customer_auth", http.StatusForbidden,
			http.Header{"Content-Type": {"application/json"}, "Www-Authenticate": {`Bearer realm="crm"`},
				"Set-Cookie": {"session=" + setCookieSecret}},
			`{"error":"forbidden","message":"this app may not read contacts"}`},
		{"500 json-rpc over 1 KiB", `mcp: crm__whoami: calling "tools/call": rejected by transport: ` +
			`sending "tools/call": Internal Server Error`, "external_server", http.StatusInternalServerError,
			http.Header{"Content-Type": {"application/json"}}, jsonRPC},
		{"400 json-rpc over 1 KiB", `mcp: crm__whoami: calling "tools/call": sending "tools/call": ` +
			"the CRM is down: rejected by transport: Bad Request", "external_server", http.StatusBadRequest,
			http.Header{"Content-Type": {"application/json"}}, jsonRPC},
		{"502 html with retry-after", `mcp: crm__whoami: calling "tools/call": rejected by transport: ` +
			`sending "tools/call": Bad Gateway`, "external_server", http.StatusBadGateway,
			http.Header{"Content-Type": {"text/html"}, "Retry-After": {"120"}},
			"<html><body>bad gateway, see https://status.crm.example/incidents?token=abc</body></html>"},
	}
	for _, c := range cases {
		s.Run(c.name, func() {
			app := s.connection("", "primary")
			d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
			s.provider.answer("primary", c.status, c.header, c.body)
			s.outcome(app, "", c.failed, c.errorType)(s.call(d, "crm__whoami", "{}"))
		})
	}
}

// TestAnErrorAnswerWhoseBodyStallsIsTheBases: a 4xx or a 5xx whose body stalls or drips,
// under a binding timeout of 1 s. What the model reads, the error, the row's error_type and
// how soon the call is back are what the same calls gave on 11102295 (rprobe-base.txt in the
// AI-1243 ledger): the MCP client closes a 500's, 502's or 503's body unread and the call is
// back at once, external_server; it reads a 400's to the end, so that one is outcome_unknown
// there too. The line keeps what came of the body at once.
func (s *CallLogSuite) TestAnErrorAnswerWhoseBodyStallsIsTheBases() {
	rejected := `mcp: crm__whoami: calling "tools/call": rejected by transport: sending "tools/call": `
	cases := []struct {
		name, body, said, failed, errorType, logged string
		status                                      int
		then                                        func(http.ResponseWriter, *http.Request)
		back                                        time.Duration
	}{
		{"500 stalls at once", "", "", rejected + "Internal Server Error", "external_server", `provider_body=""`,
			http.StatusInternalServerError, holdUntilGone, 500 * time.Millisecond},
		{"500 sends 100 B then stalls", strings.Repeat("e", 100), "", rejected + "Internal Server Error",
			"external_server", "provider_body=" + strings.Repeat("e", 100), http.StatusInternalServerError,
			holdUntilGone, 500 * time.Millisecond},
		{"502 html stalls", "<html><body>bad gateway", "", rejected + "Bad Gateway", "external_server",
			`provider_body="<html><body>bad gateway"`, http.StatusBadGateway, holdUntilGone, 500 * time.Millisecond},
		{"503 drips 40 B over 4 s", "", "", rejected + "Service Unavailable", "external_server", "provider_status=503",
			http.StatusServiceUnavailable, drip(40), 500 * time.Millisecond},
		{"400 stalls, read to the end on the base too", `{"jsonrpc":"2.0"`, outcomeUnknown("crm__whoami"), "",
			"outcome_unknown", "provider_status=400", http.StatusBadRequest, holdUntilGone, 2 * time.Second},
	}
	for _, c := range cases {
		s.Run(c.name, func() {
			app := s.connection("", "primary")
			binding := s.fixed("crm", app, "whoami")
			binding.TimeoutMs = 1000
			d, logged := s.logging(s.spec(s.config(binding), "", nil))
			s.provider.answerThen("primary", c.status, http.Header{"Content-Type": {"text/plain"}}, c.body, c.then)

			started := time.Now()
			said, err := s.call(d, "crm__whoami", "{}")

			s.Less(time.Since(started), c.back)
			s.outcome(app, c.said, c.failed, c.errorType)(said, err)
			s.Contains(s.onlyFailedLine(logged), c.logged)
		})
	}
}

// holdUntilGone holds a provider's answer until the router is gone or five seconds pass.
func holdUntilGone(_ http.ResponseWriter, r *http.Request) {
	select {
	case <-r.Context().Done():
	case <-time.After(5 * time.Second):
	}
}

// drip sends n bytes of a body, one every 100 ms, until the router is gone.
func drip(n int) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		for range n {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(100 * time.Millisecond):
			}
			_, _ = io.WriteString(w, "d")
			w.(http.Flusher).Flush()
		}
	}
}

// TestALineKeepsNoPathOrTokenTheProviderEchoes: Express's default 404 names the path it was
// sent, and a 400 repeats the token it refused (AI-1243).
func (s *CallLogSuite) TestALineKeepsNoPathOrTokenTheProviderEchoes() {
	cases := []struct {
		name, body, logged string
		status             int
	}{
		{"express 404", "<pre>Cannot POST /primary/mcp</pre>", `provider_body="<pre>Cannot POST [redacted]</pre>"`,
			http.StatusNotFound},
		{"spring error json", `{"status":404,"error":"Not Found","path":"/primary/mcp"}`,
			`provider_body="{\"status\":404,\"error\":\"Not Found\",\"path\":\"[redacted]\"}"`, http.StatusNotFound},
		{"echoed token", "invalid token " + tokenOf("primary"), `provider_body="invalid token [redacted]"`,
			http.StatusBadRequest},
	}
	for _, c := range cases {
		s.Run(c.name, func() {
			app := s.connection("", "primary")
			d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
			s.provider.answer("primary", c.status, http.Header{"Content-Type": {"text/html"}}, c.body)

			_, err := s.call(d, "crm__whoami", "{}")

			s.Require().Error(err)
			s.Contains(s.onlyFailedLine(logged), c.logged)
			for _, secret := range []string{"/primary/mcp", tokenOf("primary")} {
				s.NotContains(logged.String(), secret)
			}
		})
	}
}

// TestA400LogsItsBody: the MCP client reads a 400's body to the end, and the line has it.
func (s *CallLogSuite) TestA400LogsItsBody() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.provider.answer("primary", http.StatusBadRequest, http.Header{"Content-Type": {"text/plain"}}, "the arguments are wrong")

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "provider_status=400")
	s.Contains(line, `provider_body="the arguments are wrong"`)
}

// TestEachHeaderIsCutToAKiB: a WWW-Authenticate and a Retry-After of 2 KiB each are logged to
// their first KiB.
func (s *CallLogSuite) TestEachHeaderIsCutToAKiB() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	challenge := "Bearer realm=" + strings.Repeat("r", 2048)
	wait := strings.Repeat("9", 2048)
	s.provider.answer("primary", http.StatusBadGateway, http.Header{"Www-Authenticate": {challenge},
		"Retry-After": {wait}}, "down")

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, `www_authenticate="`+challenge[:providerMessageBytes]+`"`)
	s.True(strings.HasSuffix(line, " retry_after="+wait[:providerMessageBytes]), "the line ends with the cut Retry-After")
}

// providerHost is the host and port the suite's provider listens on.
func (s *CallLogSuite) providerHost() string {
	return strings.TrimPrefix(s.provider.URL, "https://")
}

// setCookieSecret is a cookie the provider sets, which no line may name. Synthetic.
const setCookieSecret = "set-cookie-secret-1243"

// outcome checks one call of connection: what the model read, the error's text ("" for
// none) and the row's error_type.
func (s *CallLogSuite) outcome(connection, said, failed, errorType string) func(string, error) {
	return func(gotSaid string, err error) {
		s.Equal(said, gotSaid)
		if failed == "" {
			s.NoError(err)
		} else {
			s.EqualError(err, failed)
		}
		s.Equal(errorType, s.row(connection).ErrorType)
	}
}

// loggingManager is a manager of the suite's whose logger writes text to the buffer.
func (s *CallLogSuite) loggingManager() (*Manager, *bytes.Buffer) {
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug}))
	return manager, &logged
}

// logging opens spec on a loggingManager.
func (s *CallLogSuite) logging(spec Spec) (*dispatcher, *bytes.Buffer) {
	manager, logged := s.loggingManager()
	return s.attachOn(manager, spec), logged
}

// attachOn opens spec's connectors on manager, closed when the test ends.
func (s *CallLogSuite) attachOn(manager *Manager, spec Spec) *dispatcher {
	d, _, unavailable, err := manager.attachConnectors(s.ctx, &spec)
	s.Require().NoError(err)
	s.Require().Empty(unavailable)
	s.T().Cleanup(d.Close)
	return d
}

// failedLines are the "connector call failed" lines in logged.
func (s *CallLogSuite) failedLines(logged *bytes.Buffer) []string {
	var lines []string
	for _, line := range strings.Split(logged.String(), "\n") {
		if strings.Contains(line, `msg="connector call failed"`) {
			lines = append(lines, line)
		}
	}
	return lines
}

// onlyFailedLine is the one "connector call failed" line in logged.
func (s *CallLogSuite) onlyFailedLine(logged *bytes.Buffer) string {
	lines := s.failedLines(logged)
	s.Require().Len(lines, 1, logged.String())
	return lines[0]
}

// row is the one invocation row of connection, once it is written.
func (s *CallLogSuite) row(connection string) store.ConnectorInvocation {
	var rows []store.ConnectorInvocation
	s.Require().Eventually(func() bool {
		var err error
		rows, err = s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		return err == nil && len(rows) == 1
	}, 5*time.Second, 20*time.Millisecond)
	return rows[0]
}

// sendWrongToken stores another account's token on connection, which its provider refuses.
func (s *CallLogSuite) sendWrongToken(connection string) {
	wrong, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: tokenOf("secondary")}})
	s.Require().NoError(err)
	s.setState(connection, func(state *core.CredentialState) { state.Credentials = wrong })
}

// limiter is a limiter over the suite's Redis, with a client of its own.
func (s *CallLogSuite) limiter() *core.Limiter {
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{s.address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return core.NewLimiter(client, nil)
}

// limitPerApp makes the next revision of the test's connector limited per app, as
// RateLimitSuite.limitPer does, and has the test's next connections pin it.
func (s *CallLogSuite) limitPerApp() {
	manifest, err := core.ParseManifest([]byte(fmt.Sprintf(`
id: %s
revision: 1
name: CRM
inputs:
  - name: account
    enum: [primary, secondary, moved]
endpoints:
  mcp: %s/{account}/mcp
schemes: [bearer]
identity: [account]
rate_limit:
  per: app
sources:
  - kind: mcp
    endpoint: mcp
`, s.connectorID, s.provider.URL)))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
}
