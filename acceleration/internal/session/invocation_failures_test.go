//go:build integration

package session

import (
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The provider's status and the error message an invocation row keeps (AI-1244): the line's
// provider_status when an answer came, and its provider_message, else its provider_body.

func (s *CallLogSuite) TestAFailedCallsRowKeepsTheStatusAndTheMessageOfItsLine() {
	app := s.connection("", "primary")
	d, logged := s.logging(s.spec(s.config(s.fixed("crm", app, "fails")), "", nil))

	_, err := s.call(d, "crm__fails", "{}")

	s.Require().Error(err)
	row := s.row(app)
	s.Equal(store.InvocationExternalServer, row.ErrorType)
	s.Equal(statusOf(http.StatusOK), row.ProviderStatus, "an MCP isError comes in a 200")
	s.Equal("the record is locked", row.ErrorMessage)
	s.Contains(s.onlyFailedLine(logged), `provider_message="the record is locked"`)
}

// TestARowWithNoErrorKeepsTheProvidersBody: a 429 with a body and a Retry-After, which the
// limiter holds, returns no error, so the row keeps the start of the body.
func (s *CallLogSuite) TestARowWithNoErrorKeepsTheProvidersBody() {
	manager, _ := s.loggingManager()
	manager.options.Connectors.Limiter = s.limiter()
	s.limitPerApp()
	app := s.connection("", "primary")
	d := s.attachOn(manager, s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.provider.answer("primary", http.StatusTooManyRequests, http.Header{"Retry-After": {"30"}}, "slow down, see https://crm.example/limits?key=abc")

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Contains(said, "connector_rate_limited")
	row := s.row(app)
	s.Equal(store.InvocationExternalServer, row.ErrorType)
	s.Equal(statusOf(http.StatusTooManyRequests), row.ProviderStatus)
	s.Equal("slow down, see https://crm.example", row.ErrorMessage)
}

func (s *CallLogSuite) TestACallThatAnswersKeepsNoStatusAndNoMessage() {
	app := s.connection("", "primary")
	d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal("primary", said)
	row := s.row(app)
	s.Empty(row.ErrorType)
	s.Nil(row.ProviderStatus)
	s.Empty(row.ErrorMessage)
}

// TestARefusedCallKeepsTheReasonAndNoStatus: nothing was sent, so no status.
func (s *CallLogSuite) TestARefusedCallKeepsTheReasonAndNoStatus() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	d, _ := s.logging(s.spec(config, "", nil))
	s.rebind(config, s.fixed("crm", app))

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	row := s.row(app)
	s.Equal(store.InvocationDenied, row.ErrorType)
	s.Nil(row.ProviderStatus)
	s.Equal(`session: connector "crm": the tool is no longer granted, so crm__whoami was not called`, row.ErrorMessage)
}

// TestARowKeepsNoTokenThePathOrAQueryTheProviderEchoes: a 400 whose body names the token it
// was sent, the request's path and a URL with a query, and then stalls past the binding's
// timeout, so the call is outcome_unknown with no error and the row keeps the body.
func (s *CallLogSuite) TestARowKeepsNoTokenThePathOrAQueryTheProviderEchoes() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "whoami")
	binding.TimeoutMs = 1000
	d, _ := s.logging(s.spec(s.config(binding), "", nil))
	s.provider.answerThen("primary", http.StatusBadRequest, http.Header{"Content-Type": {"text/plain"}},
		"invalid token "+tokenOf("primary")+" for /primary/mcp, see https://status.crm.example/i?token="+queryToken+" ",
		holdUntilGone)

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	row := s.row(app)
	s.Equal(store.InvocationOutcomeUnknown, row.ErrorType)
	s.Equal(statusOf(http.StatusBadRequest), row.ProviderStatus)
	s.Equal("invalid token [redacted] for [redacted], see https://status.crm.example ", row.ErrorMessage)
}

// TestARowNamesNoCredential: the provider refuses the token it was sent, and the model put
// another in the call's arguments; the row keeps neither, nor the Authorization header.
func (s *CallLogSuite) TestARowNamesNoCredential() {
	app := s.connection("", "primary")
	d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.sendWrongToken(app)

	_, err := s.call(d, "crm__whoami", `{"api_key": "`+tokenOf("moved")+`"}`)

	s.Require().Error(err)
	row := s.row(app)
	s.Equal(store.InvocationCustomerAuth, row.ErrorType)
	s.Equal(statusOf(http.StatusUnauthorized), row.ProviderStatus)
	s.Equal(`mcp: crm__whoami: calling "tools/call": sending "tools/call": rejected by transport: `+
		"mcp: the server refused the request with 401 Unauthorized", row.ErrorMessage)
}

// heldOnTheBase is what the model read of a call the limiter held before sending, the
// provider's Retry-After being 30, on 11fae4b1 (probe/base-11fae4b1.txt in the AI-1244 ledger,
// by probe/zz_probe_test.go there). Its row's error_type was denied, and its line's
// provider_message "".
const heldOnTheBase = "connector_rate_limited: the provider limits how often crm__whoami may be called, and it " +
	"was not run. retry_after_seconds: 30. Do not call it again before then."

// TestACallTheLimiterHoldsSaysSoWithItsRetryAfter: the call is not sent; its row stays denied,
// as on the base, with no status and the router's reason, which its line says too. What the
// model reads is the base's.
func (s *CallLogSuite) TestACallTheLimiterHoldsSaysSoWithItsRetryAfter() {
	manager, logged := s.loggingManager()
	manager.options.Connectors.Limiter = s.limiter()
	s.limitPerApp()
	app := s.connection("", "primary")
	d := s.attachOn(manager, s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.provider.limit("primary", "30")
	_, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.provider.lift()
	logged.Reset()

	held, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal(heldOnTheBase, held)
	s.Equal(1, s.provider.calls("primary"), "the held call was not sent")
	rows := s.rows(app, 2)
	s.Equal(store.InvocationDenied, rows[0].ErrorType)
	s.Nil(rows[0].ProviderStatus)
	s.Equal("rate limited by the router: retry after 30s", rows[0].ErrorMessage)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=denied")
	s.Contains(line, `provider_message="rate limited by the router: retry after 30s"`)
}

// TestAnIncognitoSessionsRowKeepsTheMessage: as its line does, by the owner's decision
// (AI-1242, 2026-10-14), with no session.
func (s *CallLogSuite) TestAnIncognitoSessionsRowKeepsTheMessage() {
	app := s.connection("", "primary")
	spec := s.spec(s.config(s.fixed("crm", app, "fails")), "", nil)
	spec.ID, spec.Incognito = uuid.NewString(), true
	d, _ := s.logging(spec)

	_, err := s.call(d, "crm__fails", "{}")

	s.Require().Error(err)
	row := s.row(app)
	s.Empty(row.SessionID)
	s.Equal("the record is locked", row.ErrorMessage)
}

// rows are connection's invocation rows, newest first, once count are written.
func (s *CallLogSuite) rows(connection string, count int) []store.ConnectorInvocation {
	var rows []store.ConnectorInvocation
	s.Require().Eventually(func() bool {
		var err error
		rows, err = s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		return err == nil && len(rows) == count
	}, 5*time.Second, 20*time.Millisecond)
	return rows
}

// statusOf is status as a row's ProviderStatus holds it.
func statusOf(status int) *int { return &status }
