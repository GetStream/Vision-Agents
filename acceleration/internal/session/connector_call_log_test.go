//go:build integration

package session

import (
	"bytes"
	"fmt"
	"log/slog"
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
		`provider_message="the record is locked"`} {
		s.Contains(line, field)
	}
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
	for _, secret := range []string{tokenOf("secondary"), tokenOf("primary"), tokenOf("moved"), "Bearer"} {
		s.NotContains(logged.String(), secret)
	}
}

// TestATimedOutCallLogsNoMessage: outcome_unknown returns no error, so nothing is said for it.
func (s *CallLogSuite) TestATimedOutCallLogsNoMessage() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 200
	d, logged := s.logging(s.spec(s.config(binding), "", nil))

	_, err := s.call(d, "crm__slow", "{}")

	s.Require().NoError(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=outcome_unknown")
	s.Contains(line, `provider_message=""`)
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

// TestAnIncognitoSessionsLineNamesNoSessionAndNoMessage: the message can be the tool's answer,
// which Spec.Incognito keeps out of every record.
func (s *CallLogSuite) TestAnIncognitoSessionsLineNamesNoSessionAndNoMessage() {
	app := s.connection("", "primary")
	spec := s.spec(s.config(s.fixed("crm", app, "fails")), "", nil)
	spec.ID, spec.Incognito = uuid.NewString(), true
	d, logged := s.logging(spec)

	_, err := s.call(d, "crm__fails", "{}")

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "error_type=external_server")
	s.Contains(line, `session="" `)
	s.Contains(line, `provider_message=""`)
	s.NotContains(logged.String(), spec.ID)
	s.NotContains(logged.String(), "the record is locked")
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
