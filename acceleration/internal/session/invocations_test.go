//go:build integration

package session

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// InvocationLogSuite is the row each connector tool call leaves (T29): one for every call the
// dispatcher opened, refused or run, how it failed, nothing of what it was asked or answered,
// and never on the call's way.
type InvocationLogSuite struct {
	connectorFixture
}

func TestInvocationLogSuite(t *testing.T) {
	suite.Run(t, new(InvocationLogSuite))
}

// logged waits for connection's log to hold count rows and a moment more, so a second row
// written late would be seen, and returns them newest first.
func (s *InvocationLogSuite) logged(connection string, count int) []store.ConnectorInvocation {
	read := func() []store.ConnectorInvocation {
		rows, err := s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		s.Require().NoError(err)
		return rows
	}
	s.Require().Eventually(func() bool { return len(read()) >= count }, 5*time.Second, 20*time.Millisecond)
	s.Never(func() bool { return len(read()) > count }, 300*time.Millisecond, 50*time.Millisecond)
	return read()
}

// stored is every column of connection's log rows as Postgres holds them, as text.
func (s *InvocationLogSuite) stored(connection string) string {
	var rows string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT coalesce(string_agg(to_jsonb(ci)::text, ' '), '') FROM connector_invocations AS ci WHERE connection_id = ?",
		connection).Scan(&rows))
	return rows
}

func (s *InvocationLogSuite) TestACallThatAnswersLeavesOneRowNamingItsBindingAndTool() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	spec := s.spec(config, "", nil)
	spec.ID = uuid.NewString()
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)

	_, err = s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	rows := s.logged(app, 1)
	s.Equal("crm", rows[0].Binding)
	s.Equal("whoami", rows[0].Tool)
	s.Equal(s.connectorID, rows[0].ConnectorID)
	s.Equal(config.ID, rows[0].ConfigID)
	s.Equal(spec.ID, rows[0].SessionID)
	s.Empty(rows[0].ErrorType)
	s.GreaterOrEqual(rows[0].LatencyMs, int64(0))
}

func (s *InvocationLogSuite) TestARefusedCallLeavesADeniedRow() {
	app := s.connection("", "primary")
	config := s.config(s.fixed("crm", app, "whoami"))
	d, _, _, err := s.attach(s.spec(config, "", nil))
	s.Require().NoError(err)

	s.rebind(config, s.fixed("crm", app))
	_, refused := s.call(d, "crm__whoami", "{}")

	s.Require().Error(refused)
	s.Equal(store.InvocationDenied, s.logged(app, 1)[0].ErrorType)
	s.Zero(s.provider.calls("primary"))
}

func (s *InvocationLogSuite) TestArgumentsTheSourceRefusesLeaveADeniedRow() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "echo")), "", nil))
	s.Require().NoError(err)

	_, err = s.call(d, "crm__echo", `{"note": 42}`)

	s.Require().Error(err)
	s.Equal(store.InvocationDenied, s.logged(app, 1)[0].ErrorType)
	s.Zero(s.provider.calls("primary"), "nothing was sent")
}

func (s *InvocationLogSuite) TestAToolThatReportsItsOwnFailureLeavesAnExternalServerRow() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "fails")), "", nil))
	s.Require().NoError(err)

	_, err = s.call(d, "crm__fails", "{}")

	s.Require().ErrorContains(err, "the record is locked")
	s.Equal(store.InvocationExternalServer, s.logged(app, 1)[0].ErrorType)
}

func (s *InvocationLogSuite) TestACallTheTimeoutCutsOffLeavesAnOutcomeUnknownRow() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 200
	d, _, _, err := s.attach(s.spec(s.config(binding), "", nil))
	s.Require().NoError(err)

	said, err := s.call(d, "crm__slow", "{}")

	s.Require().NoError(err)
	s.Contains(said, "outcome_unknown")
	s.Equal(store.InvocationOutcomeUnknown, s.logged(app, 1)[0].ErrorType)
}

func (s *InvocationLogSuite) TestAnInterruptedCallLeavesAnOutcomeUnknownRow() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "slow")
	binding.TimeoutMs = 30000
	d, _, _, err := s.attach(s.spec(s.config(binding), "", nil))
	s.Require().NoError(err)
	turn, interrupt := context.WithCancel(s.ctx)
	time.AfterFunc(200*time.Millisecond, interrupt)

	_, err = d.Run(turn, llm.ToolCall{ID: uuid.NewString(), Name: "crm__slow", Arguments: "{}"})

	s.ErrorIs(err, context.Canceled)
	s.Equal(store.InvocationOutcomeUnknown, s.logged(app, 1)[0].ErrorType)
}

// TestACredentialTheProviderRefusesLeavesACustomerAuthRow: the connection's token stops being
// the account's after the session opened, and the provider answers 401.
func (s *InvocationLogSuite) TestACredentialTheProviderRefusesLeavesACustomerAuthRow() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	wrong, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: tokenOf("secondary")}})
	s.Require().NoError(err)
	s.setState(app, func(state *core.CredentialState) { state.Credentials = wrong })

	_, err = s.call(d, "crm__whoami", "{}")

	s.Require().Error(err)
	s.Equal(store.InvocationCustomerAuth, s.logged(app, 1)[0].ErrorType)
}

// TestAnIncognitoSessionsCallIsLoggedWithoutItsSession: the decision this PR makes for
// incognito. The row says the connection's credential was used; nothing in it names the
// conversation. No session's row holds what the call was asked or answered.
func (s *InvocationLogSuite) TestAnIncognitoSessionsCallIsLoggedWithoutItsSession() {
	app := s.connection("", "primary")
	spec := s.spec(s.config(s.fixed("crm", app, "echo")), "", nil)
	spec.ID, spec.Incognito = uuid.NewString(), true
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	note := "note-" + uuid.NewString()

	said, err := s.call(d, "crm__echo", `{"note": "`+note+`"}`)

	s.Require().NoError(err)
	s.Equal("you said "+note, said)
	rows := s.logged(app, 1)
	s.Empty(rows[0].SessionID)
	stored := s.stored(app)
	s.NotContains(stored, note, "neither the argument nor the result is stored")
	s.NotContains(stored, spec.ID)
}

func (s *InvocationLogSuite) TestARecordedSessionsRowHoldsNoArgumentOrResultEither() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "echo")), "", nil))
	s.Require().NoError(err)
	note := "note-" + uuid.NewString()

	_, err = s.call(d, "crm__echo", `{"note": "`+note+`"}`)

	s.Require().NoError(err)
	s.logged(app, 1)
	s.NotContains(s.stored(app), note)
}

// TestANameItDoesNotOwnLeavesNoRow: a call the rest of the session answers is not a
// connector's, so there is no connection to log it against.
func (s *InvocationLogSuite) TestANameItDoesNotOwnLeavesNoRow() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	d.next = &answering{}

	_, err = s.call(d, "lookup_order", "{}")

	s.Require().NoError(err)
	s.Empty(s.logged(app, 0))
}

// TestTheLogDoesNotHoldUpTheCall: the table is locked, so the write waits; the call does not.
func (s *InvocationLogSuite) TestTheLogDoesNotHoldUpTheCall() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	held, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = held.Rollback() }()
	_, err = held.ExecContext(s.ctx, "LOCK TABLE connector_invocations IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)
	started := time.Now()

	said, err := s.call(d, "crm__whoami", "{}")
	took := time.Since(started)

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Less(took, 2*time.Second, "the call came back while its row could not be written")
	s.Never(func() bool {
		var written int
		s.Require().NoError(held.QueryRowContext(s.ctx, "SELECT count(*) FROM connector_invocations WHERE connection_id = ?", app).Scan(&written))
		return written > 0
	}, 300*time.Millisecond, 50*time.Millisecond, "the write was held by the lock")
	s.Require().NoError(held.Commit())
	s.Len(s.logged(app, 1), 1, "the row is written once the table is free")
}

// TestAWriteThatFailsDoesNotFailTheCall: the writer's database is gone; the call still answers.
func (s *InvocationLogSuite) TestAWriteThatFailsDoesNotFailTheCall() {
	app := s.connection("", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	failing := newInvocationRecorder(closed, slog.New(slog.DiscardHandler))
	s.T().Cleanup(failing.Close)
	d.invocations = failing

	said, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Empty(s.logged(app, 0))
}

// TestASessionWithNoBindingsOpensNoDispatcherAndLogsNothing: nothing configured, nothing
// written, as on base.
func (s *InvocationLogSuite) TestASessionWithNoBindingsOpensNoDispatcherAndLogsNothing() {
	app := s.connection("", "primary")

	d, tools, unavailable, err := s.attach(s.spec(s.config(), "", nil))

	s.Require().NoError(err)
	s.Nil(d)
	s.Empty(tools)
	s.Empty(unavailable)
	s.Empty(s.logged(app, 0))
}
