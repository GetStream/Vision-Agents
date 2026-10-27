//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RouterStartSuite starts the router as its serve command does, on a database of the suite's
// own, and stops it with the signal a deployment sends.
type RouterStartSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	db       *store.Store
}

func TestRouterStartSuite(t *testing.T) {
	suite.Run(t, new(RouterStartSuite))
}

func (s *RouterStartSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "start")
	s.settings.Addr = "127.0.0.1:0"
	// api_key, the default, wants a keyring to seal the keys it issues; a router of no
	// customer needs none.
	s.settings.Auth.Mode = string(auth.NoAuth)
	// The router migrates the database itself; opening it here first is what lets the test
	// store the rows it ages before the router starts.
	db, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.db = db
}

// The prunes and the redaction run as the router starts, not a day, an hour or ten minutes later.
func (s *RouterStartSuite) TestAStartingRouterPrunesAndRedactsWhatIsPastItsRetention() {
	customer := "customer-" + uuid.NewString()
	next := time.Now().UTC().Add(24 * time.Hour).Truncate(time.Second)
	task := store.ScheduledTask{
		CustomerID: customer, ConfigID: "config-1", UserID: "user-1",
		CallerKind: store.UserKindAuthenticated, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: next, Timezone: "UTC", NextRunAt: &next, State: store.ScheduledTaskActive, CreatedBy: "agent",
	}
	s.Require().NoError(s.db.CreateScheduledTask(s.ctx, &task, 25))
	s.T().Cleanup(func() {
		s.NoError(s.db.DeleteScheduledTask(context.Background(), customer, task.ID))
	})
	_, err := s.db.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, trigger, state, finished_at)
VALUES (?, ?, ?, now() - interval '91 days', 'schedule', 'delivered', now() - make_interval(secs => ?))`,
		uuid.NewString(), task.ID, customer, (store.ScheduledTasksKept + 24*time.Hour).Seconds())
	s.Require().NoError(err)
	_, err = s.db.DB().ExecContext(s.ctx, `
INSERT INTO data_changes (customer_id, table_name, op, key, at)
VALUES (?, 'agent_configs', 'insert', '{}', now() - make_interval(secs => ?))`,
		customer, (s.settings.DataMove.Retention + 24*time.Hour).Seconds())
	s.Require().NoError(err)
	session := &store.AgentSession{ID: uuid.NewString(), CustomerID: customer, AgentName: "docs", ConfigID: "config-1",
		UserID: "user-1", CallerKind: store.UserKindAuthenticated, State: store.SessionClosed, CreatedAt: time.Now().UTC()}
	s.Require().NoError(s.db.SaveSession(s.ctx, session))
	response := uuid.NewString()
	s.Require().NoError(s.db.StartResponse(s.ctx, &store.AgentResponse{
		ID: response, SessionID: session.ID, CustomerID: customer, Said: "where is my order?", CreatedAt: time.Now().UTC(),
	}))
	due := time.Now().UTC().Add(-time.Hour)
	s.Require().NoError(s.db.AppendResponseItems(s.ctx, []store.AgentResponseItem{{
		ResponseID: response, SessionID: session.ID, Kind: store.ItemToolResult, ToolName: "orders",
		Text: `{"email":"someone@example.com"}`, At: time.Now().UTC(), RedactAt: &due,
	}}))
	left := func(query string) int {
		var n int
		s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, query, customer).Scan(&n))
		return n
	}
	const (
		runs       = "SELECT count(*) FROM scheduled_task_runs WHERE customer_id = ?"
		changes    = "SELECT count(*) FROM data_changes WHERE customer_id = ?"
		unredacted = "SELECT count(*) FROM agent_response_items i JOIN agent_responses r ON r.id = i.response_id " +
			"WHERE r.customer_id = ? AND i.redact_at IS NOT NULL"
	)
	s.Require().Equal([]int{1, 1, 1}, []int{left(runs), left(changes), left(unredacted)})

	stopped := make(chan error, 1)
	go func() { stopped <- run(s.settings, slog.New(slog.DiscardHandler)) }()

	pruned := func() bool {
		return left(runs) == 0 && left(changes) == 0 && left(unredacted) == 0
	}
	deadline := time.After(30 * time.Second)
	for !pruned() {
		select {
		case err := <-stopped:
			s.Require().Failf("the router stopped before it was told to", "error: %v", err)
		case <-deadline:
			s.Require().Failf("what is past its retention is still there",
				"scheduled task runs %d, recorded changes %d, unredacted tool results %d", left(runs), left(changes), left(unredacted))
		case <-time.After(50 * time.Millisecond):
		}
	}

	// run is waiting on the signal it asked for, so the signal stops it rather than the test.
	s.Require().NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
	select {
	case err := <-stopped:
		s.Require().NoError(err)
	case <-time.After(30 * time.Second):
		s.Require().Fail("the router did not stop on SIGTERM")
	}
}
