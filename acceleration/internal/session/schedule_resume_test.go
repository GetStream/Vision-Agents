//go:build integration

package session

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ScheduleResumeSuite is the agent resuming a task with update_scheduled_task
// on a clock of the test's, over Postgres: what a resume counts from is the
// runner's now.
type ScheduleResumeSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
}

func TestScheduleResumeSuite(t *testing.T) {
	suite.Run(t, new(ScheduleResumeSuite))
}

func (s *ScheduleResumeSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "schedule_resume"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *ScheduleResumeSuite) TestAOneShotResumedInsideTheFallBackHourKeepsTheSecondPassItIsDueAt() {
	runner, task := s.pausedInTheSecondPass()
	// 02:50 CEST, the first pass: 02:20 CEST is behind, 02:20 CET an hour and a half ahead.
	resumed := time.Date(2026, 10, 25, 0, 50, 0, 0, time.UTC)
	runner.now = func() time.Time { return resumed }

	parts, err := runner.Run(s.ctx, llm.ToolCall{ID: uuid.NewString(), Name: "update_scheduled_task",
		Arguments: `{"id":"` + task.ID + `","action":"resume"}`})

	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	var answer map[string]any
	s.Require().NoError(json.Unmarshal([]byte(parts[0].Text), &answer), parts[0].Text)
	s.NotContains(answer, "state", "an active task's answer names no state: %s", parts[0].Text)
	s.NotContains(answer, "missed", parts[0].Text)
	stored, err := s.store.ScheduledTask(s.ctx, task.CustomerID, task.ID)
	s.Require().NoError(err)
	s.Equal(store.ScheduledTaskActive, stored.State)
	s.Require().NotNil(stored.NextRunAt)
	s.True(task.NextRunAt.Equal(*stored.NextRunAt), "due at the second pass, %s, not %s", task.NextRunAt, stored.NextRunAt)
	runs, err := s.store.ScheduledTaskRuns(s.ctx, task.CustomerID, task.ID, 25, nil)
	s.Require().NoError(err)
	s.Empty(runs, "nothing was slept through")
}

// pausedInTheSecondPass is a runner for a user of a new agent config, and a
// paused task of theirs stored as one made at 01:50 CEST on 2026-10-25 to run
// an hour and a half later is: at 02:20 Berlin, the second pass of the hour
// the clocks go back, 01:20Z.
func (s *ScheduleResumeSuite) pausedInTheSecondPass() (*scheduleRunner, store.ScheduledTask) {
	customer, user := "resume-"+uuid.NewString(), "user-"+uuid.NewString()
	config := store.AgentConfig{CustomerID: customer, Name: "agent-" + uuid.NewString()}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	due := time.Date(2026, 10, 25, 1, 20, 0, 0, time.UTC)
	task := store.ScheduledTask{
		CustomerID: customer, ConfigID: config.ID, UserID: user, CallerKind: string(auth.KindAuthenticated),
		Title: "Stretch", Instructions: "Remind them to stretch.", CreatedBy: "agent",
		StartLocal: time.Date(2026, 10, 25, 2, 20, 0, 0, time.UTC), Timezone: "Europe/Berlin",
		NextRunAt: &due, State: store.ScheduledTaskPaused, Channels: []string{"session"},
	}
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &task, 10))
	stored, err := s.store.ScheduledTask(s.ctx, customer, task.ID)
	s.Require().NoError(err)
	s.Require().Equal(store.ScheduledTaskPaused, stored.State)
	s.Require().True(due.Equal(*stored.NextRunAt))
	spec := Spec{CustomerID: customer, ConfigID: config.ID}
	runner := &scheduleRunner{
		db: s.store, spec: spec, user: user, kind: auth.KindAuthenticated, now: time.Now,
		logger: slog.New(slog.DiscardHandler), owned: map[string]bool{"update_scheduled_task": true},
	}
	return runner, task
}
