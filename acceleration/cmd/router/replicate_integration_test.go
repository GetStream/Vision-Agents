//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ReplicateResumeSuite runs the cut-over of `router replicate` on a database of its own,
// emptied once and migrated the way the command opens it, holding rows a copy imported.
type ReplicateResumeSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	store    *store.Store
}

func TestReplicateResumeSuite(t *testing.T) {
	suite.Run(t, new(ReplicateResumeSuite))
}

func (s *ReplicateResumeSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "resume")

	empty, err := store.Open(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	defer empty.Close()
	var database string
	s.Require().NoError(empty.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = empty.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)

	s.store, err = openStore(s.ctx, s.settings)
	s.Require().NoError(err)
}

func (s *ReplicateResumeSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *ReplicateResumeSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE scheduled_tasks, campaigns, agent_configs CASCADE")
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-1', 'acme', 'a', 'be brief')")
	s.Require().NoError(err)
}

// imported writes a task as a copy does, in the state it had where it came from.
func (s *ReplicateResumeSuite) imported(id, state string) {
	next := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: id, StartLocal: next, NextRunAt: &next}, state)
}

// importedTask is imported for a task of acme's own shape: its id, schedule and next run,
// and its customer and config when it is not acme's config-1.
func (s *ReplicateResumeSuite) importedTask(task store.ScheduledTask, state string) {
	if task.CustomerID == "" {
		task.CustomerID, task.ConfigID = "acme", "config-1"
	}
	task.UserID, task.CallerKind, task.Title, task.Instructions = "sam", "authenticated", "Stretch", "Remind Sam to stretch."
	task.Timezone, task.State, task.CreatedBy = "UTC", store.ScheduledTaskActive, "api"
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &task, 25))
	var row map[string]any
	var raw []byte
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT to_jsonb(t) FROM scheduled_tasks t WHERE id = ?", task.ID).Scan(&raw))
	s.Require().NoError(json.Unmarshal(raw, &row))
	row["state"] = state
	encoded, err := json.Marshal(row)
	s.Require().NoError(err)
	s.Require().NoError(s.store.ImportRow(s.ctx, task.CustomerID, "scheduled_tasks", encoded))
}

// overdue imports a task the source had left active, due a minute ago: the source went on
// delivering after the last catch-up and delivered that occurrence before it stopped.
func (s *ReplicateResumeSuite) overdue(id, rrule, state string) time.Time {
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: id, RRule: rrule, StartLocal: due, NextRunAt: &due}, state)
	return due
}

// task reads a task here as the worker and the API read it.
func (s *ReplicateResumeSuite) task(id string) store.ScheduledTask {
	task, err := s.store.ScheduledTask(s.ctx, "acme", id)
	s.Require().NoError(err)
	return task
}

// claimed is the tasks this deployment's worker would deliver now.
func (s *ReplicateResumeSuite) claimed() []string {
	claims, err := s.store.ClaimDueScheduledTasks(s.ctx, 10, time.Minute)
	s.Require().NoError(err)
	var ids []string
	for _, claim := range claims {
		ids = append(ids, claim.Task.ID)
	}
	return ids
}

// cutOver runs the cut-over and returns what it printed.
func (s *ReplicateResumeSuite) cutOver() string {
	var out bytes.Buffer
	s.Require().NoError(resumeMove(s.ctx, s.store, "acme", &out))
	return out.String()
}

func (s *ReplicateResumeSuite) state(id string) string {
	task, err := s.store.ScheduledTask(s.ctx, "acme", id)
	s.Require().NoError(err)
	return task.State
}

func (s *ReplicateResumeSuite) TestResumeResumesWhatTheCopyPausedAndNotWhatTheUserDid() {
	s.imported("st-active", store.ScheduledTaskActive)
	s.imported("st-paused", store.ScheduledTaskPaused)
	s.Require().Equal(store.ScheduledTaskPaused, s.state("st-active"), "the copy waits for the cut-over")

	s.Require().NoError(runReplicate([]string{"--resume", "--as", "acme"}, s.settings, slog.New(slog.DiscardHandler)))

	s.Equal(store.ScheduledTaskActive, s.state("st-active"))
	s.Equal(store.ScheduledTaskPaused, s.state("st-paused"))
}

func (s *ReplicateResumeSuite) TestResumeWithoutTheAppItLandedUnderResumesNothing() {
	s.imported("st-active", store.ScheduledTaskActive)

	err := runReplicate([]string{"--resume"}, s.settings, slog.New(slog.DiscardHandler))

	s.Require().Error(err)
	s.Contains(err.Error(), "--resume --as <app id>")
	s.Equal(store.ScheduledTaskPaused, s.state("st-active"))
}

func (s *ReplicateResumeSuite) TestResumeRefusesToCatchUpFromTheSourceFirst() {
	s.imported("st-active", store.ScheduledTaskActive)

	err := runReplicate([]string{"--resume", "--as", "acme", "--from", "http://127.0.0.1:1"}, s.settings, slog.New(slog.DiscardHandler))

	s.Require().Error(err)
	s.Contains(err.Error(), "--resume --as <app id>")
	s.Equal(store.ScheduledTaskPaused, s.state("st-active"))
}

func (s *ReplicateResumeSuite) TestResumeNamesTheCampaignsToStartHere() {
	campaign := &store.Campaign{CustomerID: "acme", Name: "Renewals", ConfigID: "config-1", FromNumber: "+15550100"}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, campaign))
	var raw []byte
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT to_jsonb(c) || '{\"state\": \"running\"}' FROM campaigns c WHERE id = ?", campaign.ID).Scan(&raw))
	s.Require().NoError(s.store.ImportRow(s.ctx, "acme", "campaigns", raw))
	s.imported("st-active", store.ScheduledTaskActive)

	var out bytes.Buffer
	s.Require().NoError(resumeMove(s.ctx, s.store, "acme", &out))

	s.Equal("resumed 1 scheduled tasks\n"+
		"campaign "+campaign.ID+" was running where it came from: start it here with POST /v1/agents/campaigns/"+campaign.ID+"/start\n",
		out.String())
}

func (s *ReplicateResumeSuite) TestResumeSaysHowManyTasksItLeftPausedForADeletedConfig() {
	s.imported("st-active", store.ScheduledTaskActive)
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme", "config-1"))

	var out bytes.Buffer
	s.Require().NoError(resumeMove(s.ctx, s.store, "acme", &out))

	s.Equal("resumed 0 scheduled tasks\n"+
		"left 1 scheduled tasks paused: their agent config is deleted or missing here\n", out.String())
	s.Equal(store.ScheduledTaskPaused, s.state("st-active"))
}

func (s *ReplicateResumeSuite) TestResumeSaysHowManyTasksItLeftPausedForWantOfRoom() {
	_, err := s.store.DB().ExecContext(s.ctx,
		`UPDATE agent_configs SET scheduling = '{"max_tasks_per_user": 1}' WHERE id = 'config-1'`)
	s.Require().NoError(err)
	next := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, &store.ScheduledTask{
		ID: "st-here", CustomerID: "acme", ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Hydrate", Instructions: "Remind Sam to drink.", StartLocal: next, Timezone: "UTC",
		NextRunAt: &next, CreatedBy: "api",
	}, 25))
	s.imported("st-1", store.ScheduledTaskActive)
	s.imported("st-2", store.ScheduledTaskActive)

	var out bytes.Buffer
	s.Require().NoError(resumeMove(s.ctx, s.store, "acme", &out))

	s.Equal("resumed 0 scheduled tasks\n"+
		"left 2 scheduled tasks paused: their user already holds the most active tasks the agent config allows. "+
		"Run --resume again once the user has made room; the oldest are resumed first\n", out.String())
	s.Equal(store.ScheduledTaskPaused, s.state("st-1"))
}

func (s *ReplicateResumeSuite) TestResumeDoesNotDeliverAgainARepeatsOccurrenceThatCameBeforeIt() {
	due := s.overdue("st-daily", "FREQ=DAILY", store.ScheduledTaskActive)

	s.Equal("resumed 1 scheduled tasks\n", s.cutOver())

	task := s.task("st-daily")
	s.Equal(store.ScheduledTaskActive, task.State)
	s.Require().NotNil(task.NextRunAt)
	s.Equal(due.Add(24*time.Hour), task.NextRunAt.UTC(), "it goes on from its first occurrence after the cut-over")
	s.Empty(s.claimed(), "the occurrence before the cut-over was the source's")
}

func (s *ReplicateResumeSuite) TestResumeClosesAOneShotWhoseTimeCameBeforeIt() {
	// Each call reads the clock on its own, so each task keeps its own time.
	due := map[string]time.Time{
		"st-once":  s.overdue("st-once", "", store.ScheduledTaskActive),
		"st-twice": s.overdue("st-twice", "", store.ScheduledTaskActive),
	}
	// The prune deletes a completed task by when it last changed, read on the database's clock.
	var before time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT now()").Scan(&before))

	s.Equal("resumed 0 scheduled tasks\n", s.cutOver())

	for _, id := range []string{"st-once", "st-twice"} {
		task := s.task(id)
		s.Equal(store.ScheduledTaskCompleted, task.State, id)
		s.False(task.UpdatedAt.Before(before), "%s was completed at the cut-over, so its history is kept from then", id)
		s.Nil(task.NextRunAt, id)
		s.False(task.MovePaused, id)
		var state, reason string
		var scheduledFor, created time.Time
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT state, error, scheduled_for, created_at FROM scheduled_task_runs WHERE task_id = ?", id).
			Scan(&state, &reason, &scheduledFor, &created), id)
		s.Equal(store.ScheduledTaskRunMissed, state, id)
		s.Equal(scheduling.ReasonPaused, reason, id)
		s.Equal(due[id], scheduledFor.UTC(), "the time it slept through here is on record")
		s.WithinDuration(time.Now(), created, time.Hour, id)
	}
	s.Empty(s.claimed())

	s.Equal("resumed 0 scheduled tasks\n", s.cutOver(), "a second cut-over finds nothing left")
	var runs int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM scheduled_task_runs WHERE task_id = 'st-once'").Scan(&runs))
	s.Equal(1, runs)
}

// The source had claimed the occurrence when the last catch-up read it, so its run came
// across while the task's next run had not moved yet.
func (s *ReplicateResumeSuite) TestResumeClosesAOneShotWhoseRunTheSourceStartedBeforeTheLastCatchUp() {
	due := s.overdue("st-once", "", store.ScheduledTaskActive)
	_, err := s.store.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, state, started_at)
VALUES ('run-source', 'st-once', 'acme', ?, 'running', now())`, due)
	s.Require().NoError(err)

	s.Equal("resumed 0 scheduled tasks\n", s.cutOver())

	s.Equal(store.ScheduledTaskCompleted, s.task("st-once").State)
	var id string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT string_agg(id, ',') FROM scheduled_task_runs WHERE task_id = 'st-once'").Scan(&id))
	s.Equal("run-source", id, "the source's run of the occurrence is the one on record")
}

// The source held the occurrence for quiet hours past the cut-over, so it never delivered it.
func (s *ReplicateResumeSuite) TestResumeLeavesAnOccurrenceTheSourceHeldPastItToTheWorker() {
	for name, rrule := range map[string]string{"daily": "FREQ=DAILY", "one-shot": ""} {
		s.Run(name, func() {
			s.SetupTest()
			// Due an hour ago, so its run is old enough for the worker to take over.
			due := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
			s.importedTask(store.ScheduledTask{ID: "st-held", RRule: rrule, StartLocal: due, NextRunAt: &due}, store.ScheduledTaskActive)
			_, err := s.store.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, state, started_at, postponed_until, attempts)
VALUES ('run-held', 'st-held', 'acme', ?, 'running', ?, ?, 1)`, due, due, time.Now().UTC().Add(8*time.Hour))
			s.Require().NoError(err)

			s.Equal("resumed 1 scheduled tasks\n", s.cutOver())

			task := s.task("st-held")
			s.Equal(store.ScheduledTaskActive, task.State)
			s.Require().NotNil(task.NextRunAt)
			s.Equal(due, task.NextRunAt.UTC(), "the held occurrence is still to deliver")
			claims, err := s.store.ClaimDueScheduledTasks(s.ctx, 10, time.Minute)
			s.Require().NoError(err)
			var runs []string
			for _, claim := range claims {
				runs = append(runs, claim.Run.ID)
			}
			s.Equal([]string{"run-held"}, runs, "the worker here takes the held run over")
		})
	}
}

func (s *ReplicateResumeSuite) TestResumeMovesOnlyTheOverdueTaskOfTheCustomer() {
	s.overdue("st-daily", "FREQ=DAILY", store.ScheduledTaskActive)
	s.overdue("st-once", "", store.ScheduledTaskActive)
	ahead := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: "st-ahead", RRule: "FREQ=WEEKLY", StartLocal: ahead, NextRunAt: &ahead}, store.ScheduledTaskActive)

	s.Equal("resumed 2 scheduled tasks\n", s.cutOver())

	task := s.task("st-ahead")
	s.Equal(store.ScheduledTaskActive, task.State, "the one-shot's completion is its own")
	s.Require().NotNil(task.NextRunAt)
	s.Equal(ahead, task.NextRunAt.UTC(), "the daily task's skip is its own")
}

// A daily task two runs behind: the source delivered both before it stopped.
func (s *ReplicateResumeSuite) TestResumeSkipsEveryOccurrenceThatCameBeforeIt() {
	start := time.Now().UTC().Add(-47 * time.Hour).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: "st-daily", RRule: "FREQ=DAILY", StartLocal: start, NextRunAt: &start}, store.ScheduledTaskActive)

	s.Equal("resumed 1 scheduled tasks\n", s.cutOver())

	task := s.task("st-daily")
	s.Require().NotNil(task.NextRunAt)
	s.Equal(start.Add(48*time.Hour), task.NextRunAt.UTC(), "it goes on from its first occurrence after the cut-over")
	s.Empty(s.claimed())
}

func (s *ReplicateResumeSuite) TestResumeCompletesARepeatWithNoOccurrenceLeft() {
	s.overdue("st-last", "FREQ=DAILY;COUNT=1", store.ScheduledTaskActive)

	s.Equal("resumed 0 scheduled tasks\n", s.cutOver())

	task := s.task("st-last")
	s.Equal(store.ScheduledTaskCompleted, task.State)
	s.Nil(task.NextRunAt)
	s.Empty(s.claimed())
	var runs int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM scheduled_task_runs WHERE task_id = 'st-last'").Scan(&runs))
	s.Zero(runs, "a pause skips what a repeat sleeps through")
}

func (s *ReplicateResumeSuite) TestResumeKeepsANextRunStillAheadAsTheSourceSetIt() {
	// Snoozed at the source: its next run is an hour ahead, not the schedule's next occurrence.
	start := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	snoozed := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: "st-snoozed", RRule: "FREQ=DAILY", StartLocal: start, NextRunAt: &snoozed}, store.ScheduledTaskActive)

	s.Equal("resumed 1 scheduled tasks\n", s.cutOver())

	task := s.task("st-snoozed")
	s.Equal(store.ScheduledTaskActive, task.State)
	s.Require().NotNil(task.NextRunAt)
	s.Equal(snoozed, task.NextRunAt.UTC())
}

func (s *ReplicateResumeSuite) TestResumeLeavesAnOverdueTaskTheUserPausedAsItIs() {
	due := s.overdue("st-paused", "FREQ=DAILY", store.ScheduledTaskPaused)

	s.Equal("resumed 0 scheduled tasks\n", s.cutOver())

	task := s.task("st-paused")
	s.Equal(store.ScheduledTaskPaused, task.State)
	s.Require().NotNil(task.NextRunAt)
	s.Equal(due, task.NextRunAt.UTC(), "a resume by the user counts its next run, not the cut-over")
}

func (s *ReplicateResumeSuite) TestResumeLeavesAMovedTaskResumedHereByHandToTheWorker() {
	due := s.overdue("st-resumed", "FREQ=DAILY", store.ScheduledTaskActive)
	// Resumed here by hand, keeping the mark the move left.
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET state = 'active' WHERE id = 'st-resumed'")
	s.Require().NoError(err)

	s.cutOver()

	task := s.task("st-resumed")
	s.Require().NotNil(task.NextRunAt)
	s.Equal(due, task.NextRunAt.UTC(), "an occurrence due after it was resumed here is this deployment's")
}

func (s *ReplicateResumeSuite) TestResumeLeavesAnotherCustomersMoveAlone() {
	_, err := s.store.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-2', 'other', 'a', 'be brief')")
	s.Require().NoError(err)
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	s.importedTask(store.ScheduledTask{ID: "st-other", CustomerID: "other", ConfigID: "config-2", RRule: "FREQ=DAILY",
		StartLocal: due, NextRunAt: &due}, store.ScheduledTaskActive)

	s.cutOver()

	task, err := s.store.ScheduledTask(s.ctx, "other", "st-other")
	s.Require().NoError(err)
	s.Require().NotNil(task.NextRunAt)
	s.Equal(due, task.NextRunAt.UTC(), "another customer's move is cut over on its own")
}

// A pause here in flight when the cut-over reads the task is waited for, not written over.
func (s *ReplicateResumeSuite) TestResumeWaitsForAPauseOfTheTaskInFlight() {
	due := s.overdue("st-daily", "FREQ=DAILY", store.ScheduledTaskActive)
	pausing, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = pausing.Rollback() }()
	_, err = pausing.ExecContext(s.ctx, "UPDATE scheduled_tasks SET move_paused = false WHERE id = 'st-daily'")
	s.Require().NoError(err)

	done := make(chan error, 1)
	go func() { done <- resumeMove(s.ctx, s.store, "acme", io.Discard) }()
	// Commit the pause once the cut-over waits on it; a cut-over that does not wait has
	// finished by then.
	deadline := time.Now().Add(30 * time.Second)
	var result error
	finished := false
	for !finished && time.Now().Before(deadline) {
		select {
		case result = <-done:
			finished = true
		case <-time.After(50 * time.Millisecond):
			var waiting int
			s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
				"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
			if waiting > 0 {
				s.Require().NoError(pausing.Commit())
				result, finished = <-done, true
			}
		}
	}
	s.Require().True(finished, "the cut-over neither finished nor waited")
	s.Require().NoError(result)
	task := s.task("st-daily")
	s.Equal(store.ScheduledTaskPaused, task.State)
	s.Require().NotNil(task.NextRunAt)
	s.Equal(due, task.NextRunAt.UTC(), "the user paused it before the cut-over could move it")
}
