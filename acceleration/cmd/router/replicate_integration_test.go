//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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
	task := &store.ScheduledTask{
		ID: id, CustomerID: "acme", ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Stretch", Instructions: "Remind Sam to stretch.", StartLocal: next, Timezone: "UTC",
		State: store.ScheduledTaskActive, NextRunAt: &next, CreatedBy: "api",
	}
	s.Require().NoError(s.store.CreateScheduledTask(s.ctx, task, 25))
	var row map[string]any
	var raw []byte
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT to_jsonb(t) FROM scheduled_tasks t WHERE id = ?", id).Scan(&raw))
	s.Require().NoError(json.Unmarshal(raw, &row))
	row["state"] = state
	encoded, err := json.Marshal(row)
	s.Require().NoError(err)
	s.Require().NoError(s.store.ImportRow(s.ctx, "acme", "scheduled_tasks", encoded))
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
