//go:build integration

package store

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// scheduledTasksVersion is 20261105120000_scheduled_tasks.sql. It comes after
// cancelled_request_log (20261104120000), accelerate's newest when the branch merged it,
// which staging has applied.
const scheduledTasksVersion int64 = 20261105120000

// stagingVersion is the version staging is at, the one the migration must apply on top of.
const stagingVersion int64 = 20261104120000

// scheduledTaskIndexes are the indexes the migration creates, N14's two included.
var scheduledTaskIndexes = []string{
	"channel_deliveries_from_idx",
	"scheduled_task_runs_occurrence_idx",
	"scheduled_task_runs_pkey",
	"scheduled_task_runs_task_idx",
	"scheduled_tasks_due_idx",
	"scheduled_tasks_origin_session_idx",
	"scheduled_tasks_pkey",
	"scheduled_tasks_session_idx",
	"scheduled_tasks_user_idx",
}

// ScheduledTasksMigrationSuite stops a database of its own at the version before the
// scheduled tasks, then applies, reverts and reapplies them.
type ScheduledTasksMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestScheduledTasksMigrationSuite(t *testing.T) {
	suite.Run(t, new(ScheduledTasksMigrationSuite))
}

func (s *ScheduledTasksMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "stmig")
	s.ctx = context.Background()
}

func (s *ScheduledTasksMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "stmig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", stagingVersion))
	s.db = db
}

// The migration applies on top of the newest base version, comes off leaving nothing of
// its own behind, and applies again.
func (s *ScheduledTasksMigrationSuite) TestTheMigrationAppliesComesOffAndAppliesAgain() {
	s.Empty(s.indexes())

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", scheduledTasksVersion))
	s.Equal(scheduledTaskIndexes, s.indexes())
	s.Equal(4, s.addedColumns())

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", stagingVersion))
	s.Empty(s.indexes())
	s.Zero(s.addedColumns())

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", scheduledTasksVersion))
	s.Equal(scheduledTaskIndexes, s.indexes())
}

// indexes is the names of the migration's indexes that exist, in order.
func (s *ScheduledTasksMigrationSuite) indexes() []string {
	rows, err := s.db.DB().QueryContext(s.ctx, `SELECT indexname FROM pg_indexes
WHERE tablename IN ('scheduled_tasks', 'scheduled_task_runs')
   OR indexname = 'channel_deliveries_from_idx'
ORDER BY indexname`)
	s.Require().NoError(err)
	defer rows.Close()
	var names []string
	for rows.Next() {
		var name string
		s.Require().NoError(rows.Scan(&name))
		names = append(names, name)
	}
	s.Require().NoError(rows.Err())
	return names
}

// addedColumns counts the columns the migration adds to tables that were there before it.
func (s *ScheduledTasksMigrationSuite) addedColumns() int {
	var count int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE (table_name, column_name) IN (('users', 'timezone'), ('agent_configs', 'scheduling'),
	('channel_deliveries', 'from_address'), ('channel_identities', 'thread'))`).Scan(&count))
	return count
}

// wakeVersion is 20261105130000_scheduled_tasks_wake.sql, the index NextDueScheduledTask reads.
const wakeVersion int64 = 20261105130000

func (s *ScheduledTasksMigrationSuite) TestTheWakeIndexAppliesComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", wakeVersion))
	s.Contains(s.indexes(), "scheduled_tasks_wake_idx")

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", scheduledTasksVersion))
	s.Equal(scheduledTaskIndexes, s.indexes())

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", wakeVersion))
	s.Contains(s.indexes(), "scheduled_tasks_wake_idx")
}
