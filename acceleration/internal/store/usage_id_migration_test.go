//go:build integration

package store

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// usageIDVersion is 20261110135000_requests_usage_id.sql. beforeUsageIDVersion is the one
// before it, 20261110130000_price_entries.sql, which the router before it runs at.
const (
	usageIDVersion       int64 = 20261110135000
	beforeUsageIDVersion int64 = 20261110130000
)

// UsageIDMigrationSuite stops a database of its own at the version before usage ids, with a
// row the router of then wrote, so the migration is applied over what a deployment has.
type UsageIDMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
	// old is the id of the row written before the migration.
	old int64
}

func TestUsageIDMigrationSuite(t *testing.T) {
	suite.Run(t, new(UsageIDMigrationSuite))
}

func (s *UsageIDMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "usagemig")
	s.ctx = context.Background()
}

func (s *UsageIDMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "usagemig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeUsageIDVersion))
	s.db = db
	s.old = s.writeAsTheRouterBefore()
}

// writeAsTheRouterBefore writes a row with the columns the router before usage ids writes,
// none of which this migration adds, and returns its id.
func (s *UsageIDMigrationSuite) writeAsTheRouterBefore() int64 {
	var id int64
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `INSERT INTO requests
(modality, customer_id, agent_id, provider, model, started_at, audio_ms, characters, input_tokens,
 cached_input_tokens, output_tokens, images, meters, attributes, cost_micros, output_cost_micros, billable, success)
VALUES ('stt', 'acme', 'agent', 'deepgram', 'flux', now(), 1000, 0, 0, 0, 0, 0, '{}', '{}', 25, 0, true, true)
RETURNING id`).Scan(&id))
	return id
}

func (s *UsageIDMigrationSuite) migrate() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", usageIDVersion))
}

// usageIDOf is the usage id of the row with that id, empty when it has none.
func (s *UsageIDMigrationSuite) usageIDOf(id int64) string {
	var usageID *string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT usage_id::text FROM requests WHERE id = ?", id).Scan(&usageID))
	if usageID == nil {
		return ""
	}
	return *usageID
}

// code is the SQLSTATE Postgres refused err with, empty for any other error.
func code(err error) string {
	var refused pgdriver.Error
	if errors.As(err, &refused) {
		return refused.Field('C')
	}
	return ""
}

func (s *UsageIDMigrationSuite) TestTheRouterBeforeStillWritesItsRowsWithAUsageID() {
	s.migrate()

	first, second := s.writeAsTheRouterBefore(), s.writeAsTheRouterBefore()

	s.NotEmpty(s.usageIDOf(first), "a rolling deploy's old pods, and a rollback, keep recording")
	s.NotEqual(s.usageIDOf(first), s.usageIDOf(second))
}

func (s *UsageIDMigrationSuite) TestARowFromBeforeKeepsNoUsageIDAndTheTableIsNotRewritten() {
	var before, after int64
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT pg_relation_filenode('requests')").Scan(&before))

	s.migrate()

	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT pg_relation_filenode('requests')").Scan(&after))
	s.Equal(before, after, "neither the columns nor the default rewrote the table")
	s.Empty(s.usageIDOf(s.old))
}

func (s *UsageIDMigrationSuite) TestAnUpdateOfARowFromBeforeMustGiveItAUsageIDOrSkipIt() {
	s.migrate()

	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE requests SET to_ship = true WHERE id = ?", s.old)
	s.Equal("23514", code(err), "the check holds an updated row to it too: %v", err)

	skipped, err := s.db.DB().ExecContext(s.ctx, "UPDATE requests SET to_ship = true WHERE id = ? AND usage_id IS NOT NULL", s.old)
	s.Require().NoError(err)
	affected, err := skipped.RowsAffected()
	s.Require().NoError(err)
	s.Zero(affected)

	_, err = s.db.DB().ExecContext(s.ctx, "UPDATE requests SET usage_id = gen_random_uuid() WHERE id = ?", s.old)
	s.Require().NoError(err)
	s.NotEmpty(s.usageIDOf(s.old))
}

func (s *UsageIDMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.migrate()
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeUsageIDVersion))

	var added int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE table_name = 'requests' AND column_name IN ('usage_id', 'organization_id', 'organization_tier', 'api_key',
	'user_id', 'usage_categories', 'session_id', 'to_ship', 'shipped_at', 'ship_error')`).Scan(&added))
	s.Zero(added)
	s.NotZero(s.writeAsTheRouterBefore())

	s.migrate()
	s.NotEmpty(s.usageIDOf(s.writeAsTheRouterBefore()))
}

func (s *UsageIDMigrationSuite) TestTheMigrationGivesUpOnTheTableRatherThanQueueingWritesBehindIt() {
	// A long read holds the lock the ALTER waits for, and every write after the ALTER would
	// wait behind it.
	reader, err := s.db.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer reader.Rollback() //nolint:errcheck // rolled back below unless the test stopped first
	_, err = reader.ExecContext(s.ctx, "LOCK TABLE requests IN ACCESS SHARE MODE")
	s.Require().NoError(err)

	started := time.Now()
	err = goose.UpToContext(s.ctx, s.db.DB().DB, ".", usageIDVersion)

	s.Require().Error(err)
	s.Contains(err.Error(), "lock timeout")
	s.Less(time.Since(started), 15*time.Second)
	s.Require().NoError(reader.Rollback())
	s.migrate()
	s.NotEmpty(s.usageIDOf(s.writeAsTheRouterBefore()), "the next start applies it")
}
