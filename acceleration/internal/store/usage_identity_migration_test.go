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

// Around 20261111120000_usage_identity.sql.
const (
	beforeUsageIdentity int64 = 20261110140000
	usageIdentity       int64 = 20261111120000
)

// UsageIdentityMigrationSuite stops a database of its own before the usage identity columns,
// writes the rows a deployment already has, and applies them.
type UsageIdentityMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestUsageIdentityMigrationSuite(t *testing.T) {
	suite.Run(t, new(UsageIdentityMigrationSuite))
}

func (s *UsageIdentityMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "identitymig")
	s.ctx = context.Background()
}

func (s *UsageIdentityMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "identitymig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeUsageIdentity))
	s.db = db
}

func (s *UsageIdentityMigrationSuite) TestCampaignsAndTasksFromBeforeReadBackTheirOrigin() {
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO campaigns (id, customer_id, name) VALUES ('old', 'acme', 'may')`)
	s.Require().NoError(err)
	_, err = s.db.DB().ExecContext(s.ctx, `INSERT INTO scheduled_tasks (id, customer_id, user_id, title, instructions, start_local, timezone)
		VALUES ('old', 'acme', 'u1', 'Stretch', 'Remind them.', now(), 'UTC')`)
	s.Require().NoError(err)
	before := s.files()

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", usageIdentity))
	s.Equal(before, s.files(), "no table was rewritten")
	_, err = s.db.DB().ExecContext(s.ctx, `INSERT INTO campaigns (id, customer_id, name) VALUES ('new', 'acme', 'june')`)
	s.Require().NoError(err)
	_, err = s.db.DB().ExecContext(s.ctx, `INSERT INTO scheduled_tasks (id, customer_id, user_id, title, instructions, start_local, timezone)
		VALUES ('new', 'acme', 'u1', 'Stretch', 'Remind them.', now(), 'UTC')`)
	s.Require().NoError(err)

	s.Equal(map[string]string{"old": "job:campaign", "new": ""}, s.keys("campaigns"))
	s.Equal(map[string]string{"old": "job:scheduled_task", "new": ""}, s.keys("scheduled_tasks"))
}

func (s *UsageIdentityMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", usageIdentity))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeUsageIdentity))
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", usageIdentity))

	var tables int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT count(*) FROM information_schema.tables WHERE table_name = 'organization_tiers'`).Scan(&tables))
	s.Equal(1, tables)
}

// files is the storage each altered table is in, which a rewrite replaces.
func (s *UsageIdentityMigrationSuite) files() map[string]int64 {
	files := map[string]int64{}
	for _, table := range []string{"campaigns", "scheduled_tasks", "agent_sessions"} {
		var file int64
		s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
			"SELECT relfilenode FROM pg_class WHERE relname = ?", table).Scan(&file))
		files[table] = file
	}
	return files
}

// keys is the api_key of every row of a table, by id.
func (s *UsageIdentityMigrationSuite) keys(table string) map[string]string {
	rows, err := s.db.DB().QueryContext(s.ctx, "SELECT id, api_key FROM "+table)
	s.Require().NoError(err)
	defer rows.Close()
	keys := map[string]string{}
	for rows.Next() {
		var id, key string
		s.Require().NoError(rows.Scan(&id, &key))
		keys[id] = key
	}
	s.Require().NoError(rows.Err())
	return keys
}
