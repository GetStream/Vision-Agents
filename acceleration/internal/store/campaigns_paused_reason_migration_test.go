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

// Around 20261111121000_campaigns_paused_reason.sql, which follows the newest migration a
// development database already has.
const (
	beforePausedReason int64 = 20261111120000
	pausedReason       int64 = 20261111121000
)

// PausedReasonMigrationSuite stops a database of its own where every development database
// is, writes a campaign there, and migrates it as a router starting does.
type PausedReasonMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestPausedReasonMigrationSuite(t *testing.T) {
	suite.Run(t, new(PausedReasonMigrationSuite))
}

func (s *PausedReasonMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "pausedmig")
	s.ctx = context.Background()
}

func (s *PausedReasonMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "pausedmig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforePausedReason))
	s.db = db
}

func (s *PausedReasonMigrationSuite) TestADatabaseAtTheNewestMigrationUpgradesInPlace() {
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO campaigns (id, customer_id, name, state) VALUES ('old', 'acme', 'may', 'paused')`)
	s.Require().NoError(err)

	s.Require().NoError(s.db.Migrate(s.ctx), "a router starting on it migrates it, with no version older than one applied")

	campaign, err := s.db.Campaign(s.ctx, "acme", "old")
	s.Require().NoError(err)
	s.Equal("paused", campaign.State)
	s.Empty(campaign.PausedReason, "a campaign paused before says no reason")
	s.Require().NoError(s.db.PauseCampaign(s.ctx, "acme", "old", "insufficient_credits"))
	campaign, err = s.db.Campaign(s.ctx, "acme", "old")
	s.Require().NoError(err)
	s.Equal("insufficient_credits", campaign.PausedReason)
}

func (s *PausedReasonMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", pausedReason))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforePausedReason))
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", pausedReason))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT count(*) FROM information_schema.columns WHERE table_name = 'campaigns' AND column_name = 'paused_reason'`).Scan(&columns))
	s.Equal(1, columns)
}
