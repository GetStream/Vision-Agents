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

// Versions either side of 20261214130000_data_changes_in_flight.sql.
const (
	schemaBeforeDataChangesInFlight int64 = 20261213090000
	dataChangesInFlightVersion      int64 = 20261214130000
)

// DataChangesInFlightMigrationSuite stops a database of its own short of the lock the
// change trigger takes, then applies the migration and takes it off again.
type DataChangesInFlightMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestDataChangesInFlightMigrationSuite(t *testing.T) {
	suite.Run(t, new(DataChangesInFlightMigrationSuite))
}

func (s *DataChangesInFlightMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "inflightmig")
	s.ctx = context.Background()
}

func (s *DataChangesInFlightMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "inflightmig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeDataChangesInFlight))
	s.db = db
}

func (s *DataChangesInFlightMigrationSuite) TestTheMigrationComesOffAgain() {
	before := s.trigger()

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", dataChangesInFlightVersion))
	s.Contains(s.trigger(), "pg_advisory_xact_lock_shared")
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeDataChangesInFlight))

	s.Equal(before, s.trigger(), "the function is the one from before")
}

// trigger is the source of the function that records a change.
func (s *DataChangesInFlightMigrationSuite) trigger() string {
	var source string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT prosrc FROM pg_proc WHERE proname = 'record_data_change'").Scan(&source))
	return source
}
