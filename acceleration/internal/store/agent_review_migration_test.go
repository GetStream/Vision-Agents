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

// Versions either side of 20261105150000_agent_config_review.sql.
const (
	schemaBeforeAgentReview int64 = 20261105140000
	agentReviewVersion      int64 = 20261105150000
)

// AgentReviewMigrationSuite stops a database of its own short of the review column, writes
// a config, then applies the migration, which StoreSuite's migrated database cannot do.
type AgentReviewMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestAgentReviewMigrationSuite(t *testing.T) {
	suite.Run(t, new(AgentReviewMigrationSuite))
}

func (s *AgentReviewMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "reviewmig")
	s.ctx = context.Background()
}

func (s *AgentReviewMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "reviewmig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeAgentReview))
	s.db = db
}

// A config stored before the column existed is reviewed by the default guide.
func (s *AgentReviewMigrationSuite) TestAConfigFromBeforeTheMigrationHasNoReviewInstructions() {
	s.exec(s.db, `INSERT INTO agent_configs (id, customer_id, name) VALUES ('older', 'app', 'support')`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", agentReviewVersion))

	// Read by its column rather than through the model, which carries columns later
	// migrations add.
	var instructions string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT review_instructions FROM agent_configs WHERE id = 'older'`).Scan(&instructions))
	s.Empty(instructions)
}

func (s *AgentReviewMigrationSuite) TestTheMigrationComesOffAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", agentReviewVersion))

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeAgentReview))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE table_name = 'agent_configs' AND column_name = 'review_instructions'`).Scan(&columns))
	s.Zero(columns)
}

func (s *AgentReviewMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}
