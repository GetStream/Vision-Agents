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

// Versions either side of 20261107120000_tool_output_retention.sql.
const (
	schemaBeforeToolOutputRetention int64 = 20261106120000
	toolOutputRetentionVersion      int64 = 20261107120000
)

// ToolOutputRetentionMigrationSuite stops a database of its own short of the retention
// columns, writes a config and a tool item, then applies the migration.
type ToolOutputRetentionMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestToolOutputRetentionMigrationSuite(t *testing.T) {
	suite.Run(t, new(ToolOutputRetentionMigrationSuite))
}

func (s *ToolOutputRetentionMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "retentionmig")
	s.ctx = context.Background()
}

func (s *ToolOutputRetentionMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "retentionmig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeToolOutputRetention))
	s.db = db
}

func (s *ToolOutputRetentionMigrationSuite) TestWhatWasStoredBeforeIsKeptAsItWas() {
	s.exec(s.db,
		`INSERT INTO agent_configs (id, customer_id, name) VALUES ('older', 'app', 'support')`,
		`INSERT INTO agent_sessions (id, customer_id, state) VALUES ('one', 'app', '{}')`,
		`INSERT INTO agent_responses (id, session_id, customer_id) VALUES ('first', 'one', 'app')`,
		`INSERT INTO agent_response_items (response_id, ordinal, session_id, kind, text) VALUES ('first', 0, 'one', 'tool_result', 'kept')`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolOutputRetentionVersion))

	// Read by its column rather than through the model, which carries columns later
	// migrations add.
	var retention string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT tool_output_retention FROM agent_configs WHERE id = 'older'`).Scan(&retention))
	s.Empty(retention, "a config from before keeps tool output as long as it always did")
	var pending int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT count(*) FROM agent_response_items WHERE redact_at IS NOT NULL`).Scan(&pending))
	s.Zero(pending, "no item from before is due to be redacted")
}

func (s *ToolOutputRetentionMigrationSuite) TestTheIndexIsBuiltValidAndHoldsOnlyWhatIsPending() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolOutputRetentionVersion))

	var valid bool
	var definition string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `
SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) FROM pg_index AS i
WHERE i.indexrelid = 'agent_response_items_redact_idx'::regclass`).Scan(&valid, &definition))
	s.True(valid)
	s.Contains(definition, "WHERE (redact_at IS NOT NULL)")
}

func (s *ToolOutputRetentionMigrationSuite) TestAMigrationThatStoppedHalfwayRunsAgain() {
	s.exec(s.db, `ALTER TABLE agent_configs ADD COLUMN tool_output_retention TEXT NOT NULL DEFAULT ''`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolOutputRetentionVersion))
}

func (s *ToolOutputRetentionMigrationSuite) TestTheMigrationComesOffAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolOutputRetentionVersion))

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeToolOutputRetention))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE (table_name = 'agent_configs' AND column_name = 'tool_output_retention')
   OR (table_name = 'agent_response_items' AND column_name = 'redact_at')`).Scan(&columns))
	s.Zero(columns)
}

func (s *ToolOutputRetentionMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}
