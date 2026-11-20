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

// Versions either side of 20270103090000_agent_config_tool_groups.sql.
const (
	schemaBeforeToolGroups int64 = 20270102090000
	toolGroupsVersion      int64 = 20270103090000
)

// ToolGroupsMigrationSuite stops a database of its own short of tool groups, writes configs
// with the agent-wide tool settings, then applies the migration.
type ToolGroupsMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestToolGroupsMigrationSuite(t *testing.T) {
	suite.Run(t, new(ToolGroupsMigrationSuite))
}

func (s *ToolGroupsMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "toolgroupsmig")
	s.ctx = context.Background()
}

func (s *ToolGroupsMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "toolgroupsmig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeToolGroups))
	s.db = db
}

// connectors is the stored connector bindings of the config called id, read by its column
// rather than through the model, which carries columns later migrations add.
func (s *ToolGroupsMigrationSuite) connectors(id string) string {
	var connectors string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT connectors::text FROM agent_configs WHERE id = ?`, id).Scan(&connectors))
	return connectors
}

func (s *ToolGroupsMigrationSuite) TestAProgressiveConfigsConnectorsAreSummarisedAndKeepTheirRetention() {
	s.exec(s.db, `INSERT INTO agent_configs (id, customer_id, name, progressive_tools, tool_output_retention, connectors)
VALUES ('older', 'app', 'support', true, '7d',
        '[{"name": "linear", "connector_id": "c1"}, {"name": "sentry", "connector_id": "c2", "output_retention": "none"}]')`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolGroupsVersion))

	s.JSONEq(`[
		{"name": "linear", "connector_id": "c1", "loading": "summary", "output_retention": "7d"},
		{"name": "sentry", "connector_id": "c2", "loading": "summary", "output_retention": "none"}
	]`, s.connectors("older"), "a binding that says its own retention keeps it, and the order stays")
}

func (s *ToolGroupsMigrationSuite) TestAConfigWithoutTheSettingsIsLeftAsItWas() {
	s.exec(s.db, `INSERT INTO agent_configs (id, customer_id, name, connectors)
VALUES ('plain', 'app', 'support', '[{"name": "linear", "connector_id": "c1"}]')`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolGroupsVersion))

	s.JSONEq(`[{"name": "linear", "connector_id": "c1"}]`, s.connectors("plain"))
	var groups *string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT tool_groups::text FROM agent_configs WHERE id = 'plain'`).Scan(&groups))
	s.Nil(groups, "a config from before names no function")
}

func (s *ToolGroupsMigrationSuite) TestAMigrationThatStoppedHalfwayRunsAgain() {
	s.exec(s.db, `ALTER TABLE agent_configs ADD COLUMN tool_groups JSONB`)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolGroupsVersion))
}

func (s *ToolGroupsMigrationSuite) TestTheMigrationComesOffAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", toolGroupsVersion))

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeToolGroups))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE table_name = 'agent_configs' AND column_name = 'tool_groups'`).Scan(&columns))
	s.Zero(columns)
}

func (s *ToolGroupsMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}
