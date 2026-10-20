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

// Versions either side of 20261103120000_agent_config_role.sql.
const (
	schemaBeforeAgentRole int64 = 20261102120000
	agentRoleVersion      int64 = 20261103120000
)

// AgentRoleMigrationSuite stops a database of its own short of the role column, writes
// configs the dashboard tagged, then applies the migration, which StoreSuite's migrated and
// truncated database cannot do.
type AgentRoleMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestAgentRoleMigrationSuite(t *testing.T) {
	suite.Run(t, new(AgentRoleMigrationSuite))
}

func (s *AgentRoleMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "rolemig")
	s.ctx = context.Background()
}

func (s *AgentRoleMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "rolemig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeAgentRole))
	s.db = db
}

// The newest live config an app tagged role=agent_moderator takes the role; an older one, a
// test copy, a deleted one and an untagged one do not.
func (s *AgentRoleMigrationSuite) TestTheNewestLiveTaggedConfigOfEachAppTakesTheRole() {
	s.config("older", "app-a", `{"role":"agent_moderator"}`, "2026-10-01", false)
	s.config("newer", "app-a", `{"role":"agent_moderator"}`, "2026-10-02", false)
	s.config("draft", "app-a", `{"role":"agent_moderator","draft_of":"newer"}`, "2026-10-03", false)
	s.config("deleted", "app-b", `{"role":"agent_moderator"}`, "2026-10-01", true)
	s.config("plain", "app-c", `{}`, "2026-10-01", false)
	s.config("other-app", "app-d", `{"role":"agent_moderator"}`, "2026-10-01", false)

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", agentRoleVersion))

	s.Equal(map[string]string{
		"older": "", "newer": RoleAgentModerator, "draft": "", "deleted": "", "plain": "",
		"other-app": RoleAgentModerator,
	}, s.roles())
}

func (s *AgentRoleMigrationSuite) TestTheMigrationComesOffAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", agentRoleVersion))

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeAgentRole))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE table_name = 'agent_configs' AND column_name = 'role'`).Scan(&columns))
	s.Zero(columns)
}

func (s *AgentRoleMigrationSuite) config(id, customer, tags, created string, deleted bool) {
	deletedAt := "NULL"
	if deleted {
		deletedAt = "now()"
	}
	s.exec(s.db, `INSERT INTO agent_configs (id, customer_id, name, tags, created_at, deleted_at)
VALUES ('`+id+`', '`+customer+`', '`+id+`', '`+tags+`', '`+created+`', `+deletedAt+`)`)
}

func (s *AgentRoleMigrationSuite) roles() map[string]string {
	rows, err := s.db.DB().QueryContext(s.ctx, "SELECT id, role FROM agent_configs")
	s.Require().NoError(err)
	defer rows.Close()
	roles := map[string]string{}
	for rows.Next() {
		var id, role string
		s.Require().NoError(rows.Scan(&id, &role))
		roles[id] = role
	}
	s.Require().NoError(rows.Err())
	return roles
}

func (s *AgentRoleMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}
