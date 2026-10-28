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

// Around 20261203120000_agent_config_visible_tools_unset.sql.
const (
	beforeVisibleToolsUnset int64 = 20261202120000
	visibleToolsUnset       int64 = 20261203120000
)

// VisibleToolsMigrationSuite stops a database of its own before visible_tools could tell
// "not set" from "none", writes configs there, and migrates it as a router starting does.
type VisibleToolsMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestVisibleToolsMigrationSuite(t *testing.T) {
	suite.Run(t, new(VisibleToolsMigrationSuite))
}

func (s *VisibleToolsMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "visiblemig")
	s.ctx = context.Background()
}

func (s *VisibleToolsMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "visiblemig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeVisibleToolsUnset))
	s.db = db
}

func (s *VisibleToolsMigrationSuite) TestAConfigThatNeverSetVisibleToolsIsLeftUnset() {
	unset := AgentConfig{CustomerID: "acme", Name: "unset", VisibleTools: []string{}}
	named := AgentConfig{CustomerID: "acme", Name: "named", VisibleTools: []string{"lookup_order"}}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &unset), "before, every config stored a list")
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &named))

	s.Require().NoError(s.db.Migrate(s.ctx))

	read, err := s.db.AgentConfig(s.ctx, "acme", unset.ID)
	s.Require().NoError(err)
	s.Nil(read.VisibleTools, "the empty list stored before meant search and web_search")
	read, err = s.db.AgentConfig(s.ctx, "acme", named.ID)
	s.Require().NoError(err)
	s.Equal([]string{"lookup_order"}, read.VisibleTools)
}

func (s *VisibleToolsMigrationSuite) TestAConfigShowingNoToolIsReadBackAsShowingNone() {
	s.Require().NoError(s.db.Migrate(s.ctx))
	unset := AgentConfig{CustomerID: "acme", Name: "unset"}
	none := AgentConfig{CustomerID: "acme", Name: "none", VisibleTools: []string{}}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &unset))
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &none))

	readUnset, err := s.db.AgentConfig(s.ctx, "acme", unset.ID)
	s.Require().NoError(err)
	readNone, err := s.db.AgentConfig(s.ctx, "acme", none.ID)
	s.Require().NoError(err)

	s.Nil(readUnset.VisibleTools)
	s.NotNil(readNone.VisibleTools)
	s.Empty(readNone.VisibleTools)
}

func (s *VisibleToolsMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", visibleToolsUnset))
	unset := AgentConfig{CustomerID: "acme", Name: "unset"}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &unset))

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeVisibleToolsUnset))
	var stored string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT visible_tools::text FROM agent_configs WHERE id = ?`, unset.ID).Scan(&stored))
	s.Equal("[]", stored, "down writes the empty list a router built before reads as not set")
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", visibleToolsUnset))

	read, err := s.db.AgentConfig(s.ctx, "acme", unset.ID)
	s.Require().NoError(err)
	s.Nil(read.VisibleTools)
}
