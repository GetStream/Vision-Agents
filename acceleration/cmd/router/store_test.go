//go:build integration

package main

import (
	"context"
	"io/fs"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// dsnEnvVar is where the store's own suite looks for a Postgres (internal/store/store_test.go).
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

// OpenStoreSuite starts the router's store the way every command does, on a database of its
// own that it empties first.
type OpenStoreSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
}

func TestOpenStoreSuite(t *testing.T) {
	suite.Run(t, new(OpenStoreSuite))
}

func (s *OpenStoreSuite) SetupTest() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "openstore")

	empty, err := store.Open(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	defer empty.Close()
	var database string
	s.Require().NoError(empty.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = empty.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
}

func (s *OpenStoreSuite) TestARouterStartingOnAnEmptyDatabaseHasTheBuiltInsAtTheRevisionsTheyName() {
	revisions := map[string]int{}
	for _, id := range []string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack"} {
		raw, err := fs.ReadFile(providers.FS, id+".yaml")
		s.Require().NoError(err)
		manifest, err := core.ParseManifest(raw)
		s.Require().NoError(err)
		revisions[id] = manifest.Revision
	}
	for range 2 {
		opened, err := openStore(s.ctx, s.settings)
		s.Require().NoError(err)
		s.T().Cleanup(func() { opened.Close() })

		definitions, err := opened.ListConnectorDefinitions(s.ctx, "acme", store.ConnectorDefinitionFilter{})
		s.Require().NoError(err)
		s.Require().Len(definitions, 7, "a restart adds no revision")
		for i, id := range []string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack"} {
			s.Equal(id, definitions[i].ID)
			s.Equal(revisions[id], definitions[i].Revision)
		}
	}
}
