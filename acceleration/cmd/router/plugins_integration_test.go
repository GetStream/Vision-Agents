//go:build integration

package main

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"testing"
	"testing/fstest"

	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// PluginsMigrateSuite runs router plugins migrate as the command builds it, on a database of
// its own. The moves themselves are tested in internal/api (PluginMigrateSuite); this is the
// command's wiring: the store, the keyring, the registry and the report.
type PluginsMigrateSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	// db is the database as a router of the base branch left it: migrated, with the eight
	// built-ins it seeded and none of the seven this branch adds.
	db *store.Store
}

// baseBuiltIns are the built-in connectors before this branch (AI-900).
var baseBuiltIns = []string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack", "slack_bot"}

func TestPluginsMigrateSuite(t *testing.T) {
	suite.Run(t, new(PluginsMigrateSuite))
}

func (s *PluginsMigrateSuite) SetupTest() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "pluginsmigrate")
	s.settings.Auth.KEK = "first-key"
	s.settings.Connectors.Enabled = true

	db, err := store.Open(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.Require().NoError(db.Migrate(s.ctx))
	_, err = db.DB().ExecContext(s.ctx, "DELETE FROM connector_definitions WHERE customer_id = '' AND id NOT IN (?)", bun.In(baseBuiltIns))
	s.Require().NoError(err)
	base := fstest.MapFS{}
	for _, id := range baseBuiltIns {
		raw, err := fs.ReadFile(providers.FS, id+".yaml")
		s.Require().NoError(err)
		base[id+".yaml"] = &fstest.MapFile{Data: raw}
	}
	s.Require().NoError(db.SeedConnectorDefinitions(s.ctx, base))
	s.db = db
}

// TestARunNeitherMigratesNorSeeds: on a database a base router left, a dry run and a real
// run leave the schema version and the built-in connectors as they were. The seven built-ins
// this branch adds are not there, so a login to one would be a skipped row.
func (s *PluginsMigrateSuite) TestARunNeitherMigratesNorSeeds() {
	before := s.schema()

	s.migrate(false, "")
	s.migrate(true, "")

	s.Equal(before, s.schema())
	s.Contains(before, "slack:")
	s.NotContains(before, "gmail:")
}

// TestADatabaseBehindThisBinaryIsRefusedAndLeftAsItIs: with its newest migration not
// applied, a dry run says so, and migrates nothing.
func (s *PluginsMigrateSuite) TestADatabaseBehindThisBinaryIsRefusedAndLeftAsItIs() {
	var newest int64
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT max(version_id) FROM goose_db_version").Scan(&newest))
	_, err := s.db.DB().ExecContext(s.ctx, "DELETE FROM goose_db_version WHERE version_id = ?", newest)
	s.Require().NoError(err)
	// Its schema change is still there, so the row goes back rather than the next setup's
	// migrate running it again.
	s.T().Cleanup(func() {
		_, err := s.db.DB().ExecContext(context.Background(), "INSERT INTO goose_db_version (version_id, is_applied) VALUES (?, true)", newest)
		s.NoError(err)
	})
	before := s.schema()

	var out bytes.Buffer
	err = migratePlugins(s.ctx, s.settings, slog.Default(), pluginmigrate.Options{}, false, &out)

	s.ErrorContains(err, fmt.Sprintf("the database is behind this binary: 1 migrations not applied (the newest %d); deploy this build first", newest))
	s.Empty(out.String())
	s.Equal(before, s.schema(), "nothing was migrated")
}

// schema is the goose versions applied and every built-in connector revision, as text.
func (s *PluginsMigrateSuite) schema() string {
	var out string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT
  (SELECT string_agg(version_id::text, ',' ORDER BY id) FROM goose_db_version WHERE is_applied) || '#' ||
  (SELECT string_agg(id || ':' || revision, ',' ORDER BY id, revision) FROM connector_definitions WHERE customer_id = '')`).Scan(&out))
	return out
}

func (s *PluginsMigrateSuite) TestAnAppWithNoPluginRowsMovesNothing() {
	customer := "app-" + store.NewID()

	dry, real := s.migrate(false, customer), s.migrate(true, customer)

	s.Contains(dry, "0 rows: 0 would move (dry run; pass --apply to write)")
	s.Contains(real, "0 rows: 0 moved")
}

// TestALoginItCannotMoveIsPrintedWithWhy: a login to an MCP server named by its URL, and one
// to a catalog plugin whose config is gone, each print a row saying why, and a real run moves
// neither.
func (s *PluginsMigrateSuite) TestALoginItCannotMoveIsPrintedWithWhy() {
	customer := "app-" + store.NewID()
	db := s.db
	byURL := store.PluginConnection{CustomerID: customer, ConfigID: "config-" + store.NewID(), PluginID: "my-server",
		Status: store.PluginConnected, AccessToken: "synthetic-access"}
	s.Require().NoError(db.UpsertPluginConnection(s.ctx, &byURL))
	orphan := store.PluginConnection{CustomerID: customer, ConfigID: "gone-" + store.NewID(), PluginID: "linear",
		Status: store.PluginConnected, AccessToken: "synthetic-access"}
	s.Require().NoError(db.UpsertPluginConnection(s.ctx, &orphan))

	out := s.migrate(true, customer)

	s.Contains(out, byURL.ID)
	s.Contains(out, "not a catalog plugin")
	s.Contains(out, orphan.ID)
	s.Contains(out, "its agent config is deleted")
	s.Contains(out, "2 rows: 0 moved, 0 already there, 2 skipped")
	s.NotContains(out, "synthetic-access")
}

func (s *PluginsMigrateSuite) migrate(apply bool, customer string) string {
	var out bytes.Buffer
	s.Require().NoError(migratePlugins(s.ctx, s.settings, slog.Default(), pluginmigrate.Options{Customer: customer}, apply, &out))
	return out.String()
}
