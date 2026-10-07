//go:build integration

package main

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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
}

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
	db, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	defer db.Close()
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
	s.Require().NoError(migratePlugins(s.ctx, s.settings, slog.Default(), apply, customer, &out))
	return out.String()
}
