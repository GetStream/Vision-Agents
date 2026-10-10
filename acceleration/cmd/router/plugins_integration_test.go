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

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
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
	// An app of its own: the runs read no other test's rows. Opening the database, the drift
	// check and the seeding the old command did are the same whatever app is asked for.
	customer := "app-" + store.NewID()
	before := s.schema()

	s.migrate(false, customer)
	s.migrate(true, customer)

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

// TestADatabaseNewerThanThisBinaryIsRefusedAndLeftAsItIs: a build this one does not know
// applied a migration, so the command says so and writes nothing.
func (s *PluginsMigrateSuite) TestADatabaseNewerThanThisBinaryIsRefusedAndLeftAsItIs() {
	// A version past any this repository will carry: migration names are timestamps.
	const future = int64(99991231235959)
	_, err := s.db.DB().ExecContext(s.ctx, "INSERT INTO goose_db_version (version_id, is_applied) VALUES (?, true)", future)
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		_, err := s.db.DB().ExecContext(context.Background(), "DELETE FROM goose_db_version WHERE version_id = ?", future)
		s.NoError(err)
	})
	before := s.schema()

	var out bytes.Buffer
	err = migratePlugins(s.ctx, s.settings, slog.Default(), pluginmigrate.Options{}, false, &out)

	s.ErrorContains(err, fmt.Sprintf("the database is newer than this binary: 1 migrations applied that it does not carry (the newest %d)", future))
	s.Empty(out.String())
	s.Equal(before, s.schema())
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

// TestADryRunLeavesAClientSecretUnderAnOlderKeyAsItIs: the app's GitHub client record was sealed
// under key version 1, and the command's keyring writes version 2. The dry run opens it to
// compare it with the plugin's, and seals nothing again: a router still on version 1 could not
// open what it wrote.
func (s *PluginsMigrateSuite) TestADryRunLeavesAClientSecretUnderAnOlderKeyAsItIs() {
	customer := "app-" + store.NewID()
	older, err := loadKeyring(s.settings, "the test needs")
	s.Require().NoError(err)
	s.T().Setenv(authKEKEnvVar+"_V2", "second-key")
	s.T().Setenv(authKEKVersionEnvVar, "2")
	sealed, version, err := api.SealConnectorOAuthClientSecret(older, customer, "github", "app-secret")
	s.Require().NoError(err)
	s.Require().Equal(1, version)
	record := store.ConnectorOAuthClient{CustomerID: customer, ConnectorID: "github", Registration: core.ClientCustomer,
		ClientID: "app-client", SecretSealed: sealed, KEKVersion: version, SigningSecretSealed: []byte{}}
	_, err = s.db.PutConnectorOAuthClient(s.ctx, &record)
	s.Require().NoError(err)
	config := "config-" + store.NewID()
	pluginSecret, err := session.SealPluginClientSecret(older, plugins.Owner{CustomerID: customer, ConfigID: config}, "github", "app-secret")
	s.Require().NoError(err)
	s.Require().NoError(s.db.SavePluginClient(s.ctx, &store.PluginClient{CustomerID: customer, ConfigID: config,
		PluginID: "github", ClientID: "app-client", SecretSealed: pluginSecret, SecretKEKVersion: 1}))

	out := s.migrate(false, customer)

	s.Contains(out, "client app-client", "the record was opened and compared")
	s.Contains(out, "1 rows: 0 would move (dry run; pass --apply to write), 1 already there, 0 skipped")
	after, err := s.db.ConnectorOAuthClient(s.ctx, customer, "github")
	s.Require().NoError(err)
	s.Equal(1, after.KEKVersion)
	s.True(bytes.Equal(sealed, after.SecretSealed), "the sealed secret is the one written under version 1")
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

// TestPluginNamesTheRowsRead: --plugin goes through runPlugins to the move, so a login of
// another plugin is not read and does not appear in the report.
func (s *PluginsMigrateSuite) TestPluginNamesTheRowsRead() {
	customer := "app-" + store.NewID()
	named := store.PluginConnection{CustomerID: customer, ConfigID: "gone-" + store.NewID(), PluginID: "linear",
		Status: store.PluginConnected, AccessToken: "synthetic-access"}
	s.Require().NoError(s.db.UpsertPluginConnection(s.ctx, &named))
	other := store.PluginConnection{CustomerID: customer, ConfigID: "gone-" + store.NewID(), PluginID: "github",
		Status: store.PluginConnected, AccessToken: "synthetic-access"}
	s.Require().NoError(s.db.UpsertPluginConnection(s.ctx, &other))

	out := s.runCommand("migrate", "--customer", customer, "--plugin", "linear")

	s.Contains(out, named.ID)
	s.NotContains(out, other.ID)
	s.Contains(out, "1 rows:")
}

// runCommand runs router plugins with args and returns what it printed.
func (s *PluginsMigrateSuite) runCommand(args ...string) string {
	reader, writer, err := os.Pipe()
	s.Require().NoError(err)
	stdout := os.Stdout
	os.Stdout = writer
	runErr := runPlugins(args, s.settings, slog.Default())
	os.Stdout = stdout
	s.Require().NoError(writer.Close())
	var out bytes.Buffer
	_, err = out.ReadFrom(reader)
	s.Require().NoError(err)
	s.Require().NoError(runErr)
	return out.String()
}
