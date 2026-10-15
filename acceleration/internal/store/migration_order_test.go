//go:build integration

package store

import (
	"context"
	"io/fs"
	"os"
	"strings"
	"testing"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Versions around the custom_models move. 20261015120000 is the file merged in 75214efe
// after 20261016120000 had shipped. 20261018120000 is where that file lives now.
const (
	schemaBeforeCustomModels int64 = 20261014120000
	oldCustomModelsVersion   int64 = 20261015120000
	shippedGooseVersion      int64 = 20261016120000
	indexSwapVersion         int64 = 20261017120000
	customModelsVersion      int64 = 20261018120000
)

// baseCustomModelColumns is information_schema.columns of custom_models, in order, after
// Migrate on c891381c, probed 2026-10-12 on model_router_w9mig_test_w9_mig_test before
// this migration was renamed. The renamed migration must create that table.
var baseCustomModelColumns = []string{
	"id",
	"customer_id",
	"name",
	"base_url",
	"model",
	"api_key_sealed",
	"kek_version",
	"context_window",
	"input_modalities",
	"per_million_input_tokens",
	"per_million_output_tokens",
	"trains_on_data",
	"retention",
	"created_at",
	"updated_at",
}

// baseCustomModelIndexes is pg_indexes (indexname, indexdef) of custom_models, in
// indexname order, after Migrate on c891381c. The renamed migration must create that set.
var baseCustomModelIndexes = []customModelIndex{
	{"custom_models_customer_idx", "CREATE INDEX custom_models_customer_idx ON public.custom_models USING btree (customer_id, created_at DESC, id DESC)"},
	{"custom_models_name_idx", "CREATE UNIQUE INDEX custom_models_name_idx ON public.custom_models USING btree (customer_id, name)"},
	{"custom_models_pkey", "CREATE UNIQUE INDEX custom_models_pkey ON public.custom_models USING btree (id)"},
}

type customModelIndex struct {
	name string
	def  string
}

// oldCustomModels is the Up of 20261015120000_custom_models.sql at 75214efe: what a
// database that already applied that file has run. It is not idempotent.
var oldCustomModels = []string{
	`CREATE TABLE custom_models (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key_sealed BYTEA,
    kek_version INTEGER NOT NULL DEFAULT 0,
    context_window BIGINT NOT NULL DEFAULT 0,
    input_modalities TEXT[] NOT NULL DEFAULT '{}',
    per_million_input_tokens DOUBLE PRECISION NOT NULL DEFAULT 0,
    per_million_output_tokens DOUBLE PRECISION NOT NULL DEFAULT 0,
    trains_on_data TEXT NOT NULL DEFAULT '',
    retention TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)`,
	`CREATE UNIQUE INDEX custom_models_name_idx ON custom_models (customer_id, name)`,
	`CREATE INDEX custom_models_customer_idx ON custom_models (customer_id, created_at DESC, id DESC)`,
}

// MigrationOrderSuite rebuilds goose's version table for each case. StoreSuite migrates
// once and truncates, which cannot represent a database stopped at an older version.
type MigrationOrderSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
}

func TestMigrationOrderSuite(t *testing.T) {
	suite.Run(t, new(MigrationOrderSuite))
}

func (s *MigrationOrderSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "w9mig")
	s.ctx = context.Background()
}

// TestAFreshDatabaseAppliesCustomModelsAtTheLaterVersion: an empty database applies
// 20261018120000, with the columns and indexes the base migration created, and without
// a row for the old version.
func (s *MigrationOrderSuite) TestAFreshDatabaseAppliesCustomModelsAtTheLaterVersion() {
	db := s.open()

	s.Require().NoError(db.Migrate(s.ctx))

	s.Equal(baseCustomModelColumns, s.customModelColumns(db))
	s.Equal(baseCustomModelIndexes, s.customModelIndexes(db))
	applied := s.applied(db)
	s.True(applied[customModelsVersion])
	s.True(applied[indexSwapVersion])
	s.True(applied[shippedGooseVersion])
	s.False(applied[oldCustomModelsVersion])
}

// TestADatabaseAtTheShippedVersionAppliesTheLaterMigrations: goose is at
// 20261016120000 and custom_models was never applied, which is a database that ran the
// build from before 75214efe. Migrate applies the index swap and custom_models.
func (s *MigrationOrderSuite) TestADatabaseAtTheShippedVersionAppliesTheLaterMigrations() {
	db := s.open()
	s.upTo(db, schemaBeforeCustomModels)
	s.applyShippedValidation(db)
	s.False(s.customModelsExists(db))
	s.Equal(shippedGooseVersion, s.maxVersion(db))

	s.Require().NoError(db.Migrate(s.ctx))

	s.True(s.customModelsExists(db))
	s.Equal(baseCustomModelColumns, s.customModelColumns(db))
	applied := s.applied(db)
	s.True(applied[indexSwapVersion])
	s.True(applied[customModelsVersion])
	s.False(applied[oldCustomModelsVersion])
}

// TestADatabaseThatAppliedTheOldVersionKeepsTheTable: 20261015120000 is recorded and
// the table holds a row, then 20261016120000 was applied. UpContext has no file for
// that recorded version; it applies 20261018120000, and the row is still there.
func (s *MigrationOrderSuite) TestADatabaseThatAppliedTheOldVersionKeepsTheTable() {
	db := s.open()
	s.upTo(db, schemaBeforeCustomModels)
	s.exec(db, oldCustomModels...)
	s.recordVersion(db, oldCustomModelsVersion)
	s.exec(db, `INSERT INTO custom_models (id, customer_id, name, base_url, model)
VALUES ('kept', 'cust', 'kept', 'https://models.example.test/v1', 'weights')`)
	s.upTo(db, shippedGooseVersion)
	s.Equal("kept", s.keptName(db))

	s.Require().NoError(db.Migrate(s.ctx))

	s.Equal("kept", s.keptName(db))
	s.Equal(baseCustomModelColumns, s.customModelColumns(db))
	s.Equal(baseCustomModelIndexes, s.customModelIndexes(db))
	applied := s.applied(db)
	s.True(applied[oldCustomModelsVersion], "the recorded version with no file stays")
	s.True(applied[shippedGooseVersion])
	s.True(applied[indexSwapVersion])
	s.True(applied[customModelsVersion])
}

// TestDownOfCustomModelsSucceedsWhenTheTableIsAlreadyGone: Down is DROP TABLE IF EXISTS,
// so a table that is already gone does not stop the version coming off. DownTo the index
// swap rolls this version back even when a later migration exists.
func (s *MigrationOrderSuite) TestDownOfCustomModelsSucceedsWhenTheTableIsAlreadyGone() {
	db := s.open()
	s.Require().NoError(db.Migrate(s.ctx))
	s.exec(db, `DROP TABLE custom_models`)
	s.False(s.customModelsExists(db))

	s.goose()
	s.Require().NoError(goose.DownToContext(s.ctx, db.DB().DB, ".", indexSwapVersion))

	s.False(s.customModelsExists(db))
	applied := s.applied(db)
	s.False(applied[customModelsVersion])
	s.True(applied[indexSwapVersion])
}

func (s *MigrationOrderSuite) open() *Store {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		s.NoError(db.Close())
	})
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().NotEqual("model_router", name)
	s.Require().NotEqual("model_router_test", name)
	s.Require().Contains(name, "w9mig")
	s.T().Logf("database=%s", name)
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	return db
}

func (s *MigrationOrderSuite) goose() {
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
}

func (s *MigrationOrderSuite) upTo(db *Store, version int64) {
	s.goose()
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", version))
}

func (s *MigrationOrderSuite) applyShippedValidation(db *Store) {
	body, err := fs.ReadFile(migrations.FS, "20261016120000_connector_connection_validations.sql")
	s.Require().NoError(err)
	s.exec(db, upStatements(string(body))...)
	s.recordVersion(db, shippedGooseVersion)
}

func (s *MigrationOrderSuite) recordVersion(db *Store, version int64) {
	_, err := db.DB().ExecContext(s.ctx, `INSERT INTO goose_db_version (version_id, is_applied) VALUES (?, true)`, version)
	s.Require().NoError(err)
}

func (s *MigrationOrderSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err)
	}
}

func (s *MigrationOrderSuite) applied(db *Store) map[int64]bool {
	rows, err := db.DB().QueryContext(s.ctx, `SELECT version_id FROM goose_db_version WHERE is_applied`)
	s.Require().NoError(err)
	defer rows.Close()
	got := map[int64]bool{}
	for rows.Next() {
		var version int64
		s.Require().NoError(rows.Scan(&version))
		got[version] = true
	}
	s.Require().NoError(rows.Err())
	return got
}

func (s *MigrationOrderSuite) maxVersion(db *Store) int64 {
	var version int64
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, `SELECT max(version_id) FROM goose_db_version WHERE is_applied`).Scan(&version))
	return version
}

func (s *MigrationOrderSuite) customModelsExists(db *Store) bool {
	var exists bool
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, `SELECT to_regclass('public.custom_models') IS NOT NULL`).Scan(&exists))
	return exists
}

func (s *MigrationOrderSuite) customModelIndexes(db *Store) []customModelIndex {
	rows, err := db.DB().QueryContext(s.ctx, `
SELECT indexname, indexdef FROM pg_indexes
WHERE schemaname = 'public' AND tablename = 'custom_models'
ORDER BY indexname`)
	s.Require().NoError(err)
	defer rows.Close()
	var indexes []customModelIndex
	for rows.Next() {
		var index customModelIndex
		s.Require().NoError(rows.Scan(&index.name, &index.def))
		indexes = append(indexes, index)
	}
	s.Require().NoError(rows.Err())
	return indexes
}

func (s *MigrationOrderSuite) customModelColumns(db *Store) []string {
	rows, err := db.DB().QueryContext(s.ctx, `
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'custom_models'
ORDER BY ordinal_position`)
	s.Require().NoError(err)
	defer rows.Close()
	var columns []string
	for rows.Next() {
		var column string
		s.Require().NoError(rows.Scan(&column))
		columns = append(columns, column)
	}
	s.Require().NoError(rows.Err())
	return columns
}

func (s *MigrationOrderSuite) keptName(db *Store) string {
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, `SELECT name FROM custom_models WHERE id = 'kept'`).Scan(&name))
	return name
}

// upStatements is the Up half of a goose SQL file. Comments go first: one of them contains
// a semicolon, and splitting on it would run the rest of the sentence as SQL.
func upStatements(body string) []string {
	up, _, _ := strings.Cut(body, "-- +goose Down")
	var statements []string
	for _, part := range strings.Split(stripSQLComments(up), ";") {
		if strings.TrimSpace(part) == "" {
			continue
		}
		statements = append(statements, strings.TrimSpace(part))
	}
	return statements
}

func stripSQLComments(statement string) string {
	var b strings.Builder
	for _, line := range strings.Split(statement, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "--") {
			continue
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}

// The versions around the audit action tool_reapproved (AI-1305): the newest migration of its
// base 9a74ce52, and its own.
const (
	beforeToolReapproved  int64 = 20261101120000
	toolReapprovedVersion int64 = 20261101130000
)

// TestTheAuditCheckKeepsEveryRowAndTakesToolReapproved: a database holding a row of every
// action the CHECK allowed migrates, keeps them all, and takes tool_reapproved and nothing
// else new. Its Down keeps the tool_reapproved rows.
func (s *MigrationOrderSuite) TestTheAuditCheckKeepsEveryRowAndTakesToolReapproved() {
	db := s.open()
	s.upTo(db, beforeToolReapproved)
	audit := func(action string) error {
		_, err := db.DB().ExecContext(s.ctx, "INSERT INTO connector_audit (id, customer_id, connection_id, connector_id, "+
			"owner_type, action, created_at) VALUES (?, 'app', 'connection', 'acme', 'app', ?, now())", newID(), action)
		return err
	}
	for _, action := range []string{"grant_created", "grant_refreshed", "grant_revoked", "proxy_call", "token_export"} {
		s.Require().NoError(audit(action), action)
	}

	s.upTo(db, toolReapprovedVersion)

	s.Equal(5, s.auditRows(db))
	s.Require().NoError(audit(AuditToolReapproved))
	s.Require().ErrorContains(audit("tool_approved"), "connector_audit_action_check")
	s.goose()
	s.Require().NoError(goose.DownToContext(s.ctx, db.DB().DB, ".", beforeToolReapproved))
	s.Equal(6, s.auditRows(db))
	s.Require().ErrorContains(audit(AuditToolReapproved), "connector_audit_action_check")
}

func (s *MigrationOrderSuite) auditRows(db *Store) int {
	var rows int
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_audit").Scan(&rows))
	return rows
}
