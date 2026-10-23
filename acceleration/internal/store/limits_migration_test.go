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
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Around 20261112120000_limits.sql. beforeLimits is the migration just before it, where a
// database that already runs the previous release stands.
const (
	beforeLimits      int64 = 20261111130000
	limitsTable       int64 = 20261112120000
	limitGrantsColumn int64 = 20261112121000
)

// LimitsMigrationSuite stops a database of its own at the migration before the limits table,
// writes budgets into its policies as the policy API did, and migrates it as a router starting
// does.
type LimitsMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestLimitsMigrationSuite(t *testing.T) {
	suite.Run(t, new(LimitsMigrationSuite))
}

func (s *LimitsMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "limitsmig")
	s.ctx = context.Background()
}

func (s *LimitsMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "limitsmig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeLimits))
	s.db = db
}

// budgets writes, as the policy API did: an organization's daily budget beside its tags, the
// weekly budget of an app seen in that organization, and the monthly budget of an app seen in
// none.
func (s *LimitsMigrationSuite) budgets() {
	for _, statement := range []string{
		`INSERT INTO app_organizations (app_id, organization_id, seen_at) VALUES ('app-in', 'org-1', now())`,
		`INSERT INTO policies (scope, scope_id, document, updated_at) VALUES
			('organization', 'org-1', '{"budget": {"limit_micros": 5000000, "interval": "daily"}, "tags": {"team": "a"}}', now()),
			('app', 'app-in', '{"budget": {"limit_micros": 2000000, "interval": "weekly"}}', now()),
			('app', 'app-alone', '{"budget": {"limit_micros": 1000000, "interval": "monthly"}, "prompt_injection": true}', now()),
			('app', 'app-none', '{"prompt_injection": true}', now())`,
	} {
		_, err := s.db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err)
	}
}

func (s *LimitsMigrationSuite) TestADatabaseAtTheNewestMigrationMovesItsBudgetsIntoLimits() {
	s.budgets()

	s.Require().NoError(s.db.Migrate(s.ctx), "a router starting on it migrates it, with no version older than one applied")

	limits, err := s.db.LimitsFor(s.ctx, []string{"org-1", "app-alone"})
	s.Require().NoError(err)
	s.Require().Len(limits, 3)
	moved := map[string]Limit{}
	for _, limit := range limits {
		moved[string(limit.Subject)+":"+limit.AppID] = limit
		s.Equal("migration", limit.UpdatedBy)
		s.Equal("moved from the policy budget", limit.Note)
	}
	s.Equal(Limit{Subject: usage.SubjectOrganization, Level: LevelOrganization, OrganizationID: "org-1", Interval: usage.Daily, LimitMicros: 5_000_000},
		trimmed(moved["organization:"]))
	s.Equal(Limit{Subject: usage.SubjectApp, Level: LevelApp, OrganizationID: "org-1", AppID: "app-in", Interval: usage.Weekly, LimitMicros: 2_000_000},
		trimmed(moved["app:app-in"]), "under the organization the app was seen in")
	s.Equal(Limit{Subject: usage.SubjectApp, Level: LevelApp, OrganizationID: "app-alone", AppID: "app-alone", Interval: usage.Monthly, LimitMicros: 1_000_000},
		trimmed(moved["app:app-alone"]), "an app seen in no organization is its own")

	var budgets int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM policies WHERE document ? 'budget'`).Scan(&budgets))
	s.Zero(budgets, "no document keeps a budget")
	document, err := s.db.Policy(s.ctx, ScopeOrganization, "org-1")
	s.Require().NoError(err)
	s.Equal(map[string]string{"team": "a"}, document.Tags, "the rest of the document stays")
}

func (s *LimitsMigrationSuite) TestAnHourlyBudgetStopsTheMigration() {
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO policies (scope, scope_id, document, updated_at) VALUES
		('app', 'app-hourly', '{"budget": {"limit_micros": 1000000, "interval": "hourly"}}', now())`)
	s.Require().NoError(err)

	err = goose.UpToContext(s.ctx, s.db.DB().DB, ".", limitsTable)

	s.Require().ErrorContains(err, "an hourly budget exists")
	version, err := goose.GetDBVersionContext(s.ctx, s.db.DB().DB)
	s.Require().NoError(err)
	s.Equal(beforeLimits, version, "nothing of it was applied")
	var document string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT document::text FROM policies WHERE scope_id = 'app-hourly'`).Scan(&document))
	s.Contains(document, "hourly", "the budget is where it was")
}

func (s *LimitsMigrationSuite) TestComingDownPutsTheBudgetsBack() {
	s.budgets()
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", limitsTable))
	// A limit made since, on a scope with no policy, comes back as that scope's budget too.
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO limits (subject, level, organization_id, app_id, interval, limit_micros, updated_by)
		VALUES ('app', 'app', 'org-1', 'app-new', 'daily', 3000000, 'someone'),
		       ('api_key', 'app', 'org-1', 'app-new', 'daily', 1, 'someone')`)
	s.Require().NoError(err)

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeLimits))

	for scope, want := range map[string]string{
		"org-1":     `{"tags": {"team": "a"}, "budget": {"interval": "daily", "limit_micros": 5000000}}`,
		"app-in":    `{"budget": {"interval": "weekly", "limit_micros": 2000000}}`,
		"app-alone": `{"budget": {"interval": "monthly", "limit_micros": 1000000}, "prompt_injection": true}`,
		"app-new":   `{"budget": {"interval": "daily", "limit_micros": 3000000}}`,
		"app-none":  `{"prompt_injection": true}`,
	} {
		var document string
		s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT document::text FROM policies WHERE scope_id = ?`, scope).Scan(&document))
		s.JSONEq(want, document, scope)
	}
	var tables int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT count(*) FROM information_schema.tables WHERE table_name IN ('limits', 'usage_resets', 'limit_grants')`).Scan(&tables))
	s.Zero(tables)
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", limitsTable), "and it applies again")
}

func (s *LimitsMigrationSuite) TestARowFromBeforeTheGrantsColumnNamesNoGrant() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", limitsTable))
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO requests (modality, customer_id, provider, model, started_at, success)
		VALUES ('llm', 'acme', 'openai', 'm', now(), true)`)
	s.Require().NoError(err)

	s.Require().NoError(s.db.Migrate(s.ctx))

	var grants string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT limit_grants::text FROM requests WHERE customer_id = 'acme'`).Scan(&grants))
	s.Equal("{}", grants)
	var valid bool
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
		WHERE c.relname = 'requests_limit_grants_idx'`).Scan(&valid))
	s.True(valid)
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", limitsTable))
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", limitGrantsColumn), "and it applies again")
}

// trimmed is a limit without what the database chose for it.
func trimmed(limit Limit) Limit {
	return Limit{Subject: limit.Subject, Level: limit.Level, OrganizationID: limit.OrganizationID, AppID: limit.AppID, Interval: limit.Interval, LimitMicros: limit.LimitMicros}
}
