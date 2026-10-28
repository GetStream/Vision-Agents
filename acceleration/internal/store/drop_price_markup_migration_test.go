//go:build integration

package store

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// 20261130120000_drop_price_markup.sql.
const dropPriceMarkup int64 = 20261130120000

// DropPriceMarkupMigrationSuite stops a database of its own before the stored prices, with a
// list written under markups, and migrates it past the drop of markup_bps.
type DropPriceMarkupMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestDropPriceMarkupMigrationSuite(t *testing.T) {
	suite.Run(t, new(DropPriceMarkupMigrationSuite))
}

func (s *DropPriceMarkupMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "markupdrop")
	s.ctx = context.Background()
}

func (s *DropPriceMarkupMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "markupdrop")
	s.db = db
	s.exec("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeStoredPrices))
	s.exec(
		`INSERT INTO price_changes (name, kind, reason, source, effective_at, markup_bps, created_by) VALUES
		 ('20261101000000_seed', 'seed', 'The first price list', 'files', now() - interval '10 days', 2000, 'router v1')`,
		`INSERT INTO price_entries (change, modality, provider, model, rates, markup_bps, effective_at) VALUES
		 ('20261101000000_seed', 'llm', 'openai', 'a', '{"input_tokens": {"usd": 1, "per": 1000000}}', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'own', '{"input_tokens": {"usd": 2, "per": 1000000}}', 500, now())`,
	)
}

func (s *DropPriceMarkupMigrationSuite) exec(statements ...string) {
	for _, statement := range statements {
		_, err := s.db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}

func (s *DropPriceMarkupMigrationSuite) count(query string) int {
	var n int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, query).Scan(&n), query)
	return n
}

func (s *DropPriceMarkupMigrationSuite) markupColumns() int {
	return s.count(`SELECT count(*) FROM information_schema.columns
WHERE table_name IN ('price_changes', 'price_entries') AND column_name = 'markup_bps'`)
}

func (s *DropPriceMarkupMigrationSuite) markupChecks() int {
	return s.count(`SELECT count(*) FROM pg_constraint
WHERE conname IN ('price_changes_markup_bps_check', 'price_entries_markup_bps_check')`)
}

// scheduled writes a change made through the API, in force in an hour, of one entry, on the
// schema every migration has run on.
func (s *DropPriceMarkupMigrationSuite) scheduled(name string) {
	change := PriceChange{Name: name, Kind: PriceChangeAPI, Reason: "dearer", Source: "a test",
		EffectiveAt: time.Now().Add(time.Hour), CreatedBy: "operator"}
	_, err := s.db.DB().NewInsert().Model(&change).Exec(s.ctx)
	s.Require().NoError(err)
	entry := entryRow(pricing.Entry{Modality: "llm", Provider: "openai", Model: "b", Region: pricing.RegionUS,
		Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1)}})
	entry.Change = name
	_, err = s.db.DB().NewInsert().Model(&entry).Exec(s.ctx)
	s.Require().NoError(err)
}

// scheduledBeforeRegions is scheduled on a schema the regions migration has not run on, whose
// entries are in price_entries and name no region.
func (s *DropPriceMarkupMigrationSuite) scheduledBeforeRegions(name string) {
	s.exec(`INSERT INTO price_changes (name, kind, reason, source, effective_at, created_by)
VALUES ('`+name+`', 'api', 'dearer', 'a test', now() + interval '1 hour', 'operator')`,
		`INSERT INTO price_entries (change, sku, modality, provider, model, rates, prices, effective_at)
VALUES ('`+name+`', 'llm-openai-b', 'llm', 'openai', 'b', '{"input_tokens": {"usd": 1, "per": 1000000}}',
        '{"input_tokens": {"usd": 1, "per": 1000000}}', now())`)
}

func (s *DropPriceMarkupMigrationSuite) TestTheColumnsAndTheirChecksAreGoneAndEveryPriceStays() {
	s.Require().NoError(s.db.Migrate(s.ctx))

	s.Zero(s.markupColumns())
	s.Zero(s.markupChecks())
	s.Zero(s.count(`SELECT count(*) FROM pg_proc WHERE proname IN ('price_changes_guard', 'price_entries_guard')
AND prosrc LIKE '%markup%'`), "no guard names the column")
	rows, err := s.db.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	prices := map[string]float64{}
	for _, row := range rows {
		prices[row.Model] = row.Prices[pricing.InputTokens].USD
	}
	s.InDeltaMapValues(map[string]float64{"a": 1.2, "own": 2.1}, prices, 1e-12,
		"what the stored prices took from the markups is kept")
}

// TestAChangeIsStillCancelledOnceAndNothingElse: the guard on price_changes runs its new body,
// and the one on price_entries still refuses every update.
func (s *DropPriceMarkupMigrationSuite) TestAChangeIsStillCancelledOnceAndNothingElse() {
	s.Require().NoError(s.db.Migrate(s.ctx))
	s.scheduled("20990101000000_dearer")
	s.scheduled("20990101000001_other")

	_, reason := s.db.DB().ExecContext(s.ctx, "UPDATE price_changes SET reason = 'rewritten' WHERE name = '20990101000001_other'")
	_, both := s.db.DB().ExecContext(s.ctx,
		"UPDATE price_changes SET cancelled_at = now(), source = 'rewritten' WHERE name = '20990101000001_other'")
	change, _, cancelled, err := s.db.CancelPriceChange(s.ctx, "20990101000000_dearer", "not now", "staff-1", "Staff")
	_, again := s.db.DB().ExecContext(s.ctx, "UPDATE price_changes SET cancelled_at = now() WHERE name = '20990101000000_dearer'")
	_, entry := s.db.DB().ExecContext(s.ctx, "UPDATE price_list_entries SET rates = '{}' WHERE model = 'a'")

	s.ErrorContains(reason, "a price change can only be cancelled")
	s.ErrorContains(both, "a price change can only be cancelled")
	s.Require().NoError(err)
	s.True(cancelled)
	s.NotNil(change.CancelledAt)
	s.ErrorContains(again, "a price change can only be cancelled, once")
	s.ErrorContains(entry, "price_entries rows are never changed or deleted")
}

func (s *DropPriceMarkupMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", dropPriceMarkup))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", storedPrices))

	s.Equal(2, s.markupColumns())
	s.Equal(2, s.markupChecks())
	s.Zero(s.count("SELECT count(*) FROM price_changes WHERE markup_bps IS NOT NULL"), "the markups do not come back")
	_, changeOutOfRange := s.db.DB().ExecContext(s.ctx, `INSERT INTO price_changes (name, kind, reason, source, effective_at, markup_bps, created_by)
VALUES ('20990101000001_dear', 'api', 'dear', 'a test', now() + interval '1 hour', 100001, 'operator')`)
	_, entryOutOfRange := s.db.DB().ExecContext(s.ctx, `INSERT INTO price_entries (change, modality, provider, model, sku, markup_bps, effective_at)
VALUES ('20261101000000_seed', 'llm', 'openai', 'c', 'llm-openai-c', 100001, now())`)
	s.ErrorContains(changeOutOfRange, "price_changes_markup_bps_check")
	s.ErrorContains(entryOutOfRange, "price_entries_markup_bps_check")
	s.scheduledBeforeRegions("20990101000000_dearer")
	_, markup := s.db.DB().ExecContext(s.ctx,
		"UPDATE price_changes SET cancelled_at = now(), markup_bps = 1000 WHERE name = '20990101000000_dearer'")
	s.ErrorContains(markup, "a price change can only be cancelled", "the guard compares markup_bps again")
	s.exec("UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'staff-1', cancel_reason = 'not now' " +
		"WHERE name = '20990101000000_dearer'")

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", dropPriceMarkup))
	s.Zero(s.markupColumns())
}
